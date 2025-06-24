"""
OMG Transformer: Lark AST transformer for OMG DSL.

This module provides the DslTransformer class that converts Lark parse trees
into OMG AST structures. It handles the transformation of grammar elements
like imports, rules, patterns, and resolver configurations into typed AST nodes.
"""

from typing import Optional

from lark import Transformer, v_args

from dsl import omg_ast as ast


@v_args(inline=True)  # This simplifies most method signatures
class DslTransformer(Transformer):  # pylint: disable=too-many-public-methods
    """
    Transformer that converts Lark parse trees into OMG AST structures.

    Handles conversion of grammar elements including imports, rules, patterns,
    quantifiers, and resolver configurations into properly typed AST nodes.
    """

    def __init__(self, dsl_file_path: Optional[str] = None):
        super().__init__()
        self.dsl_file_path = dsl_file_path
        self._manual_resolver_mode = False

    def set_manual_resolver_mode(self, enabled: bool = True):
        """
        Set manual resolver mode to allow dotted rules without immediate
        resolver requirements.
        """
        self._manual_resolver_mode = enabled

    def root(self, *children):
        """Transform root node with version, imports, rules, and default resolver."""
        version = children[0]
        imports = []
        rules = []
        default_resolver = None
        for item in children[1:]:
            if isinstance(item, ast.Import):
                imports.append(item)
            elif isinstance(item, ast.ResolverDefault):
                default_resolver = item
            else:
                rules.append(item)
        return ast.Root(
            version=version,
            imports=tuple(imports),
            rules=tuple(rules),
            default_resolver=default_resolver,
            dsl_file_path=self.dsl_file_path,
        )

    def version_stmt(self, version_token):
        """Transform version statement."""
        return ast.Version(value=str(version_token))

    def import_stmt(self, path, alias, opts=None):
        """Transform import statement with path, alias, and optional flags."""
        flags = opts if opts is not None else []
        return ast.Import(path=path[1:-1], alias=alias, flags=tuple(flags))

    def import_opts(self, *flags):
        """Transform import options list."""
        return list(flags)

    def import_flag(self, token):
        """Transform import flag token."""
        return str(token)

    def rule_def(self, name, _equal, pattern, uses=None, _newline=None):
        """Transform rule definition with validation and resolver config."""

        def contains_list_match(node):
            if isinstance(node, ast.ListMatch):
                return True
            if isinstance(
                node, (ast.Concat, ast.Alt, ast.NamedCapture, ast.Quantified)
            ):
                children = (
                    getattr(node, "parts", None)
                    or getattr(node, "options", None)
                    or [getattr(node, "expr", None)]
                )
                return any(contains_list_match(child) for child in children if child)
            return False

        if not contains_list_match(pattern):
            raise ValueError(f"Rule '{name}' must include at least one list match")

        rule_name = str(name)
        has_uses = isinstance(uses, dict)

        # Check if we're in manual resolver parsing mode (resolver will be attached later)
        is_dotted = "." in rule_name
        manual_resolver_mode = (
            hasattr(self, "_manual_resolver_mode") and self._manual_resolver_mode
        )

        if has_uses:
            # Both parent and child rules can have resolver clauses
            rc = ast.ResolverConfig(
                method=uses["method"],
                args=tuple(uses.get("args", [])),
                flags=tuple(uses.get("flags", [])),
                optional_tokens=tuple(uses.get("optional_tokens", [])),
            )
        else:
            # Child rules require resolvers (unless in manual mode)
            # Parent rules have optional resolvers for canonicalization
            if not manual_resolver_mode and is_dotted:
                raise ValueError(
                    f"Child rule '{rule_name}' must include a resolver clause"
                )
            rc = None
        return ast.RuleDef(name=rule_name, pattern=pattern, resolver_config=rc)

    def escape(self, token):
        """Transform escape sequences like ^, $, \\d, \\w, etc."""
        val = token.value
        if val == "^":
            return ast.LineStart()
        if val == "$":
            return ast.LineEnd()
        if val in {r"\d", r"\D", r"\w", r"\W", r"\s", r"\S"}:
            return ast.Escape(value=val)
        raise ValueError(f"Unsupported escape: {val}")

    def dot(self, _):
        """Transform dot (.) wildcard."""
        return ast.Dot()

    def string(self, token):
        """Transform string literal, stripping quotes."""
        return ast.Literal(value=token.value[1:-1])  # strip quotes

    def IDENT(self, token):  # pylint: disable=invalid-name
        """Transform identifier token (follows Lark naming convention)."""
        return str(token)

    def list_match(self, *items):
        """Transform list match [[name]] or [[name|filter(arg)]]."""
        name = str(items[0])
        filter_expr = items[1] if len(items) > 1 else None
        return ast.ListMatch(name=name, filter=filter_expr)

    def filter_expr(self, func, _lpar, arg, _rpar):
        """Transform filter expression like filter(arg)."""
        return ast.FilterExpr(func=func, arg=arg)

    def quantified(self, expr, quant=None):
        """Transform quantified expression with optional quantifier."""
        if quant is None:
            return expr
        return ast.Quantified(expr=expr, quant=quant)

    def quantifier(self, item):
        """Transform quantifier expression."""
        return item

    def qmark(self, _token):
        """Transform ? quantifier (0 or 1 occurrence)."""
        return ast.Quantifier(min=0, max=1)

    def range(self, _lbrace, min_tok, _comma, max_tok, _rbrace):
        """Transform range quantifier {min,max}."""
        min_val = int(min_tok)
        max_val = int(max_tok)
        if min_val > max_val:
            raise ValueError(
                f"Minimum value ({min_val}) cannot be greater than "
                f"maximum value ({max_val}) in range"
            )
        return ast.Quantifier(min=min_val, max=max_val)

    def plus(self, _token):
        """Transform + quantifier (1 or more occurrences)."""
        return ast.Quantifier(min=1, max=None)

    def star(self, _token):
        """Transform * quantifier (0 or more occurrences)."""
        return ast.Quantifier(min=0, max=None)

    def range_quant(self, _lbrace, min_tok, _comma, max_tok, _rbrace):
        """Transform range quantifier with explicit syntax."""
        return self.range(_lbrace, min_tok, _comma, max_tok, _rbrace)

    def named_capture(self, _open, name, _gt, expr, _rpar):
        """Transform named capture group (?<name>expr)."""
        return ast.NamedCapture(name=name, expr=expr)

    def group_expr(self, _lpar, expr, _rpar):
        """Transform grouped expression (expr), unwrapping simple cases."""
        # Just unwrap single alt/concat
        if isinstance(expr, ast.Alt) and len(expr.options) == 1:
            inner = expr.options[0]
            if isinstance(inner, ast.Concat) and len(inner.parts) == 1:
                return inner.parts[0]
            return inner
        return expr

    def anchor(self, children):
        """Transform anchor tokens ^ and $."""
        token = children[0]
        val = token.value if hasattr(token, "value") else token
        if val == "^":
            return ast.LineStart()
        if val == "$":
            return ast.LineEnd()
        raise ValueError(f"Unexpected anchor token: {val}")

    def concat(self, *parts):
        """Transform concatenation of pattern parts."""
        # Flatten any nested Concat nodes
        flat_parts = []
        for part in parts:
            if isinstance(part, ast.Concat):
                flat_parts.extend(part.parts)
            else:
                flat_parts.append(part)
        return ast.Concat(parts=tuple(flat_parts))

    def alt(self, *items):
        """Transform alternation (choice) pattern."""
        return ast.Alt(options=tuple(items))

    def charclass(self, items):
        """Transform character class [abc] pattern."""
        return ast.CharClass(parts=tuple(items))

    def charclass_item(self, item):
        """Transform individual character class item."""
        return item

    def charclass_items(self, *items):
        """Transform list of character class items."""
        return tuple(items)

    def CHAR_RANGE(self, token):  # pylint: disable=invalid-name
        """Transform character range a-z (follows Lark naming convention)."""
        a, _, b = token.value
        return ast.CharRange(start=a, end=b)

    def ESCAPE(self, token):  # pylint: disable=invalid-name
        """Transform escape sequence token (follows Lark naming convention)."""
        return ast.Escape(value=token.value)

    def DOT(self, _token):  # pylint: disable=invalid-name
        """Transform dot token (follows Lark naming convention)."""
        return ast.Dot()

    def INT(self, token):  # pylint: disable=invalid-name
        """Transform integer token (follows Lark naming convention)."""
        return int(token)

    # === Resolver clauses ===
    def resolver_arg(self, *items):
        """Transform resolver argument - either value or key=value pair."""
        # items: either (STRING,) or (IDENT, STRING)
        if len(items) == 2:
            key = str(items[0])
            val = items[1]
            return (key, val)
        return items[0]

    def resolver_arg_list(self, *args):
        """Transform list of resolver arguments."""
        return list(args)

    def resolver_method(self, name, args=None):
        """Transform resolver method with optional arguments."""
        return {"method": str(name), "args": args or []}

    def optional_tokens_clause(self, _tok, *paths):
        """Transform optional tokens clause with file paths."""
        # strip quotes from each STRING
        toks = [p[1:-1] for p in paths]
        return ("optional-tokens", toks)

    def resolver_flag(self, *flags):
        """Transform resolver flag or optional tokens tuple."""
        # Accept one flag or optional_tokens tuple
        if not flags:
            return None
        flag = flags[0]
        if isinstance(flag, tuple) and flag[0] == "optional-tokens":
            return flag
        return str(flag)

    def resolver_with(self, _with, *flags):
        """Transform resolver with clause containing flags and optional tokens."""
        out = {"flags": [], "optional_tokens": []}
        for f in flags:
            if isinstance(f, tuple) and f[0] == "optional-tokens":
                out["optional_tokens"].extend(f[1])
            else:
                out["flags"].append(str(f))
        return out

    def uses_clause(self, *items):
        """Transform uses clause combining method and with configurations."""
        # items may include method_dict and with_dict
        method_dict = next(
            (i for i in items if isinstance(i, dict) and "method" in i), {}
        )
        with_dict = next((i for i in items if isinstance(i, dict) and "flags" in i), {})
        return {
            "method": method_dict.get("method"),
            "args": method_dict.get("args", []),
            "flags": with_dict.get("flags", []),
            "optional_tokens": with_dict.get("optional_tokens", []),
        }

    def resolver_default(self, _res, _def, uses_info, _newline=None):
        """Transform default resolver configuration."""
        # uses_info is config dict from uses_clause
        return ast.ResolverDefault(
            method=uses_info["method"],
            args=tuple(uses_info.get("args", [])),
            flags=tuple(uses_info.get("flags", [])),
            optional_tokens=tuple(uses_info.get("optional_tokens", [])),
        )

    def exact_range(self, _lbrace, n_tok, _rbrace):
        """Transform exact quantifier {n}."""
        n = int(n_tok)
        return ast.Quantifier(min=n, max=n)
