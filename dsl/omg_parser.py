import re
from pathlib import Path
from typing import Optional

from lark import Lark
from lark.exceptions import VisitError

from dsl.omg_ast import ResolverConfig, ResolverDefault, Root, RuleDef
from dsl.omg_transformer import DslTransformer

# OMG DSL Version
# This should match the version in the grammar file and be updated accordingly
# when the grammar changes.
# It is used to ensure compatibility between the parser and the grammar.
# If the grammar version changes, this should be updated to reflect that.
# The version is also used in the Root AST node to indicate the DSL version.
# The version is currently set to 1.0, which is the initial version of the
# OMG DSL grammar.
OMG_DSL_VERSION = "1.0"

GRAMMAR_PATH = Path(__file__).parent / "omg_grammar.lark"
with open(GRAMMAR_PATH, "r", encoding="utf-8") as f:
    DSL_GRAMMAR = f.read()

dsl_parser = Lark(DSL_GRAMMAR, start="root", parser="lalr", propagate_positions=True)

# Pre-compile regex patterns for performance
RE_RESOLVER_DEFAULT = re.compile(r"^\s*resolver\s+default\s+uses\b(.*)$")
RE_RESOLVER_RULE = re.compile(
    r"^\s*(?P<rule>[A-Za-z_][A-Za-z0-9_\.]*)\s*=.*uses\b(.*)$"
)
RE_OPTIONAL_TOKENS = re.compile(r"optional-tokens\(([^)]*)\)")
RE_FLAG_SPLIT = re.compile(r",\s*")
RE_VALID_FLAGS = {
    "ignore-case",
    "ignore-punctuation",
    "elide-whitespace",
    "word-boundary",
    "word-prefix",
    "word-suffix",
    "line-start",
    "line-end",
}
RE_EXTRACT_FLAGS = re.compile(
    r"\b(?:ignore-case|ignore-punctuation|elide-whitespace|word-boundary|word-prefix|word-suffix|line-start|line-end)\b"
)


def parse_string(
    code: str, *, unwrap: bool = True, dsl_file_path: Optional[str] = None
) -> Root:
    # Handle default and per-rule resolver clauses manually

    # First, resolve line continuations to get logical lines
    def _resolve_line_continuations(text):
        """Resolve backslash line continuations to get logical lines."""
        lines = text.splitlines()
        resolved = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check if line ends with backslash continuation
            while i < len(lines) and line.rstrip().endswith("\\"):
                # Remove the backslash and add the next line
                line = line.rstrip()[:-1] + " "  # Replace \\ with space
                i += 1
                if i < len(lines):
                    line += lines[
                        i
                    ].lstrip()  # Add next line without leading whitespace
            resolved.append(line)
            i += 1
        return resolved

    logical_lines = _resolve_line_continuations(code)

    default_clause = None
    per_rule_clauses = {}
    filtered = []
    for ln in logical_lines:
        m_def = RE_RESOLVER_DEFAULT.match(ln)
        m_rule = RE_RESOLVER_RULE.match(ln)
        if m_def:
            default_clause = m_def.group(1).strip()
        elif m_rule:
            rule_name = m_rule.group("rule")
            per_rule_clauses[rule_name] = ln.split("uses", 1)[1].strip()
            filtered.append(ln.split("uses", 1)[0].rstrip())
        else:
            filtered.append(ln)
    filtered_code = "\n".join(filtered)
    tree = dsl_parser.parse(filtered_code)
    try:
        transformer = DslTransformer(dsl_file_path=dsl_file_path)
        # Set manual resolver mode flag if we have per-rule resolvers OR any dotted rules
        # This allows dotted rules to exist without immediate resolver requirements
        # Optimize has_dotted_rules: avoid unnecessary splits
        has_dotted_rules = False
        for ln in logical_lines:
            if "=" in ln and not ln.lstrip().startswith("#"):
                eq_idx = ln.find("=")
                if "." in ln[:eq_idx]:
                    has_dotted_rules = True
                    break
        if per_rule_clauses or has_dotted_rules:
            transformer.set_manual_resolver_mode(True)
        root = transformer.transform(tree)

        # Make sure the version matches the expected OMG DSL version
        if root.version.value != OMG_DSL_VERSION:
            raise ValueError(
                f"Unsupported OMG DSL version: {root.version}. Expected {OMG_DSL_VERSION}."
            )

        # Process default resolver
        default_resolver = None
        if default_clause:
            parts = default_clause.split(" with ", 1)
            method_part = parts[0]
            flags_part = parts[1] if len(parts) > 1 else ""
            method = method_part.split("(")[0].strip()
            default_args = tuple(
                a.split("=", 1)[0].strip()
                for a in RE_FLAG_SPLIT.split(
                    method_part[method_part.find("(") + 1 : method_part.rfind(")")]
                    if "(" in method_part
                    else ""
                )
                if a.strip()
            )
            # Extract optional-tokens and flags only once
            default_optional = tuple(
                tok.strip().strip('"')
                for match in RE_OPTIONAL_TOKENS.findall(flags_part)
                for tok in match.split(",")
                if tok.strip()
            )
            flags_part_clean = RE_OPTIONAL_TOKENS.sub("", flags_part)
            default_flags = tuple(
                flag
                for flag in RE_FLAG_SPLIT.split(flags_part_clean)
                if flag.strip() and flag.strip() in RE_VALID_FLAGS
            )
            default_resolver = ResolverDefault(
                method=method,
                args=default_args,
                flags=default_flags,
                optional_tokens=default_optional,
            )

        # Process per-rule resolvers and create new rules with resolver configs
        updated_rules = []
        for rule in root.rules:
            clause = per_rule_clauses.get(rule.name)
            if clause:
                parts = clause.split(" with ", 1)
                method_part = parts[0]
                if method_part.startswith("resolver "):
                    method_part = method_part[len("resolver ") :].strip()
                flags_part = parts[1] if len(parts) > 1 else ""
                rule_args: list[str] = []
                if "(" in method_part:
                    inner = method_part[
                        method_part.find("(") + 1 : method_part.rfind(")")
                    ]
                    for a in RE_FLAG_SPLIT.split(inner):
                        if "=" in a:
                            k, v = a.split("=", 1)
                            rule_args.append(f"{k.strip()}={v.strip()}")
                        elif a.strip():
                            rule_args.append(a.strip())
                method = method_part.split("(")[0].strip()
                # Extract optional-tokens and flags only once
                rule_optional = tuple(
                    tok.strip().strip('"')
                    for match in RE_OPTIONAL_TOKENS.findall(clause)
                    for tok in match.split(",")
                    if tok.strip()
                )
                flags_clean = RE_OPTIONAL_TOKENS.sub("", flags_part)
                rule_flags = tuple(RE_EXTRACT_FLAGS.findall(flags_clean))
                resolver_config = ResolverConfig(
                    method=method,
                    args=tuple(rule_args),
                    flags=rule_flags,
                    optional_tokens=rule_optional,
                )
                updated_rule = RuleDef(
                    name=rule.name,
                    pattern=rule.pattern,
                    resolver_config=resolver_config,
                )
                updated_rules.append(updated_rule)
            else:
                updated_rules.append(rule)

        # Create new Root object with default resolver and updated rules
        root = Root(
            version=root.version,
            imports=root.imports,
            rules=tuple(updated_rules),
            default_resolver=default_resolver,
            dsl_file_path=dsl_file_path,
        )
        return root
    except VisitError as ve:
        if unwrap:
            raise ve.orig_exc from ve
        raise


def parse_file(path, *, unwrap: bool = True) -> Root:
    with open(path, "r", encoding="utf-8") as file:
        return parse_string(file.read(), unwrap=unwrap, dsl_file_path=path)
