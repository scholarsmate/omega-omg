import os
import pprint
import sys

import pytest
from lark import Tree
from lark.exceptions import UnexpectedToken, VisitError
from lark.lexer import Token

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dsl import omg_parser
from dsl.omg_ast import (
    Alt,
    CharClass,
    CharRange,
    Concat,
    Dot,
    Escape,
    FilterExpr,
    Import,
    LineEnd,
    LineStart,
    ListMatch,
    NamedCapture,
    Quantified,
    ResolverConfig,
    ResolverDefault,
    Root,
    RuleDef,
)


def parse_rule(dsl_code):
    root = omg_parser.parse_string(dsl_code)
    pprint.pprint(root)
    assert isinstance(root, Root)
    imports = root.imports
    assert isinstance(imports, tuple)
    rules = root.rules
    assert isinstance(rules, tuple)
    return imports, rules


def test_parse_string_no_rules():
    with pytest.raises(UnexpectedToken):
        _, _ = parse_rule("")  # No version, rules or imports


def test_parse_string_with_version():
    code = 'version 1.0\nimport "/media/foo.txt" as foo\nrule = [[foo]]'
    imports, rules = parse_rule(code)
    assert len(imports) == 1
    assert imports[0].path == "/media/foo.txt"
    assert imports[0].alias == "foo"
    assert len(rules) == 1
    assert rules[0].name == "rule"
    assert isinstance(rules[0].pattern, Alt)


def test_unsupported_version():
    code = 'version 1.99\nimport "/media/foo.txt" as foo\nrule = [[foo]]'
    with pytest.raises(ValueError):
        omg_parser.parse_string(code)


def test_malformed_import():
    with pytest.raises(Exception):  # or GrammarError
        omg_parser.parse_string('version 1.0\nimport "/media/foo.txt"')  # missing 'as'


def test_invalid_named_capture():
    code = 'version 1.0\nimport "/media/foo.txt" as foo\nrule = (?P<name>'
    with pytest.raises(Exception):
        omg_parser.parse_string(code)


def test_basic_list_match_quantifier():
    code = 'version 1.0\nimport "/media/foo.txt" as foo\nrule = [[foo]]{2,4}'
    _, rules = parse_rule(code)
    assert isinstance(rules, tuple)
    assert isinstance(rules[0], RuleDef)
    pattern = rules[0].pattern
    assert isinstance(pattern, Alt)
    assert len(pattern.options) == 1
    concat = pattern.options[0]
    assert isinstance(concat, Concat)
    assert len(concat.parts) == 1
    quantified = concat.parts[0]
    assert isinstance(quantified, Quantified)
    assert isinstance(quantified.expr, ListMatch)


def test_escape_sequence_with_quantifier():
    code = 'version 1.0\nimport "/media/foo.txt" as foo\nrule = [[foo]]\\s{1,4}'
    root = omg_parser.parse_string(code)
    rules = root.rules
    pattern = rules[0].pattern
    assert isinstance(pattern, Alt)
    assert len(pattern.options) == 1
    concat = pattern.options[0]
    assert isinstance(concat, Concat)
    assert len(concat.parts) == 2
    quantified = concat.parts[1]
    assert isinstance(quantified, Quantified)
    assert isinstance(quantified.expr, Escape)


def test_import_and_list_match():
    code = 'version 1.0\nimport "/media/test.txt" as test\nrule = [[test]]'
    imports, rules = parse_rule(code)

    assert len(imports) == 1
    assert imports[0] == Import(path="/media/test.txt", alias="test")

    rule = rules[0]
    assert rule.name == "rule"
    pattern = rule.pattern
    assert isinstance(pattern, Alt)
    concat = pattern.options[0]
    assert isinstance(concat, Concat)
    assert isinstance(concat.parts[0], ListMatch)


def test_list_match_with_filter():
    code = 'version 1.0\nimport "/media/foo.txt" as foo\nrule = [[foo:startsWith("A")]]'
    _, rules = parse_rule(code)
    pattern = rules[0].pattern
    lm = pattern.options[0].parts[0]
    assert isinstance(lm, ListMatch)
    assert isinstance(lm.filter, FilterExpr)
    assert lm.filter.func == "startsWith"
    assert lm.filter.arg == '"A"'


def test_named_capture():
    code = 'version 1.0\nimport "/media/first.txt" as first\nrule = (?P<name>[[first]])'
    _, rules = parse_rule(code)
    nc = rules[0].pattern.options[0].parts[0]
    assert isinstance(nc, NamedCapture)
    assert nc.name == "name"
    assert isinstance(nc.expr, Alt)


def test_complex_concat_with_named_capture_and_quantifier():
    code = """
    version 1.0
    import "/media/surnames.txt" as surnames with ignore-case, word-boundary
    import "/media/firsts.txt" as firsts with ignore-case, word-boundary
    rule = (?P<first>[[firsts]])\\s{1,10}(?P<last>[[surnames:startsWith("Mc")]])
    """
    _, rules = parse_rule(code)
    concat = rules[0].pattern.options[0]
    assert isinstance(concat.parts[0], NamedCapture)
    assert isinstance(concat.parts[1], Quantified)
    assert isinstance(concat.parts[2], NamedCapture)
    assert isinstance(concat.parts[2].expr.options[0].parts[0].filter, FilterExpr)


def test_nested_rule_reference():
    code = """
    version 1.0
    import "/media/firsts.txt" as firsts with word-boundary
    ruleA = [[firsts]]
    ruleB = [[ruleA]]\\s{1,10}\\w{1,10}
    """
    _, rules = parse_rule(code)
    assert rules[0].name == "ruleA"
    assert rules[1].name == "ruleB"
    concat = rules[1].pattern.options[0]
    assert concat.parts[0].name == "ruleA"
    assert isinstance(concat.parts[1], Quantified)
    assert isinstance(concat.parts[2], Quantified)


def test_alternation_expr():
    code = """
    version 1.0
    import "/media/colors.txt" as colors with elide-whitespace
    import "/media/fruit.txt" as fruit with elide-whitespace
    rule = [[colors]] | [[fruit]]
    """
    _, rules = parse_rule(code)
    pattern = rules[0].pattern
    assert isinstance(pattern, Alt)
    assert len(pattern.options) == 2

    assert isinstance(pattern.options[0].parts[0], ListMatch)
    assert isinstance(pattern.options[1].parts[0], ListMatch)


def test_char_class():
    code = """
    version 1.0
    import "/media/foo.txt" as foo with ignore-punctuation
    rule = [abc][[foo]]
    """
    _, rules = parse_rule(code)
    pattern = rules[0].pattern
    concat = pattern.options[0]
    char_class = concat.parts[0]
    assert isinstance(char_class, CharClass)
    assert hasattr(char_class, "parts")
    assert isinstance(char_class.parts, tuple)  # changed from list to tuple
    assert all(isinstance(p, (CharRange, Escape, str)) for p in char_class.parts)


def test_nested_named_capture_groups():
    code = """
    version 1.0
    import "/media/firsts.txt" as firsts
    import "/media/lasts.txt" as lasts
    rule = (?P<outer>(?P<inner>[[firsts]])\\s{1,10}[[lasts]])
    """
    _, rules = parse_rule(code)
    pattern = rules[0].pattern
    assert isinstance(pattern, Alt)
    outer = pattern.options[0].parts[0]
    assert isinstance(outer, NamedCapture)
    assert outer.name == "outer"
    inner_expr = outer.expr.options[0]
    inner = inner_expr.parts[0]
    assert isinstance(inner, NamedCapture)
    assert inner.name == "inner"
    assert isinstance(inner.expr.options[0].parts[0], ListMatch)
    assert isinstance(inner_expr.parts[1], Quantified)
    assert isinstance(inner_expr.parts[2], ListMatch)
    assert inner_expr.parts[1].quant.min == 1
    assert inner_expr.parts[1].quant.max == 10


def test_char_range_transform():
    transformer = omg_parser.DslTransformer()
    token = Token("CHAR_RANGE", "a-z")
    result = transformer.CHAR_RANGE(token)

    assert isinstance(result, CharRange)
    assert result.start == "a"
    assert result.end == "z"


def test_group_with_quantifier_unwraps_to_list_match_inside():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = ([[foo]]){1,10}
    """
    root = omg_parser.parse_string(code)
    rules = root.rules

    pattern = rules[0].pattern

    # The outer expression is Alt -> Concat -> Quantified
    assert isinstance(pattern, Alt)
    assert len(pattern.options) == 1

    concat = pattern.options[0]
    assert isinstance(concat, Concat)
    assert len(concat.parts) == 1

    quantified = concat.parts[0]
    assert isinstance(quantified, Quantified)
    assert quantified.quant.min == 1
    assert quantified.quant.max == 10

    # The inside of the quantifier should be the unwrapped ListMatch
    assert isinstance(quantified.expr, ListMatch)


def test_group_unwraps_alt_single_concat_multiple_parts():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = ([[foo]] [[foo]])
    """
    root = omg_parser.parse_string(code)
    rules = root.rules
    pattern = rules[0].pattern
    pprint.pprint(pattern)

    # Should be a single Concat inside a single Alt — unwrapped one level
    assert isinstance(pattern, Alt)
    assert len(pattern.options) == 1

    # That Concat should have 2 parts
    concat = pattern.options[0]
    assert isinstance(concat, Concat)
    assert len(concat.parts) == 2
    assert all(isinstance(p, ListMatch) for p in concat.parts)


def test_group_with_question_quantifier():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    import "/media/bar.txt" as bar
    rule = ([[foo]] [[bar]])?
    """
    _, rules = parse_rule(code)
    pattern = rules[0].pattern.options[0]
    print(f"Pattern: {pattern}")
    assert isinstance(pattern.parts[0], Quantified)
    inner = pattern.parts[0].expr
    assert isinstance(inner, Concat)
    assert len(inner.parts) == 2
    assert all(isinstance(p, ListMatch) for p in inner.parts)


def test_parse_file(tmp_path):
    dsl_code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = [[foo]]
    """
    # Create temp DSL file
    dsl_file = tmp_path / "sample.dsl"
    dsl_file.write_text(dsl_code)

    # Parse it using the file parser
    root = omg_parser.parse_file(str(dsl_file))
    assert root.version.value == "1.0"
    assert len(root.imports) == 1
    assert root.imports[0].alias == "foo"


# -- Tests for error handling and edge cases -- #


def test_parser_handles_empty_input():
    # Ensure we handle the case of empty input gracefully
    with pytest.raises(UnexpectedToken):
        omg_parser.parse_string("")


def test_invalid_quantifier_missing_start():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = [[foo]]{,4}
    """
    with pytest.raises(UnexpectedToken):
        omg_parser.parse_string(code)


def test_invalid_quantifier_reversed_bounds():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = [[foo]]{4,2}
    """
    with pytest.raises(VisitError) as exc_info:
        omg_parser.parse_string(code, unwrap=False)
    orig_exc = exc_info.value.orig_exc
    assert isinstance(orig_exc, ValueError)
    assert (
        str(orig_exc)
        == "Minimum value (4) cannot be greater than maximum value (2) in range"
    )


def test_parser_fails_cleanly():
    with pytest.raises(Exception):
        omg_parser.parse_string("garbage that is not valid DSL")


def test_unexpected_tree_node():
    transformer = omg_parser.DslTransformer()
    dummy_tree = Tree("unknown_node", [])
    result = transformer.__default__(
        "unknown_node", dummy_tree.children, dummy_tree.meta
    )
    assert isinstance(result, Tree)
    assert result.data == "unknown_node"
    assert result.children == []


def test_anchor_tokens():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = ^[[foo]]$
    """
    _, rules = parse_rule(code)
    pattern = rules[0].pattern.options[0]
    assert isinstance(pattern.parts[0], LineStart)
    assert isinstance(pattern.parts[1], ListMatch)
    assert pattern.parts[1].name == "foo"
    assert isinstance(pattern.parts[2], LineEnd)


def test_question_quantifier():
    code = """
    version 1.0\nimport "/media/foo.txt" as foo
    import "/media/bar.txt" as bar
    rule = [[foo]][[bar]]?
    """
    _, rules = parse_rule(code)
    quant = rules[0].pattern.options[0].parts[1]
    assert isinstance(quant, Quantified)
    assert quant.quant.min == 0
    assert quant.quant.max == 1


def test_dot_token():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    import "/media/bar.txt" as bar
    rule = [[foo]].[[bar]] # Match any single character between lists foo and bar
    """
    _, rules = parse_rule(code)
    pattern = rules[0].pattern.options[0].parts[1]
    assert isinstance(pattern, Dot)


def test_grouping_parens():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = ([[foo]])
    """
    # Note: The outer parentheses are not necessary, but they should be handled correctly
    _, rules = parse_rule(code)
    pattern = rules[0].pattern
    assert isinstance(pattern, Alt)
    concat = pattern.options[0]
    assert isinstance(concat, Concat)
    inner = concat.parts[0]
    assert isinstance(inner, ListMatch)
    assert inner.name == "foo"


def test_grouping_with_alternation():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    import "/media/bar.txt" as bar
    rule = ([[foo]] | [[bar]])
    """
    _, rules = parse_rule(code)
    pattern = rules[0].pattern
    assert isinstance(pattern, Alt)
    concat = pattern.options[0]
    assert isinstance(concat, Concat)
    inner = concat.parts[0]
    assert isinstance(inner, Alt)
    assert len(inner.options) == 2
    assert isinstance(inner.options[0].parts[0], ListMatch)
    assert inner.options[0].parts[0].name == "foo"
    assert isinstance(inner.options[1].parts[0], ListMatch)
    assert inner.options[1].parts[0].name == "bar"


def test_malformed_range_node():
    transformer = omg_parser.DslTransformer()
    malformed = Tree("range", ["unexpected"])

    with pytest.raises(VisitError) as exc_info:
        transformer.transform(malformed)

    # Confirm it's wrapping the correct original exception
    orig_exc = exc_info.value.orig_exc
    assert isinstance(orig_exc, TypeError)
    assert "range() missing 4 required positional arguments" in str(orig_exc)


def test_rule_missing_list_match_raises():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = "hello"
    """
    # This rule is missing a list match, which should raise an error
    with pytest.raises(VisitError) as exc_info:
        omg_parser.parse_string(code, unwrap=False)
    assert isinstance(exc_info.value.orig_exc, ValueError)
    assert (
        str(exc_info.value.orig_exc)
        == "Rule 'rule' must include at least one list match"
    )


def test_default_resolver_clause():
    code = """
    version 1.0
    import "names.txt" as names with word-boundary
    resolver default uses exact with ignore-case, optional-tokens("a.txt")
    rule = [[names]]
    """
    root = omg_parser.parse_string(code)
    assert isinstance(root.default_resolver, ResolverDefault)
    rd = root.default_resolver
    assert rd.method == "exact"
    assert "ignore-case" in rd.flags
    assert rd.optional_tokens == ("a.txt",)


def test_rule_resolver_clause():
    code = """
    version 1.0
    import "foo.txt" as foo
    foo.bar = [[foo]] uses resolver fuzzy(threshold="0.75") with ignore-case
    """
    root = omg_parser.parse_string(code)
    print(f"Parsed root: {root}")
    assert len(root.rules) == 1
    rd = root.rules[0]
    assert isinstance(rd.resolver_config, ResolverConfig)
    rc = rd.resolver_config
    assert rc.method == "fuzzy"
    assert "ignore-case" in rc.flags
    # args may include key or value depending on parsing; allow key=value string too
    assert rc.args in ((("threshold", '"0.75"'),), ('"0.75"',), ('threshold="0.75"',))


# === Comprehensive Resolver Tests ===


def test_resolver_default_with_multiple_flags():
    """Test default resolver with multiple flags and optional tokens."""
    code = """
    version 1.0
    import "names.txt" as names
    resolver default uses fuzzy(threshold="0.8") with ignore-case, ignore-punctuation, optional-tokens("the", "a")
    rule = [[names]]
    """
    root = omg_parser.parse_string(code)
    assert isinstance(root.default_resolver, ResolverDefault)
    rd = root.default_resolver
    assert rd.method == "fuzzy"
    assert "ignore-case" in rd.flags
    assert "ignore-punctuation" in rd.flags
    assert rd.optional_tokens == ("the", "a")
    # Check that the arg was parsed (threshold parameter)
    assert len(rd.args) > 0


def test_resolver_default_exact_method():
    """Test default resolver with exact method."""
    code = """
    version 1.0
    import "cities.txt" as cities
    resolver default uses exact with word-boundary
    rule = [[cities]]
    """
    root = omg_parser.parse_string(code)
    assert isinstance(root.default_resolver, ResolverDefault)
    rd = root.default_resolver
    assert rd.method == "exact"
    assert "word-boundary" in rd.flags
    assert len(rd.optional_tokens) == 0


def test_resolver_default_contains_method():
    """Test default resolver with contains method."""
    code = """
    version 1.0
    import "keywords.txt" as keywords
    resolver default uses contains with elide-whitespace
    rule = [[keywords]]
    """
    root = omg_parser.parse_string(code)
    assert isinstance(root.default_resolver, ResolverDefault)
    rd = root.default_resolver
    assert rd.method == "contains"
    assert "elide-whitespace" in rd.flags


def test_resolver_per_rule_fuzzy():
    """Test per-rule resolver with fuzzy matching."""
    code = """
    version 1.0
    import "people.txt" as people
    import "places.txt" as places
    person.name = [[people]] uses resolver fuzzy(threshold="0.9")
    place.city = [[places]] uses resolver exact with ignore-case
    """
    root = omg_parser.parse_string(code)
    assert len(root.rules) == 2

    # Check first rule (person.name)
    person_rule = root.rules[0]
    assert person_rule.name == "person.name"
    assert isinstance(person_rule.resolver_config, ResolverConfig)
    pc = person_rule.resolver_config
    assert pc.method == "fuzzy"
    assert any("threshold" in str(arg) for arg in pc.args)

    # Check second rule (place.city)
    place_rule = root.rules[1]
    assert place_rule.name == "place.city"
    assert isinstance(place_rule.resolver_config, ResolverConfig)
    plc = place_rule.resolver_config
    assert plc.method == "exact"
    assert "ignore-case" in plc.flags


def test_resolver_per_rule_with_optional_tokens():
    """Test per-rule resolver with optional tokens."""
    code = """
    version 1.0
    import "organizations.txt" as orgs
    org.name = [[orgs]] uses resolver fuzzy(threshold="0.85") with ignore-case, optional-tokens("Inc", "Corp", "LLC")
    """
    root = omg_parser.parse_string(code)
    assert len(root.rules) == 1
    rule = root.rules[0]
    assert rule.name == "org.name"
    assert isinstance(rule.resolver_config, ResolverConfig)
    rc = rule.resolver_config
    assert rc.method == "fuzzy"
    assert "ignore-case" in rc.flags
    assert "Inc" in rc.optional_tokens
    assert "Corp" in rc.optional_tokens
    assert "LLC" in rc.optional_tokens


def test_resolver_multiple_args():
    """Test resolver with multiple arguments."""
    code = """
    version 1.0
    import "data.txt" as data
    complex.rule = [[data]] uses resolver advanced(threshold="0.7", mode="strict", weight="high")
    """
    root = omg_parser.parse_string(code)
    rule = root.rules[0]
    assert isinstance(rule.resolver_config, ResolverConfig)
    rc = rule.resolver_config
    assert rc.method == "advanced"
    # Check that we parsed multiple arguments
    assert len(rc.args) >= 3


def test_resolver_default_and_per_rule_combined():
    """Test having both default resolver and per-rule resolvers."""
    code = """
    version 1.0
    import "general.txt" as general
    import "specific.txt" as specific
    resolver default uses exact with ignore-case
    normal_rule = [[general]]
    special.rule = [[specific]] uses resolver fuzzy(threshold="0.95") with word-boundary
    """
    root = omg_parser.parse_string(code)

    # Check default resolver
    assert isinstance(root.default_resolver, ResolverDefault)
    rd = root.default_resolver
    assert rd.method == "exact"
    assert "ignore-case" in rd.flags

    # Check rules
    assert len(root.rules) == 2
    normal_rule = root.rules[0]
    special_rule = root.rules[1]

    # Normal rule should not have resolver config (uses default)
    assert normal_rule.name == "normal_rule"
    assert normal_rule.resolver_config is None

    # Special rule should have its own resolver config
    assert special_rule.name == "special.rule"
    assert isinstance(special_rule.resolver_config, ResolverConfig)
    src = special_rule.resolver_config
    assert src.method == "fuzzy"
    assert "word-boundary" in src.flags


def test_resolver_error_non_dotted_rule():
    """Test that resolver clauses are allowed on non-dotted (parent) rules."""
    # According to RESOLUTION.md, parent rules can have resolvers for canonicalization
    code = """
    version 1.0
    import "test.txt" as test
    simple_rule = [[test]] uses resolver exact
    """
    # The parser should accept this and create the rule with resolver config
    root = omg_parser.parse_string(code)
    assert len(root.rules) == 1
    rule = root.rules[0]
    assert rule.name == "simple_rule"
    # For parent rules, the resolver clause should be accepted
    assert rule.resolver_config is not None
    assert rule.resolver_config.method == "exact"


def test_resolver_error_missing_resolver_clause():
    """Test that the manual parsing mode handles dotted rules without resolvers."""
    code = """
    version 1.0
    import "test.txt" as test
    dotted.rule = [[test]]
    """
    # In manual parsing mode, this should work fine
    # The resolver config will be None initially (can be added later if needed)
    root = omg_parser.parse_string(code)
    assert len(root.rules) == 1
    assert root.rules[0].name == "dotted.rule"
    # In manual mode, resolver_config will be None initially
    assert root.rules[0].resolver_config is None


def test_resolver_complex_optional_tokens():
    """Test resolver with complex optional tokens patterns."""
    code = """
    version 1.0
    import "companies.txt" as companies
    company.name = [[companies]] uses resolver fuzzy(threshold="0.8") with ignore-case, optional-tokens("Inc.", "Corporation", "Co.", "Ltd.")
    """
    root = omg_parser.parse_string(code)
    rule = root.rules[0]
    assert isinstance(rule.resolver_config, ResolverConfig)
    rc = rule.resolver_config
    assert "Inc." in rc.optional_tokens
    assert "Corporation" in rc.optional_tokens
    assert "Co." in rc.optional_tokens
    assert "Ltd." in rc.optional_tokens


def test_resolver_method_without_args():
    """Test resolver method without parentheses/arguments."""
    code = """
    version 1.0
    import "simple.txt" as simple
    resolver default uses exact
    rule = [[simple]]
    """
    root = omg_parser.parse_string(code)
    assert isinstance(root.default_resolver, ResolverDefault)
    rd = root.default_resolver
    assert rd.method == "exact"
    assert len(rd.args) == 0
    assert len(rd.flags) == 0
    assert len(rd.optional_tokens) == 0


def test_resolver_per_rule_method_without_args():
    """Test per-rule resolver method without arguments."""
    code = """
    version 1.0
    import "data.txt" as data
    data.item = [[data]] uses resolver contains
    """
    root = omg_parser.parse_string(code)
    rule = root.rules[0]
    assert isinstance(rule.resolver_config, ResolverConfig)
    rc = rule.resolver_config
    assert rc.method == "contains"
    assert len(rc.args) == 0
