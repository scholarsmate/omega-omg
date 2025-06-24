import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

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
    Literal,
    NamedCapture,
    Quantified,
    ResolverConfig,
    ResolverDefault,
    Root,
)


def test_ast_minimal_dsl():
    code = """
    version 1.0
    import "/media/colors.txt" as colors
    rule = [[colors]]
    """
    root = omg_parser.parse_string(code)
    assert isinstance(root, Root)
    assert root.version.value == "1.0"
    assert len(root.imports) == 1
    assert root.imports[0] == Import(path="/media/colors.txt", alias="colors")
    assert len(root.rules) == 1
    rule = root.rules[0]
    assert rule.name == "rule"
    assert isinstance(rule.pattern, Alt)
    assert isinstance(rule.pattern.options[0], Concat)
    assert isinstance(rule.pattern.options[0].parts[0], ListMatch)
    assert rule.pattern.options[0].parts[0].name == "colors"


def test_ast_import_with_flags():
    code = """
    version 1.0
    import "/media/foo.txt" as foo with ignore-case, elide-whitespace
    rule = [[foo]]
    """
    root = omg_parser.parse_string(code)
    imp = root.imports[0]
    assert imp.alias == "foo"
    assert imp.flags == ("ignore-case", "elide-whitespace")


def test_ast_named_capture_with_quantifier():
    code = """version 1.0
    import "/media/verbs.txt" as verbs
    rule = (?P<action>[[verbs]]){1,10}
    """
    root = omg_parser.parse_string(code)
    rule = root.rules[0]
    quant = rule.pattern.options[0].parts[0]
    assert isinstance(quant, Quantified)
    assert isinstance(quant.expr, NamedCapture)
    assert quant.expr.name == "action"
    assert isinstance(quant.expr.expr, Alt)


def test_ast_char_class_and_escape():
    code = """
    version 1.0
    import "/media/adjectives.txt" as adjectives
    rule = [A-Z][[adjectives]]\\s{1,2}
    """
    root = omg_parser.parse_string(code)
    parts = root.rules[0].pattern.options[0].parts
    assert isinstance(parts[0], CharClass)
    assert isinstance(parts[1], ListMatch)
    assert isinstance(parts[2], Quantified)
    assert parts[2].quant.min == 1
    assert parts[2].quant.max == 2
    assert isinstance(parts[2].expr, Escape)
    assert parts[2].expr.value == "\\s"


def test_ast_alternation_and_nested_groups():
    code = """version 1.0
    import "/media/colors.txt" as colors
    import "/media/shapes.txt" as shapes
    rule = ([[colors]] | [[shapes]]){1,2}
    """
    root = omg_parser.parse_string(code)
    quant = root.rules[0].pattern.options[0].parts[0]
    assert isinstance(quant, Quantified)
    assert isinstance(quant.expr, Alt)
    assert {o.parts[0].name for o in quant.expr.options} == {"colors", "shapes"}


def test_ast_group_expr_unwraps_deep():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = ([[foo]])
    """
    root = omg_parser.parse_string(code)
    pattern = root.rules[0].pattern
    # Should be an Alt with one Concat, one ListMatch
    assert isinstance(pattern, Alt)
    assert len(pattern.options) == 1
    concat = pattern.options[0]
    assert isinstance(concat, Concat)
    assert len(concat.parts) == 1
    assert isinstance(concat.parts[0], ListMatch)


def test_ast_group_expr_unwraps_alt_single_concat_multi_parts():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = ([[foo]] [[foo]])
    """
    root = omg_parser.parse_string(code)
    pattern = root.rules[0].pattern
    assert isinstance(pattern, Alt)
    assert isinstance(pattern.options[0], Concat)
    assert len(pattern.options[0].parts) == 2


def test_ast_filter_expression():
    code = """
    version 1.0
    import "/media/names.txt" as names
    rule = [[names:startsWith("A")]]
    """
    root = omg_parser.parse_string(code)
    lm = root.rules[0].pattern.options[0].parts[0]
    assert isinstance(lm, ListMatch)
    assert lm.name == "names"
    assert isinstance(lm.filter, FilterExpr)
    assert lm.filter.func == "startsWith"
    assert lm.filter.arg == '"A"'


def test_ast_dot_and_anchor():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = ^.[[foo]]$
    """
    root = omg_parser.parse_string(code)
    parts = root.rules[0].pattern.options[0].parts
    assert isinstance(parts[0], LineStart)
    assert isinstance(parts[1], Dot)
    assert isinstance(parts[2], ListMatch) and parts[2].name == "foo"
    assert isinstance(parts[3], LineEnd)


def test_ast_char_range_and_literal():
    code = """
    version 1.0
    import "/media/adjectives.txt" as adjectives
    rule = [A-Z][[adjectives]]"!"
    """
    root = omg_parser.parse_string(code)
    parts = root.rules[0].pattern.options[0].parts
    assert isinstance(parts[0], CharClass)
    assert isinstance(parts[0].parts[0], CharRange)
    assert parts[0].parts[0].start == "A"
    assert parts[0].parts[0].end == "Z"
    assert isinstance(parts[1], ListMatch)
    assert parts[1].name == "adjectives"
    assert isinstance(parts[2], Literal)
    assert parts[2].value == "!"


def test_ast_quantifier_range():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = [[foo]]{2,5}
    """
    root = omg_parser.parse_string(code)
    quant = root.rules[0].pattern.options[0].parts[0]
    assert isinstance(quant, Quantified)
    assert quant.quant.min == 2
    assert quant.quant.max == 5


def test_ast_parse_file(tmp_path):
    dsl_code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = [[foo]]
    """
    file_path = tmp_path / "example.dsl"
    file_path.write_text(dsl_code)

    root = omg_parser.parse_file(str(file_path))
    assert isinstance(root, Root)
    assert root.version.value == "1.0"
    assert len(root.imports) == 1
    assert root.imports[0].alias == "foo"


# -- Invalid Tests --


def test_ast_invalid_rule_quantifier_range():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = [[foo]]{5,2}
    """
    with pytest.raises(
        ValueError,
        match="Minimum value \\(5\\) cannot be greater than maximum value \\(2\\) in range",
    ):
        omg_parser.parse_string(code)


def test_ast_invalid_rule_without_list_match():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = "hello"
    """
    with pytest.raises(
        ValueError, match="Rule 'rule' must include at least one list match"
    ):
        omg_parser.parse_string(code)


def test_ast_invalid_rule_without_list_match_wrapped():
    code = """
    version 1.0
    import "/media/foo.txt" as foo
    rule = "hello"
    """
    with pytest.raises(Exception) as exc_info:
        omg_parser.parse_string(code, unwrap=False)

    from lark.exceptions import VisitError

    assert isinstance(exc_info.value, VisitError)
    orig = exc_info.value.orig_exc
    assert isinstance(orig, ValueError)
    assert str(orig) == "Rule 'rule' must include at least one list match"


# === Resolver AST Tests ===


def test_ast_default_resolver_basic():
    """Test basic default resolver AST structure."""
    code = """
    version 1.0
    import "/media/names.txt" as names
    resolver default uses fuzzy(threshold="0.8") with ignore-case
    rule = [[names]]
    """
    root = omg_parser.parse_string(code)
    assert isinstance(root.default_resolver, ResolverDefault)
    rd = root.default_resolver
    assert rd.method == "fuzzy"
    assert "ignore-case" in rd.flags
    assert len(rd.args) > 0  # Should have threshold parameter


def test_ast_default_resolver_multiple_flags():
    """Test default resolver with multiple flags and optional tokens."""
    code = """
    version 1.0
    import "/media/data.txt" as data
    resolver default uses exact with ignore-case, ignore-punctuation, optional-tokens("the", "a", "an")
    rule = [[data]]
    """
    root = omg_parser.parse_string(code)
    assert isinstance(root.default_resolver, ResolverDefault)
    rd = root.default_resolver
    assert rd.method == "exact"
    assert "ignore-case" in rd.flags
    assert "ignore-punctuation" in rd.flags
    assert "the" in rd.optional_tokens
    assert "a" in rd.optional_tokens
    assert "an" in rd.optional_tokens


def test_ast_per_rule_resolver_basic():
    """Test basic per-rule resolver AST structure."""
    code = """
    version 1.0
    import "/media/people.txt" as people
    person.name = [[people]] uses resolver fuzzy(threshold="0.9") with ignore-case
    """
    root = omg_parser.parse_string(code)
    assert len(root.rules) == 1
    rule = root.rules[0]
    assert rule.name == "person.name"
    assert isinstance(rule.resolver_config, ResolverConfig)
    rc = rule.resolver_config
    assert rc.method == "fuzzy"
    assert "ignore-case" in rc.flags
    assert len(rc.args) > 0  # Should have threshold parameter


def test_ast_per_rule_resolver_with_optional_tokens():
    """Test per-rule resolver with optional tokens."""
    code = """
    version 1.0
    import "/media/companies.txt" as companies
    company.name = [[companies]] uses resolver exact with optional-tokens("Inc", "Corp", "LLC")
    """
    root = omg_parser.parse_string(code)
    rule = root.rules[0]
    assert isinstance(rule.resolver_config, ResolverConfig)
    rc = rule.resolver_config
    assert rc.method == "exact"
    assert "Inc" in rc.optional_tokens
    assert "Corp" in rc.optional_tokens
    assert "LLC" in rc.optional_tokens


def test_ast_multiple_dotted_rules_with_resolvers():
    """Test multiple dotted rules each with their own resolver configurations."""
    code = """
    version 1.0
    import "/media/people.txt" as people
    import "/media/places.txt" as places
    import "/media/organizations.txt" as orgs
    person.name = [[people]] uses resolver fuzzy(threshold="0.85") with ignore-case
    place.city = [[places]] uses resolver exact with word-boundary
    org.company = [[orgs]] uses resolver contains with ignore-punctuation, optional-tokens("Inc.", "Corp.")
    """
    root = omg_parser.parse_string(code)
    assert len(root.rules) == 3

    # Check person.name
    person_rule = root.rules[0]
    assert person_rule.name == "person.name"
    assert isinstance(person_rule.resolver_config, ResolverConfig)
    assert person_rule.resolver_config.method == "fuzzy"
    assert "ignore-case" in person_rule.resolver_config.flags

    # Check place.city
    place_rule = root.rules[1]
    assert place_rule.name == "place.city"
    assert isinstance(place_rule.resolver_config, ResolverConfig)
    assert place_rule.resolver_config.method == "exact"
    assert "word-boundary" in place_rule.resolver_config.flags

    # Check org.company
    org_rule = root.rules[2]
    assert org_rule.name == "org.company"
    assert isinstance(org_rule.resolver_config, ResolverConfig)
    assert org_rule.resolver_config.method == "contains"
    assert "ignore-punctuation" in org_rule.resolver_config.flags
    assert "Inc." in org_rule.resolver_config.optional_tokens
    assert "Corp." in org_rule.resolver_config.optional_tokens


def test_ast_default_and_per_rule_resolvers_combined():
    """Test having both default resolver and per-rule resolvers in the same AST."""
    code = """
    version 1.0
    import "/media/general.txt" as general
    import "/media/specific.txt" as specific
    resolver default uses exact with ignore-case
    normal_rule = [[general]]
    special.rule = [[specific]] uses resolver fuzzy(threshold="0.95") with word-boundary
    another_normal = [[general]]
    """
    root = omg_parser.parse_string(code)

    # Check default resolver
    assert isinstance(root.default_resolver, ResolverDefault)
    rd = root.default_resolver
    assert rd.method == "exact"
    assert "ignore-case" in rd.flags

    # Check rules
    assert len(root.rules) == 3

    # Normal rules should not have resolver configs
    normal_rule1 = root.rules[0]
    normal_rule2 = root.rules[2]
    assert normal_rule1.name == "normal_rule"
    assert normal_rule1.resolver_config is None
    assert normal_rule2.name == "another_normal"
    assert normal_rule2.resolver_config is None

    # Special rule should have its own resolver config
    special_rule = root.rules[1]
    assert special_rule.name == "special.rule"
    assert isinstance(special_rule.resolver_config, ResolverConfig)
    src = special_rule.resolver_config
    assert src.method == "fuzzy"
    assert "word-boundary" in src.flags


def test_ast_resolver_complex_arguments():
    """Test resolver with complex argument patterns."""
    code = """
    version 1.0
    import "/media/data.txt" as data
    complex.rule = [[data]] uses resolver advanced(threshold="0.7", mode="strict", weight="high") with ignore-case
    """
    root = omg_parser.parse_string(code)
    rule = root.rules[0]
    assert isinstance(rule.resolver_config, ResolverConfig)
    rc = rule.resolver_config
    assert rc.method == "advanced"
    assert len(rc.args) >= 3  # Should have multiple arguments
    assert "ignore-case" in rc.flags


def test_ast_resolver_methods_without_arguments():
    """Test various resolver methods without arguments."""
    code = """
    version 1.0
    import "/media/test1.txt" as test1
    import "/media/test2.txt" as test2
    import "/media/test3.txt" as test3
    resolver default uses exact
    rule1.exact = [[test1]] uses resolver fuzzy
    rule2.contains = [[test2]] uses resolver contains with ignore-case
    rule3.custom = [[test3]] uses resolver custom_method with word-boundary, optional-tokens("test")
    """
    root = omg_parser.parse_string(code)

    # Check default resolver
    assert root.default_resolver.method == "exact"
    assert len(root.default_resolver.args) == 0

    # Check per-rule resolvers
    assert len(root.rules) == 3

    rule1 = root.rules[0]
    assert rule1.resolver_config.method == "fuzzy"
    assert len(rule1.resolver_config.args) == 0

    rule2 = root.rules[1]
    assert rule2.resolver_config.method == "contains"
    assert "ignore-case" in rule2.resolver_config.flags

    rule3 = root.rules[2]
    assert rule3.resolver_config.method == "custom_method"
    assert "word-boundary" in rule3.resolver_config.flags
    assert "test" in rule3.resolver_config.optional_tokens


def test_ast_resolver_dataclass_fields():
    """Test that resolver AST nodes have correct dataclass fields."""
    code = """
    version 1.0
    import "/media/test.txt" as test
    resolver default uses fuzzy(threshold="0.8") with ignore-case, optional-tokens("the")
    test.rule = [[test]] uses resolver exact(mode="strict") with word-boundary, optional-tokens("a", "an")
    """
    root = omg_parser.parse_string(code)

    # Test ResolverDefault dataclass
    rd = root.default_resolver
    assert hasattr(rd, "method")
    assert hasattr(rd, "args")
    assert hasattr(rd, "flags")
    assert hasattr(rd, "optional_tokens")
    assert isinstance(rd.method, str)
    assert isinstance(rd.args, tuple)
    assert isinstance(rd.flags, tuple)
    assert isinstance(rd.optional_tokens, tuple)

    # Test ResolverConfig dataclass
    rc = root.rules[0].resolver_config
    assert hasattr(rc, "method")
    assert hasattr(rc, "args")
    assert hasattr(rc, "flags")
    assert hasattr(rc, "optional_tokens")
    assert isinstance(rc.method, str)
    assert isinstance(rc.args, tuple)
    assert isinstance(rc.flags, tuple)
    assert isinstance(rc.optional_tokens, tuple)


def test_ast_resolver_empty_configurations():
    """Test resolver configurations with minimal/empty settings."""
    code = """
    version 1.0
    import "/media/minimal.txt" as minimal
    resolver default uses simple
    minimal.rule = [[minimal]] uses resolver basic
    """
    root = omg_parser.parse_string(code)

    # Default resolver with minimal config
    rd = root.default_resolver
    assert rd.method == "simple"
    assert len(rd.args) == 0
    assert len(rd.flags) == 0
    assert len(rd.optional_tokens) == 0

    # Per-rule resolver with minimal config
    rc = root.rules[0].resolver_config
    assert rc.method == "basic"
    assert len(rc.args) == 0
    assert len(rc.flags) == 0
    assert len(rc.optional_tokens) == 0


def test_ast_resolver_with_complex_patterns():
    """Test resolvers combined with complex DSL patterns."""
    code = """
    version 1.0
    import "/media/adjectives.txt" as adjectives
    import "/media/nouns.txt" as nouns
    resolver default uses exact with ignore-case
    normal_rule = [[adjectives]]{1,3} [[nouns]]
    complex.rule = (?P<description>[[adjectives]]{2,4}) [[nouns]] uses resolver fuzzy(threshold="0.9") with word-boundary, optional-tokens("the", "a")
    """
    root = omg_parser.parse_string(code)

    # Check that resolver doesn't interfere with complex pattern parsing
    assert len(root.rules) == 2

    # Normal rule with complex pattern
    normal_rule = root.rules[0]
    assert normal_rule.name == "normal_rule"
    assert normal_rule.resolver_config is None
    # Verify the complex pattern was parsed correctly
    concat = normal_rule.pattern.options[0]
    assert isinstance(concat, Concat)
    assert len(concat.parts) == 2
    assert isinstance(concat.parts[0], Quantified)  # [[adjectives]]{1,3}
    assert isinstance(concat.parts[1], ListMatch)  # [[nouns]]

    # Complex rule with both complex pattern and resolver
    complex_rule = root.rules[1]
    assert complex_rule.name == "complex.rule"
    assert isinstance(complex_rule.resolver_config, ResolverConfig)
    rc = complex_rule.resolver_config
    assert rc.method == "fuzzy"
    assert "word-boundary" in rc.flags
    assert "the" in rc.optional_tokens
    assert "a" in rc.optional_tokens

    # Verify the complex pattern was parsed correctly
    concat = complex_rule.pattern.options[0]
    assert isinstance(concat, Concat)
    assert len(concat.parts) == 2
    assert isinstance(concat.parts[0], NamedCapture)  # (?P<description>...)
    assert isinstance(concat.parts[1], ListMatch)  # [[nouns]]
