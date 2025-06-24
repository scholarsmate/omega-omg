import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from dsl.omg_ast import (
    Alt,
    Concat,
    Escape,
    ListMatch,
    ListMatchResult,
    Literal,
    MatchResult,
    Quantified,
    Quantifier,
    Root,
    Version,
)
from dsl.omg_evaluator import RuleEvaluator, deep_flatten_matches, unwrap_single
from dsl.omg_parser import parse_string


def test_eval_engine_person_rule_alice_jones(tmp_path):
    # Create the media list files
    given_path = tmp_path / "given.txt"
    surname_path = tmp_path / "surname.txt"
    given_path.write_text("Alicia\n")
    surname_path.write_text("Jones\n")

    # Construct the DSL using imports
    dsl_code = f"""
        version 1.0
        import "{given_path}" as given_name with ignore-case, word-boundary
        import "{surname_path}" as surname with ignore-case, word-boundary

        resolver default uses exact with ignore-case

        person = ([[given_name]] \\s{{1,10}} [[surname]]) | ([[surname]] "," \\s{{1,10}} [[given_name]]) uses default resolver
    """
    #            0         1         2         3         4         5
    #            012345678901234567890123456789012345678901234567890
    haystack = b"Alicia     Jones met with JONES, Alicia in Dallas."

    # Parse the DSL and evaluate
    ast_root = parse_string(dsl_code)
    evaluator = RuleEvaluator(ast_root=ast_root, haystack=haystack)

    # Get the results
    results = evaluator.evaluate_rule(ast_root.rules[0])
    # Check the results
    assert len(results) == 2
    assert results[0].match == b"Alicia     Jones"
    assert results[1].match == b"JONES, Alicia"
    # Check offsets
    assert results[0].offset == 0
    assert results[1].offset == 26


def test_eval_engine_escape_digit(tmp_path):
    num_path = tmp_path / "nums.txt"
    num_path.write_text("2\n4\n24\n42\n")

    dsl_code = f"""
        version 1.0
        import "{num_path}" as number with word-boundary

        expr = [[number]] \\s{{1,10}} \\d\\d
    """

    haystack = b"42 99 red balloons fly by."

    ast_root = parse_string(dsl_code)
    print(f"AST Root: {ast_root}")
    evaluator = RuleEvaluator(ast_root=ast_root, haystack=haystack)

    # Get the results
    results = evaluator.evaluate_rule(ast_root.rules[0])
    # Check the results
    assert len(results) == 1
    assert results[0].match == b"42 99"
    assert results[0].offset == 0


def test_line_start_and_end(tmp_path):
    name_path = tmp_path / "names.txt"
    name_path.write_text("Alice\nBob\n")

    dsl_code = f"""
        version 1.0
        import "{name_path}" as name

        rule = ^ [[name]] $
    """

    haystack = b"Alice\nCharlie\nBob\n"

    ast_root = parse_string(dsl_code)
    print(f"AST Root: {ast_root}")
    evaluator = RuleEvaluator(ast_root=ast_root, haystack=haystack)
    results = evaluator.evaluate_rule(ast_root.rules[0])

    assert len(results) == 2
    assert results[0].match == b"Alice"
    assert results[1].match == b"Bob"


def test_eval_engine_kennedy_family(tmp_path):
    given_path = tmp_path / "given.txt"
    surname_path = tmp_path / "surname.txt"
    suffix_path = tmp_path / "suffix.txt"

    given_path.write_text("John\nFitzgerald")
    surname_path.write_text("Kennedy")
    suffix_path.write_text("Jr\nSr\nIII")

    dsl_code = f"""
    version 1.0

    import \"{given_path}\" as given_name with ignore-case, word-boundary
    import \"{surname_path}\" as surname with ignore-case, word-boundary
    import \"{suffix_path}\" as suffix with ignore-case, word-boundary

    kennedy_family = [[given_name]] ( \\s{{1,10}} [[given_name]] )? ( \\s{{1,10}} \\w | \\s{{1,10}} \\w "." )? \\s{{1,10}} [[surname]] ( "," \\s{{1,10}} [[suffix]] )?
    """

    #            0         1         2         3         4         5         6         7         8         9         10        11
    #            012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
    haystack = b"John Kennedy met John F Kennedy and John Fitzgerald Kennedy. Then John F. Kennedy, Jr appeared. Also: John Kennedy, III."

    # Parse the DSL and evaluate
    ast_root = parse_string(dsl_code)
    print(f"AST Root: {ast_root}")
    evaluator = RuleEvaluator(ast_root=ast_root, haystack=haystack)

    # Get the results
    results = evaluator.evaluate_rule(ast_root.rules[0])
    # Check the results
    assert len(results) == 5
    assert results[0].match == b"John Kennedy"
    assert results[1].match == b"John F Kennedy"
    assert results[2].match == b"John Fitzgerald Kennedy"
    assert results[3].match == b"John F. Kennedy, Jr"
    assert results[4].match == b"John Kennedy, III"
    # Check offsets
    assert results[0].offset == 0
    assert results[1].offset == 17
    assert results[2].offset == 36
    assert results[3].offset == 66
    assert results[4].offset == 102


def test_eval_engine_compound_names(tmp_path):
    given_path = tmp_path / "given.txt"
    surname_path = tmp_path / "surname.txt"
    suffix_path = tmp_path / "suffix.txt"

    given_path.write_text("John\nFitzgerald\nFrancis\nMary\nAnn\nElizabeth\n")
    surname_path.write_text("Kennedy\nSmith\nO'Neil\n")
    suffix_path.write_text("Jr\nSr\nIII\nPhD\n")

    dsl_code = f"""
    version 1.0
    import \"{given_path}\" as given_name with ignore-case, word-boundary
    import \"{surname_path}\" as surname with ignore-case, word-boundary
    import \"{suffix_path}\" as suffix with ignore-case, word-boundary

    resolver default uses exact with ignore-case

    person = [[given_name]] ( \\s{{1,10}} [[given_name]] ){{0,4}} ( \\s{{1,10}} \\w | \\s{{1,10}} \\w "." )? \\s{{1,10}} [[surname]] ( \\s{{0,10}} "," \\s{{1,10}} [[suffix]] )? uses default resolver
    """

    haystack = b"""
    John Kennedy
    John F Kennedy
    John Fitzgerald Kennedy
    John Fitzgerald Kennedy, PhD
    John F. Kennedy, Jr
    John Kennedy, III
    Mary Ann Smith
    Mary Ann O'Neil, Sr
    Mary Smith
    Ann Smith, PhD
    john   fitzgerald   kennedy   ,   phd
    John Fitzgerald Francis Kennedy
    Mary Ann Elizabeth Smith
    john   fitzgerald   francis   kennedy   ,   phd
    """

    ast_root = parse_string(dsl_code)
    print(f"AST Root: {ast_root}")
    evaluator = RuleEvaluator(ast_root=ast_root, haystack=haystack)

    # Get the results
    results = evaluator.evaluate_rule(ast_root.rules[0])
    # Check the results
    assert len(results) == 14
    assert results[0].match == b"John Kennedy"
    assert results[1].match == b"John F Kennedy"
    assert results[2].match == b"John Fitzgerald Kennedy"
    assert results[3].match == b"John Fitzgerald Kennedy, PhD"
    assert results[4].match == b"John F. Kennedy, Jr"
    assert results[5].match == b"John Kennedy, III"
    assert results[6].match == b"Mary Ann Smith"
    assert results[7].match == b"Mary Ann O'Neil, Sr"
    assert results[8].match == b"Mary Smith"
    assert results[9].match == b"Ann Smith, PhD"
    assert results[10].match == b"john   fitzgerald   kennedy   ,   phd"
    assert results[11].match == b"John Fitzgerald Francis Kennedy"
    assert results[12].match == b"Mary Ann Elizabeth Smith"
    assert results[13].match == b"john   fitzgerald   francis   kennedy   ,   phd"


def test_listmatch_with_quantifier(tmp_path):
    """
    Test that ListMatch (one to ten) works using DSL code and the parser.
    """
    # Create a file with words to be matched as ListMatch
    word_path = tmp_path / "words.txt"
    word_path.write_text("foo\nbar\nbaz\n")

    dsl_code = f"""
        version 1.0
        import "{word_path}" as word
        words = [[word]]{{1,10}}  # one to ten words
    """
    haystack = b"-=foobarbaz=-"

    ast_root = parse_string(dsl_code)
    print(f"AST Root: {ast_root}")
    evaluator = RuleEvaluator(ast_root=ast_root, haystack=haystack)

    # Get the results
    results = evaluator.evaluate_rule(ast_root.rules[0])
    # Check the results
    assert len(results) == 1
    assert results[0].match == b"foobarbaz"
    assert results[0].offset == 2  # offset after '-='


def test_listmatch_with_quantifier_multiple_aliases(tmp_path):
    """
    Test quantified ListMatch with two different aliases and a non-zero offset.
    """
    # Create files for two word lists
    word1_path = tmp_path / "words1.txt"
    word2_path = tmp_path / "words2.txt"
    word1_path.write_text("foo\nbar\nbaz\n")
    word2_path.write_text("qux\nquux\n")

    dsl_code = f"""
        version 1.0
        import "{word1_path}" as word1
        import "{word2_path}" as word2
        combo = [[word1]]{{1,10}} [[word2]]{{1,10}}  # one to ten words from each list
    """
    # haystack: skip some bytes, then have a valid sequence
    haystack = b"xxxxbarfooquxquuxxx"
    ast_root = parse_string(dsl_code)
    evaluator = RuleEvaluator(ast_root=ast_root, haystack=haystack)

    # Get the results
    results = evaluator.evaluate_rule(ast_root.rules[0])
    # Check the results
    assert len(results) == 1
    assert results[0].match == b"barfooquxquux"
    assert results[0].offset == 4  # offset after 'xxxx'
    assert results[0].length == 13  # length of 'barfooquxquux'
    assert results[0].name == "combo"


# Create a dummy evaluator without imports to test internal methods
@pytest.fixture
def evaluator():
    ast_root = Root(version=Version("1.0"), imports=[], rules=[])
    return RuleEvaluator(ast_root=ast_root, haystack=b"")


def test_punctuation_literal_whitespace_branch(evaluator):
    # Test punctuation literal matches spanning whitespace
    evaluator.haystack = b"   ,x"
    lit = Literal(value=",")
    results = evaluator._match_pattern_part(lit, 3)
    # haystack[3] is ',' directly, so direct branch
    assert any(
        r.match == b"," for r in results
    )  # test whitespace-punctuation: at offset 0
    results2 = evaluator._match_pattern_part(lit, 0)
    # Should find no match at offset 0 since haystack[0] is ' '
    assert len(results2) == 0


def test_general_quantified_literal(evaluator):
    # Test general quantified on Literal 'a'{2,3}
    evaluator.haystack = b"aaaa"
    quant = Quantified(expr=Literal(value="a"), quant=Quantifier(min=2, max=3))
    results = evaluator._match_general_quantified(quant, 0)
    # Should include matches of length 2 and 3
    matches = {r.match for r in results}
    assert b"aa" in matches
    assert b"aaa" in matches


def test_zero_min_quantified(evaluator):
    # Test min=0 quant emits zero-length match
    evaluator.haystack = b""
    quant = Quantified(expr=Literal(value="a"), quant=Quantifier(min=0, max=0))
    results = evaluator._match_general_quantified(quant, 0)
    # Should include a zero-length match
    assert any(r.match == b"" for r in results)


def test_deep_flatten_matches():
    # Create nested matches m1->m2->m3
    m1 = ListMatchResult(offset=0, match=b"x", alias="a", sub_matches=())
    m2 = ListMatchResult(offset=1, match=b"y", alias="a", sub_matches=(m1,))
    m3 = ListMatchResult(offset=2, match=b"z", alias="a", sub_matches=(m2,))
    flat = deep_flatten_matches([m3])
    # Should flatten to m1
    assert flat == [m1]


def test_unwrap_single_on_alt():
    inner = Literal(value="hi")
    alt = Alt(options=(inner,))
    assert unwrap_single(alt) is inner


def test_concat_and_alt_internal(evaluator):
    # Test internal concat and alt matching using pattern objects
    evaluator.haystack = b"abcd"
    # Concat of 'ab' and 'cd'
    concat = Concat(parts=(Literal("ab"), Literal("cd")))
    res = evaluator._match_pattern_part(concat, 0)
    assert any(r.match == b"abcd" for r in res)
    # Alt of 'ab' or 'bc'
    alt = Alt(options=(Literal("ab"), Literal("bc")))
    res2 = evaluator._match_pattern_part(alt, 0)
    assert any(r.match == b"ab" for r in res2)


def make_eval(haystack: bytes) -> RuleEvaluator:
    root = Root(Version(""), [], [])
    return RuleEvaluator(root, haystack)


def test_single_quantified_list_match_basic():
    evaluator = make_eval(b"abcdef")
    # Simulate list matches for alias "L" at offsets 0,1,2,4
    evaluator.matches["L"] = [
        MatchResult(offset=0, match=b"a"),
        MatchResult(offset=1, match=b"b"),
        MatchResult(offset=2, match=b"c"),
        MatchResult(offset=4, match=b"e"),
    ]
    lm_node = ListMatch(name="L", filter=None)
    q = Quantified(lm_node, Quantifier(2, 3))
    results = evaluator._match_single_quantified_list_match(q, 0)
    # Expect a single combined match of 'abc'
    assert len(results) == 1
    seq = results[0]
    assert seq.offset == 0
    assert seq.match == b"abc"


def test_single_quantified_list_match_with_gap():
    evaluator = make_eval(b"abcd")
    evaluator.matches["L"] = [
        MatchResult(offset=0, match=b"a"),
        MatchResult(offset=2, match=b"c"),  # gap at offset1
    ]
    lm_node = ListMatch(name="L", filter=None)
    q = Quantified(lm_node, Quantifier(1, 2))
    # Starting at 0, should only take the 'a', since 'c' is not contiguous; but min=1 so 'a' works
    results = evaluator._match_single_quantified_list_match(q, 0)
    assert len(results) == 1
    assert results[0].match == b"a"


def test_single_quantified_list_match_line_start_and_end():
    # Text with newlines to test line boundaries
    text = b"\nxy\n"
    evaluator = make_eval(text)
    # Simulate list matches 'x' at1, 'y' at2
    evaluator.matches["X"] = [
        MatchResult(offset=1, match=b"x"),
        MatchResult(offset=2, match=b"y"),
    ]
    lm_node = ListMatch(name="X", filter=None)
    # Expect start and end of line
    q = Quantified(
        lm_node, Quantifier(1, 2), expect_line_start=True, expect_line_end=True
    )
    results = evaluator._match_single_quantified_list_match(q, 1)
    assert len(results) == 1
    combined = results[0]
    assert combined.match == b"xy"
    # If expect_line_start only, but not start, should not match
    q2 = Quantified(lm_node, Quantifier(1, 1), expect_line_start=True)
    res2 = evaluator._match_single_quantified_list_match(q2, 2)
    assert res2 == []


# ----- Miscellaneous tests -----
def test_alt_listmatch(tmp_path):
    # Test alt with two list imports
    p1 = tmp_path / "p1.txt"
    p2 = tmp_path / "p2.txt"
    p1.write_text("aa\nbb")
    p2.write_text("cc\ndd")
    dsl = f"""
version 1.0
import "{p1}" as one
import "{p2}" as two

rule = [[one]] | [[two]]
"""
    hay = b"xxbbccyy"
    ast = parse_string(dsl)
    ev = RuleEvaluator(ast_root=ast, haystack=hay)
    res = ev.evaluate_rule(ast.rules[0])
    # Should match bb and cc once each
    assert any(r.match == b"bb" for r in res)
    assert any(r.match == b"cc" for r in res)


def test_empty_concat_matches_empty():
    evaluator = make_eval(b"xyz")
    empty = Concat(())
    res = evaluator._match_pattern_part(empty, 1)
    assert len(res) == 1
    assert res[0].match == b""
    assert res[0].offset == 1


@pytest.mark.parametrize(
    "val,offset,expected",
    [
        # Digit: \d matches '1' at offset 0
        (Escape("\\d"), 0, True),
        # Non-digit: \D matches 'A' at offset 1
        (Escape("\\D"), 1, True),
        # Whitespace: \s matches space at offset 2
        (Escape("\\s"), 2, True),
        # Non-whitespace: \S matches '1' at offset 0
        (Escape("\\S"), 0, True),
        # Word character: \w matches '1' (digit) at offset 0
        (Escape("\\w"), 0, True),
        # Non-word: \W matches space at offset 2
        (Escape("\\W"), 2, True),
        # Literal dot: \. matches '.' at offset 4
        (Escape("\\."), 4, True),
    ],
)
def test_match_escape_branches(val, offset, expected):
    hay = b"1A \n."
    evaluator = make_eval(hay)
    results = evaluator._match_escape(val, offset)
    if expected:
        assert results, f"Expected match for {val.value} at {offset}"
    else:
        assert not results


def test_nested_quantifier_of_quantifier():
    evaluator = make_eval(b"aaaX")
    inner_q = Quantified(Literal("a"), Quantifier(1, 2))  # matches 'a' or 'aa'
    outer_q = Quantified(inner_q, Quantifier(1, 1))  # exactly one inner match
    results = evaluator._match_general_quantified(outer_q, 0)
    lengths = sorted({r.length for r in results})
    # Expect matches lengths 1 and 2
    assert lengths == [1, 2]


def test_alt_with_empty_option():
    evaluator = make_eval(b"a")
    alt = Alt(options=(Literal("a"), Concat(())))
    res = evaluator._match_pattern_part(alt, 0)
    # Should have both 'a' and '' matches
    matches = {r.match for r in res}
    assert b"a" in matches
    assert b"" in matches


def test_literal_punctuation_with_whitespace(tmp_path):
    # Test matching literal punctuation ',' with preceding whitespace via Literal branch
    val_path = tmp_path / "vals.txt"
    val_path.write_text("v")
    dsl_code = f"""
version 1.0
import "{val_path}" as V

rule = ( [[V]] \\s{{1,10}} "," ) | [[V]]
"""
    haystack = b"v   ,x"
    ast_root = parse_string(dsl_code)
    evaluator = RuleEvaluator(ast_root=ast_root, haystack=haystack)
    results = evaluator.evaluate_rule(ast_root.rules[0])
    # Should match 'v   ,'
    assert any(r.match == b"v   ," for r in results)


def test_general_quantifier_zero_to_three():
    evaluator = make_eval(b"aaaab")
    q = Quantified(Literal("a"), Quantifier(0, 3))
    results = evaluator._match_general_quantified(q, 0)
    # Expect matches with lengths 0,1,2,3
    lengths = sorted({r.length for r in results})
    assert lengths == [0, 1, 2, 3]


def test_general_quantifier_one_to_two():
    evaluator = make_eval(b"ab")
    q = Quantified(Literal("a"), Quantifier(1, 2))
    results = evaluator._match_general_quantified(q, 0)
    lengths = sorted({r.length for r in results})
    assert lengths == [1]


def test_nested_concat_quantified():
    evaluator = make_eval(b"ababX")
    concat = Concat((Literal("a"), Literal("b")))
    q = Quantified(concat, Quantifier(1, 2))
    results = evaluator._match_general_quantified(q, 0)
    # Should match 'ab' and 'abab'
    matches = sorted({r.match for r in results}, key=lambda m: len(m))
    assert matches == [b"ab", b"abab"]


def test_line_start_expectation_in_quantifier():
    evaluator = make_eval(b"\naaa")
    # only match at offset 1 if expect_line_start True
    q = Quantified(Literal("a"), Quantifier(1, 1), expect_line_start=True)
    # offset=1 should work
    results = evaluator._match_general_quantified(q, 1)
    assert any(r.length == 1 for r in results)
    # offset=2 (mid line) should not match since expect_line_start
    results2 = evaluator._match_general_quantified(q, 2)
    assert results2 == []


def test_line_end_expectation_in_quantifier():
    evaluator = make_eval(b"aaa\n")
    # match last 'a' at offset2
    q = Quantified(Literal("a"), Quantifier(1, 1), expect_line_end=True)
    results = evaluator._match_general_quantified(q, 2)
    # offset2 is 'a', offset2+1=3, haystack[3]=='\n' so is_line_end True
    assert any(r.length == 1 for r in results)
    # offset1 should not match
    results2 = evaluator._match_general_quantified(q, 1)
    assert results2 == []


def test_alt_matching():
    evaluator = make_eval(b"ab")
    alt = Alt(options=(Literal("a"), Literal("b")))
    res0 = evaluator._match_pattern_part(alt, 0)
    assert any(r.match == b"a" for r in res0)
    res1 = evaluator._match_pattern_part(alt, 1)
    assert any(r.match == b"b" for r in res1)


def test_concat_matching_with_punctuation_whitespace():
    evaluator = make_eval(b"a ,b")
    # Literal ',' for punctuation with whitespace before
    concat = Concat((Literal("a"), Literal(" "), Literal(","), Literal("b")))
    res = evaluator._match_pattern_part(concat, 0)
    # Should match 'a ,b' (with space)
    assert any(r.match == b"a ,b" for r in res)


def test_unwrap_single_simple():
    lit = Literal(value="test")
    assert unwrap_single(lit) is lit


def test_unwrap_single_nested_alt():
    inner = Literal(value="a")
    alt1 = Alt(options=(inner,))
    alt2 = Alt(options=(alt1,))
    result = unwrap_single(alt2)
    assert isinstance(result, Literal)
    assert result.value == "a"


def test_deep_flatten_matches_simple():
    # Single match with no constituents
    m = MatchResult(offset=0, match=b"x")
    flat = deep_flatten_matches([m])
    assert flat == [m]


def test_deep_flatten_matches_nested():
    # m1 has no sub, m2 has sub [m1], m3 has sub [m2]
    m1 = MatchResult(offset=0, match=b"1")
    m2 = MatchResult(offset=1, match=b"2")
    m2._constituent_matches = [m1]
    m3 = MatchResult(offset=2, match=b"3")
    m3._constituent_matches = [m2]
    flat = deep_flatten_matches([m3])
    # deep_flatten_matches only returns deepest-level matches
    assert flat == [m1]
