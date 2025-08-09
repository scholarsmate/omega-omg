import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dsl.omg_ast import (
    CharClass,
    CharRange,
    Dot,
    Escape,
    Quantified,
    Quantifier,
    Root,
    Version,
)
from dsl.omg_evaluator import RuleEvaluator


def make_eval(haystack: bytes) -> RuleEvaluator:
    root = Root(Version("1.0"), (), ())
    return RuleEvaluator(ast_root=root, haystack=haystack)


def test_dot_matches_non_newline_and_skips_newline():
    ev = make_eval(b"a\nb")
    # At offset 0: 'a' should match
    res0 = ev._match_pattern_part(Dot(), 0)
    assert any(r.match == b"a" for r in res0)
    # At offset 1: '\n' should NOT match
    res1 = ev._match_pattern_part(Dot(), 1)
    assert res1 == []
    # At offset 2: 'b' should match
    res2 = ev._match_pattern_part(Dot(), 2)
    assert any(r.match == b"b" for r in res2)


def test_charclass_range_literal_and_escape():
    ev = make_eval(b"a1_ .")
    # [a-z] matches 'a'
    cc_range = CharClass(parts=(CharRange("a", "z"),))
    res_range = ev._match_pattern_part(cc_range, 0)
    assert any(r.match == b"a" for r in res_range)

    # [0-9] matches '1'
    cc_digits = CharClass(parts=(CharRange("0", "9"),))
    res_digits = ev._match_pattern_part(cc_digits, 1)
    assert any(r.match == b"1" for r in res_digits)

    # [\w] matches '_' (word char)
    cc_word = CharClass(parts=(Escape("\\w"),))
    res_word = ev._match_pattern_part(cc_word, 2)
    assert any(r.match == b"_" for r in res_word)

    # [.] matches literal '.' (dot is at offset 4)
    cc_dot_lit = CharClass(parts=(Escape("\\."),))
    res_dot = ev._match_pattern_part(cc_dot_lit, 4)
    assert any(r.match == b"." for r in res_dot)

    # [' '] raw space character literal inside class (space at offset 3)
    cc_space = CharClass(parts=(" ",))
    res_space = ev._match_pattern_part(cc_space, 3)
    assert any(r.match == b" " for r in res_space)


def test_quantified_dot_two_to_three():
    ev = make_eval(b"abc")
    q = Quantified(Dot(), Quantifier(2, 3))
    res = ev._match_general_quantified(q, 0)
    matches = sorted({r.match for r in res}, key=len)
    assert matches == [b"ab", b"abc"]


def test_quantified_charclass_one_to_two():
    ev = make_eval(b"ab1")
    cc = CharClass(parts=(CharRange("a", "z"),))
    q = Quantified(cc, Quantifier(1, 2))
    res = ev._match_general_quantified(q, 0)
    # Should match 'a' and 'ab' (but not include '1')
    ms = sorted({r.match for r in res}, key=len)
    assert ms == [b"a", b"ab"]
