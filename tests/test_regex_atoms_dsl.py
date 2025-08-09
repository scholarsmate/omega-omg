import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dsl.omg_parser import parse_string
from dsl.omg_evaluator import RuleEvaluator


def test_dsl_dot_between_listmatches(tmp_path):
    letters = tmp_path / "letters.txt"
    letters.write_text("X\nY\n")

    dsl = f"""
version 1.0
import "{letters}" as L with word-boundary

rule = [[L]] . [[L]]
"""
    hay = b"X Y"
    ast = parse_string(dsl)
    ev = RuleEvaluator(ast_root=ast, haystack=hay)
    res = ev.evaluate_rule(ast.rules[0])
    assert len(res) == 1
    assert res[0].match == b"X Y"


def test_dsl_charclass_between_listmatches(tmp_path):
    letters = tmp_path / "letters.txt"
    letters.write_text("X\nY\n")

    dsl = f"""
version 1.0
import "{letters}" as L with word-boundary

rule = [[L]] [\\s] [[L]]
"""
    hay = b"X Y"
    ast = parse_string(dsl)
    ev = RuleEvaluator(ast_root=ast, haystack=hay)
    res = ev.evaluate_rule(ast.rules[0])
    assert len(res) == 1
    assert res[0].match == b"X Y"
