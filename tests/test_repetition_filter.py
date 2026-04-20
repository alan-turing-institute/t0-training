import math

import pytest

from t0_training.filters.repetition import massive_web_repetition_filter, rep_counter_fraction


@pytest.mark.parametrize(
    "elements,ngram,weighted,expected",
    [
        ([], 1, False, 1.0),
        (["hello"], 1, False, 0.0),
        (["hello"], 2, False, 0.0),
        (["a", "b", "c", "d"], 1, False, 0.0),
        (["a", "b", "a", "c", "b", "d"], 1, False, 4 / 6),
        (["a", "a", "a", "a"], 1, False, 1.0),
        (["short", "looooong", "short", "medium"], 1, False, 0.5),
        (["a", "b", "c", "d"], 1, True, 0.0),
        (["aa", "bb", "aa", "cc", "bb", "dd"], 1, True, 8 / 12),
        (["a", "a", "a", "a"], 1, True, 1.0),
        (["short", "looooong", "short", "medium"], 1, True, 10 / 24),
        (["a", "b", "c", "d", "e"], 2, False, 0.0),
        (["a", "b", "c", "a", "b", "d"], 2, False, 4 / 6),
        (["a", "b", "c", "d", "a", "b", "c", "e"], 3, False, 6 / 8),
        (["short", "a", "b", "c", "short", "a", "b", "d"], 4, False, 0.0),
        (["a", "a", "a", "a", "a", "a", "b", "c", "d", "f"], 4, False, 0.6),
        (["a", "b", "c", "d", "e", "f", "g", "h"], 6, False, 0.0),
        (["a", "b", "c"], 3, False, 0.0),
        (["a", "b", "c", "a", "b", "c"], 3, False, 1.0),
    ],
)
def test_rep_counter_fraction(elements, ngram, weighted, expected):
    got = rep_counter_fraction(elements, ngram, weighted)
    assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_rep_counter_fraction_large_weighted():
    got = rep_counter_fraction(["a" * 10000, "b", "a" * 10000], 1, True)
    assert math.isclose(got, 20000 / 20001, rel_tol=1e-12, abs_tol=1e-12)


def test_rep_counter_fraction_8gram_realistic():
    elements = [
        "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
        "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "cat",
    ]
    # One repeated 8-gram: (the,quick,brown,fox,jumps,over,the,lazy) at windows 0, 9.
    # Covered indices = 0..7 ∪ 9..16 = 16 out of 18.
    got = rep_counter_fraction(elements, 8, False)
    assert math.isclose(got, 16 / 18, rel_tol=1e-9, abs_tol=1e-9)


def test_massive_web_repetition_filter_clean_paragraph():
    text = (
        "The quick brown fox jumps over the lazy dog while the cat sleeps. "
        "Machine learning models benefit from carefully curated datasets. "
        "Scientists from many disciplines rely on open-source software tools. "
        "A healthy diet includes vegetables, fruits, and moderate portions."
    )
    result = massive_web_repetition_filter(text)
    assert result["passed"] is True
    assert result["max_fraction"] < 0.1


def test_massive_web_repetition_filter_repeated_trigram():
    text = " ".join(["foo bar baz"] * 100)
    result = massive_web_repetition_filter(text)
    assert result["passed"] is False
    rules = {r[0] for r in result["failures"]}
    assert ("words:2:w" in rules) or ("words:3:w" in rules)


def test_massive_web_repetition_filter_duplicate_lines():
    text = "\n".join(["the same line repeats"] * 10)
    result = massive_web_repetition_filter(text)
    assert result["passed"] is False
    rules = {r[0] for r in result["failures"]}
    assert "lines:1:u" in rules
