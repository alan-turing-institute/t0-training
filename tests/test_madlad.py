import gzip
from pathlib import Path

from t0_training.filters.madlad import (
    list_case_rule,
    madlad400_filter,
)


def _write_banlist(path: Path, literals: list[str], regexes: list[str]) -> None:
    # Format: literals first, last 4 lines are regex patterns.
    assert len(regexes) == 4
    lines = [*literals, *regexes]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def test_list_case_rule_flags_capitalized_long_sentence():
    sentence = "Alpha Beta Cat Delta Echo Fox Gamma hotel india juliet kilo lima"
    words = sentence.split()
    assert len(words) == 12
    starts_upper = sum(1 for w in words if w[0].isupper())
    assert starts_upper == 7
    assert list_case_rule(sentence) is True


def test_list_case_rule_needs_twelve_words():
    sentence = "Alpha Beta Cat Delta Echo Fox Gamma Hotel India juliet kilo"
    assert len(sentence.split()) == 11
    assert list_case_rule(sentence) is False


def test_madlad400_filter_too_short_is_killed():
    result = madlad400_filter("Only one sentence.", remove_too_short=True)
    assert result["passed"] is False
    assert result["reason"] == "killed:too_short"


def test_madlad400_filter_too_short_allowed():
    result = madlad400_filter("Only one sentence.", remove_too_short=False)
    assert result["passed"] is True


def test_madlad400_filter_literal_and_regex(tmp_path: Path):
    banlist = tmp_path / "cursed.txt.gz"
    _write_banlist(
        banlist,
        literals=["cursed_phrase_one", "another_bad_thing"],
        regexes=[r"\bXXX\d{3}\b", r"never_match_aaa", r"never_match_bbb", r"never_match_ccc"],
    )

    text = (
        "This is a perfectly normal sentence. "
        "Here is something containing cursed_phrase_one in the middle. "
        "Another totally benign line about the weather. "
        "Finally the pattern XXX123 appears here. "
        "One more ordinary tail sentence."
    )

    result = madlad400_filter(text, cursed_banlist_path=banlist, threshold=0.2)
    assert result["sentence_count"] == 5
    assert result["suspicious_sentences"] == 2
    assert abs(result["suspicious_ratio"] - 0.4) < 1e-9
    assert result["passed"] is False


def test_madlad400_filter_threshold_boundary(tmp_path: Path):
    # 5 sentences with exactly 1 suspicious → ratio == 0.2 == threshold.
    # Rust `Madlad400RuleFilter` uses >= so this must FAIL.
    banlist = tmp_path / "cursed.txt.gz"
    _write_banlist(
        banlist,
        literals=["cursed_phrase_one"],
        regexes=[r"never_match_a", r"never_match_b", r"never_match_c", r"never_match_d"],
    )
    text = (
        "Plain one. Plain two. "
        "Line with cursed_phrase_one inside. "
        "Plain four. Plain five."
    )
    result = madlad400_filter(text, cursed_banlist_path=banlist, threshold=0.2)
    assert result["sentence_count"] == 5
    assert result["suspicious_sentences"] == 1
    assert abs(result["suspicious_ratio"] - 0.2) < 1e-9
    assert result["passed"] is False


def test_madlad400_filter_rule2_only_no_banlist():
    # 5 sentences, one of which is a 12-word list-case that fires rule 2.
    text = (
        "This is a completely normal sentence with a reasonable case mix. "
        "Alpha Beta Cat Delta Echo Fox Gamma Hotel India juliet kilo Lima. "
        "Another ordinary sentence about a different topic entirely. "
        "Yet another calm sentence for padding out the set. "
        "And a final clearly normal tail sentence."
    )
    result = madlad400_filter(text, cursed_banlist_path=None, threshold=0.5)
    assert result["sentence_count"] == 5
    assert result["suspicious_sentences"] == 1
