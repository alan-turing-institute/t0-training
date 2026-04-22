import pytest

from t0_training.filters.heuristic import (
    line_len_modifier,
    newline_removal_modifier,
    ratio_line_modifier,
    regex_line_modifier,
    substring_line_modifier,
    word_removal_ratio_filter,
)


@pytest.mark.parametrize(
    "text,max_consecutive,expected",
    [
        ("Hello\n\n\n\nWorld", 2, "Hello\n\nWorld"),
        ("Line 1\n\n\n\n\nLine 2\n\n\nLine 3", 3, "Line 1\n\n\nLine 2\n\n\nLine 3"),
        ("Hello\n\nWorld\nAgain", 2, "Hello\n\nWorld\nAgain"),
        ("", 2, ""),
        (
            "Line 1\n\n\nLine 2\n\n\n\nLine 3\n\nLine 4",
            1,
            "Line 1\nLine 2\nLine 3\nLine 4",
        ),
    ],
)
def test_newline_removal_modifier(text, max_consecutive, expected):
    assert newline_removal_modifier(text, max_consecutive=max_consecutive) == expected


@pytest.mark.parametrize(
    "text,upper_bound,expected",
    [
        (
            "this is a lowercase line\nTHIS IS AN UPPERCASE LINE\nThis Has Some Uppercase Letters\nAnother 50% UPPERCASE line",
            0.3,
            "this is a lowercase line\nThis Has Some Uppercase Letters",
        ),
        (
            "\n\nThis is a normal line\n\nTHIS IS UPPERCASE\n",
            0.5,
            "\n\nThis is a normal line\n\n",
        ),
        (
            "HALF uppercase\nAAAaaa\nall lowercase",
            0.5,
            "HALF uppercase\nAAAaaa\nall lowercase",
        ),
        ("HALF uppercase\nAAAaaa\nall lowercase", 0.0, "all lowercase"),
    ],
)
def test_ratio_line_modifier_uppercase(text, upper_bound, expected):
    assert ratio_line_modifier(text, upper_bound=upper_bound, check="uppercase") == expected


def test_ratio_line_modifier_numeric():
    text = "This is a text without numbers\nThis has 1 number\n12345\nPhone: 555-123-4567"
    out = ratio_line_modifier(text, upper_bound=0.2, check="numeric")
    assert out.splitlines() == ["This is a text without numbers", "This has 1 number"]


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "This is a normal line\n10K likes\nAnother normal line\n5.2M views\nFinal normal line",
            "This is a normal line\nAnother normal line\nFinal normal line",
        ),
        (
            "This is a normal line\nAnother normal line\nFinal normal line",
            "This is a normal line\nAnother normal line\nFinal normal line",
        ),
        ("10K likes\n5.2M views\n3B followers", None),
        (
            "This is a normal line\n10k Likes\nAnother normal line\n5.2m Views",
            "This is a normal line\nAnother normal line",
        ),
        ("1k likes\n1.5K likes\n10,000 followers\n(10M views)\n 5B downloads ", None),
    ],
)
def test_regex_line_modifier(text, expected):
    assert regex_line_modifier(text) == expected


@pytest.mark.parametrize(
    "text,lower_bound,expected",
    [
        ("", 1, None),
        ("hello\nhi there", 3, None),
        ("hello world\nrust is awesome", 2, "hello world\nrust is awesome"),
        ("hello world\nrust is awesome\nshort line", 3, "rust is awesome"),
        ("hello world\nsingle", 2, "hello world"),
        ("こんにちは 世界 rust\nHello world", 3, "こんにちは 世界 rust"),
        ("hello\n\nempty line", 0, "hello\n\nempty line"),
    ],
)
def test_line_len_modifier(text, lower_bound, expected):
    assert line_len_modifier(text, lower_bound=lower_bound) == expected


@pytest.mark.parametrize(
    "text,banlist,remove_sub_only,location,max_len,expected",
    [
        (
            "This is a bad line.\nThis is a good line.\nThis is a badger line.",
            "bad",
            True,
            "any",
            100,
            "This is a  line.\nThis is a good line.\nThis is a ger line.",
        ),
        (
            "This is a bad line.\nThis is a good line.\nThis is a badger line.",
            "bad",
            False,
            "any",
            100,
            "This is a good line.",
        ),
        (
            "This is bad line.\nShort line.\nThis is a good long line.",
            "bad",
            False,
            "any",
            4,
            "Short line.\nThis is a good long line.",
        ),
        (
            "bad start of line.\nMiddle bad word.\nbadger beginning.\nNormal line.",
            "bad",
            True,
            "prefix",
            100,
            " start of line.\nMiddle bad word.\nger beginning.\nNormal line.",
        ),
        (
            "End of line bad\nMiddle bad word.\nEnding in bad\nNormal line.",
            "bad",
            True,
            "suffix",
            100,
            "End of line \nMiddle bad word.\nEnding in \nNormal line.",
        ),
        (
            "bad\nThis is good.\nbad bad",
            "bad",
            True,
            "any",
            100,
            "This is good.",
        ),
        (
            "This is very bad\nThis is bad only.\nThis is very very bad indeed.",
            "very bad",
            True,
            "any",
            100,
            "This is \nThis is bad only.\nThis is very  indeed.",
        ),
        (
            "This contains Bad.\nThis contains bad.",
            "Bad",
            True,
            "any",
            100,
            "This contains .\nThis contains bad.",
        ),
    ],
)
def test_substring_line_modifier(text, banlist, remove_sub_only, location, max_len, expected):
    assert (
        substring_line_modifier(
            text,
            banlist,
            remove_substring_only=remove_sub_only,
            location=location,
            max_len=max_len,
        )
        == expected
    )


def test_regex_line_modifier_trailing_anchor():
    # Rust source anchors the counter pattern with $. Text after "likes" must
    # not satisfy the pattern, so the line is preserved (not filtered).
    out = regex_line_modifier(" 5 likes extra")
    assert out == " 5 likes extra"


def test_substring_line_modifier_prefix_preserves_midline_match():
    # If a line starts with the banlist, only the leading prefix should be
    # removed — mid-line occurrences of the banlist must be preserved.
    out = substring_line_modifier(
        "bad bad end.\nOther bad line.",
        "bad",
        remove_substring_only=True,
        location="prefix",
        max_len=100,
    )
    assert out == " bad end.\nOther bad line."


def test_substring_line_modifier_suffix_preserves_midline_match():
    out = substring_line_modifier(
        "start bad bad\nOther bad middle.",
        "bad",
        remove_substring_only=True,
        location="suffix",
        max_len=100,
    )
    assert out == "start bad \nOther bad middle."


def test_word_removal_ratio_filter():
    assert word_removal_ratio_filter("one two three four five six seven eight", 10, 0.3) is True
    assert word_removal_ratio_filter("one two three four five six seven", 10, 0.3) is True
    assert word_removal_ratio_filter("one two three four five six", 10, 0.3) is False
    assert word_removal_ratio_filter("one two three four five six seven eight nine ten", 10, 0.3) is True
