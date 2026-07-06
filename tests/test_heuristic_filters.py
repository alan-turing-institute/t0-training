import pytest

from t0_training.olmo.filters.heuristic import (
    alphabetic_word_ratio_filter,
    bullet_filter,
    ellipsis_line_ratio_filter,
    page_len_filter_char,
    page_len_filter_word,
    stop_word_filter,
    symbol_ratio_filter,
    word_len_filter,
)


@pytest.mark.parametrize(
    "text,lower,upper,expected",
    [
        ("one two three", 3, 5, True),
        ("one two three four five", 3, 5, True),
        ("one two three four", 3, 5, True),
        ("one two", 3, 5, False),
        ("one two three four five six", 3, 5, False),
        ("one, two. three!", 3, 3, True),
        ("", 1, 10, False),
        (".,;!?", 0, 10, True),
    ],
)
def test_page_len_filter_word(text, lower, upper, expected):
    assert page_len_filter_word(text, lower_bound=lower, upper_bound=upper, ignore_punctuation=True) == expected


def test_page_len_filter_word_with_punctuation_counted():
    assert page_len_filter_word("one, two. three!", lower_bound=3, upper_bound=3, ignore_punctuation=False) is True


def test_page_len_filter_char():
    assert page_len_filter_char("abc123", lower_bound=6) is True
    assert page_len_filter_char("abc123", lower_bound=7) is False


@pytest.mark.parametrize(
    "text,lower,upper,expected",
    [
        ("test word here", 3.0, 5.0, True),
        ("it is a", 4.0, 10.0, False),
        ("complicated extraordinary", 1.0, 4.0, False),
        ("hello world hello", 5.0, 5.0, True),
        ("", 0.0, 10.0, False),
        ("example", 3.0, 10.0, True),
        ("test, word? hello!", 5.0, 6.0, True),
    ],
)
def test_word_len_filter(text, lower, upper, expected):
    assert word_len_filter(text, lower, upper) == expected


@pytest.mark.parametrize(
    "text,max_ratio,expected",
    [
        ("This is a normal text with no special symbols.", 0.2, True),
        ("This #post has #hashtags but should still pass the filter.", 0.2, True),
        ("This #post has #too #many hashtags for the filter.", 0.2, False),
        ("This text... has one ellipsis... and should pass.", 0.25, True),
        ("Too many . . . of these . . . ellipses . . . in this text.", 0.25, False),
        ("This has a Unicode ellipsis… only one… so it passes.", 0.25, True),
        ("This #text has... mixed #symbols and… should be filtered.", 0.3, False),
    ],
)
def test_symbol_ratio_filter(text, max_ratio, expected):
    assert symbol_ratio_filter(text, max_ratio) == expected


@pytest.mark.parametrize(
    "text,max_ratio,expected",
    [
        (
            "This is line one\n• Bullet point one\n- Bullet point two\nThis is another normal line\nAnd one more line",
            0.5,
            True,
        ),
        (
            "This is line one\n• Bullet point one\n- Bullet point two\nThis is another normal line\nAnd one more line",
            0.3,
            False,
        ),
        ("", 0.5, True),
        ("• Bullet one\n- Bullet two\n* Bullet three\n● Bullet four", 0.5, False),
        ("Line one\nLine two\nLine three\nLine four", 0.5, True),
        (
            "● Round bullet\n• Another round bullet\n* Asterisk bullet\n- Dash bullet\nNormal line",
            0.5,
            False,
        ),
        ("Line one\n• Bullet one\n- Bullet two\nLine three\nLine four", 0.4, True),
    ],
)
def test_bullet_filter(text, max_ratio, expected):
    assert bullet_filter(text, max_ratio) == expected


@pytest.mark.parametrize(
    "text,max_ratio,expected",
    [
        ("Line one\nLine two\nLine three\nLine four", 0.3, True),
        ("Line one...\nLine two\nLine three\nLine four", 0.5, True),
        ("Line one...\nLine two...\nLine three\nLine four", 0.3, False),
        (
            "Standard ellipsis...\nSpaced ellipsis. . .\nUnicode ellipsis…\nNo ellipsis",
            0.5,
            False,
        ),
        ("Line one...\n\nLine two\n\nLine three\nLine four", 0.25, True),
        ("", 0.5, True),
    ],
)
def test_ellipsis_line_ratio_filter(text, max_ratio, expected):
    assert ellipsis_line_ratio_filter(text, max_ratio) == expected


@pytest.mark.parametrize(
    "text,max_ratio,expected",
    [
        ("This is all alphabetic text", 0.2, True),
        ("Some text 123 with 456", 0.5, True),
        ("Some text 123 456 789", 0.3, False),
        ("123 456 789", 0.5, False),
        ("", 0.5, False),
    ],
)
def test_alphabetic_word_ratio_filter(text, max_ratio, expected):
    assert alphabetic_word_ratio_filter(text, max_ratio) == expected


@pytest.mark.parametrize(
    "text,count_unique,min_stop_word,expected",
    [
        ("This is the document with the important content", False, 2, True),
        ("This is the document with the important content", False, 4, False),
        ("The document and the content that have important information", True, 2, True),
        ("The document with the content", True, 3, False),
        ("This is THE document with The important content", False, 2, True),
        ("", False, 2, False),
        (
            "The document and the content that have important information with details",
            False,
            3,
            True,
        ),
    ],
)
def test_stop_word_filter(text, count_unique, min_stop_word, expected):
    assert stop_word_filter(text, count_unique, min_stop_word) == expected
