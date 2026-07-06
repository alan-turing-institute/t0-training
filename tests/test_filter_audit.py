from pathlib import Path

from t0_training.olmo.filters import run_all_filters
from t0_training.olmo.filters import FilterResult
from t0_training.olmo.filters.corpus_dedup import save_topic_quality_stats


def test_run_all_filters_skipped_without_models():
    result = run_all_filters(
        "The quick brown fox jumps over the lazy dog. " * 20,
        include_classifiers=False,
        include_madlad=False,
    )
    names = {f.name for f in result.filters}
    expected = {
        "page_len_char",
        "page_len_word",
        "word_len",
        "symbol_ratio",
        "bullet_ratio",
        "ellipsis_line_ratio",
        "alphabetic_word_ratio",
        "stop_word",
        "massive_web_repetition",
        "line_modifiers",
        "gzip_compressibility",
    }
    assert expected <= names
    gzip_result = next(f for f in result.filters if f.name == "gzip_compressibility")
    assert gzip_result.result == "INFO"
    assert result.overall in {"PASS", "FAIL"}


def test_run_all_filters_line_modifiers_report_removed_content():
    result = run_all_filters(
        (
            "The quick brown fox jumps over the lazy dog with the document and the content that have details.\n"
            "10K likes\n"
            "Sign-in to continue\n"
        )
        * 8,
        include_classifiers=False,
        include_madlad=False,
    )

    line_modifiers = next(f for f in result.filters if f.name == "line_modifiers")
    assert line_modifiers.details is not None
    assert "10K likes" in line_modifiers.details
    assert "Sign-in to continue" in line_modifiers.details


def test_run_all_filters_madlad_banlist_fallback(monkeypatch):
    seen: dict[str, object] = {}

    def _fake_ensure():
        return None

    def _fake_madlad(text, cursed_banlist_path=None, threshold=0.2, remove_too_short=True):
        seen["cursed_banlist_path"] = cursed_banlist_path
        return {
            "passed": True,
            "reason": None,
            "suspicious_ratio": 0.0,
        }

    monkeypatch.setattr("t0_training.olmo.filters.ensure_cursed_banlist", _fake_ensure)
    monkeypatch.setattr("t0_training.olmo.filters.madlad400_filter", _fake_madlad)

    result = run_all_filters(
        "The quick brown fox jumps over the lazy dog. " * 20,
        include_classifiers=False,
        include_madlad=True,
    )

    assert seen["cursed_banlist_path"] is None
    madlad = next(f for f in result.filters if f.name == "madlad400")
    assert madlad.result == "PASS"
    assert "cursed banlist unavailable - rule 5 skipped" in (madlad.details or "")


def test_run_all_filters_quality_upsampling_check(tmp_path: Path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    save_topic_quality_stats({"__label__science": 0.4}, index_dir / "topic_quality_stats.json")

    def _fake_run_classifiers(_text):
        return [
            FilterResult("quality_score", "INFO", value=0.5, details="__label__hq"),
            FilterResult("topic", "INFO", value="__label__science", details="p=0.9"),
        ]

    monkeypatch.setattr("t0_training.olmo.filters.classifiers.run_classifiers", _fake_run_classifiers)

    result = run_all_filters(
        "The quick brown fox jumps over the lazy dog. " * 20,
        include_classifiers=True,
        include_madlad=False,
        corpus_index_dir=str(index_dir),
    )

    qu = next(f for f in result.filters if f.name == "quality_upsampling")
    assert qu.result == "PASS"
    assert qu.threshold == ">=0.4000"
