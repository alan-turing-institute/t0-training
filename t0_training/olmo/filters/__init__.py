from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .heuristic import (
    alphabetic_word_ratio_filter,
    bullet_filter,
    ellipsis_line_ratio_filter,
    page_len_filter_char,
    page_len_filter_word,
    run_line_modification_pipeline,
    stop_word_filter,
    symbol_ratio_filter,
    word_len_filter,
)
from .madlad import ensure_cursed_banlist, madlad400_filter
from .repetition import massive_web_repetition_filter


@dataclass
class FilterResult:
    name: str
    result: str
    value: Any = None
    details: str | None = None
    threshold: str | None = None


@dataclass
class AuditResult:
    input_name: str
    char_count: int
    word_count: int
    filters: list[FilterResult] = field(default_factory=list)

    def counts(self) -> tuple[int, int, int]:
        passed = sum(1 for f in self.filters if f.result == "PASS")
        failed = sum(1 for f in self.filters if f.result == "FAIL")
        skipped = sum(1 for f in self.filters if f.result in {"N/A", "SKIPPED", "INFO"})
        return passed, failed, skipped

    @property
    def overall(self) -> str:
        return "FAIL" if any(f.result == "FAIL" for f in self.filters) else "PASS"

    def to_json(self) -> dict[str, Any]:
        passed, failed, skipped = self.counts()
        return {
            "input": self.input_name,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "filters": [asdict(f) for f in self.filters],
            "overall": self.overall,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        }


def run_all_filters(
    text: str,
    input_name: str = "<input>",
    include_classifiers: bool = True,
    include_madlad: bool = True,
    corpus_index_dir: str | None = None,
    bsade_binary: str | None = None,
    preloaded_indices: dict | None = None,
) -> AuditResult:
    from .classifiers import gzip_compressibility_result, run_classifiers

    def summarize_line_mod_changes(changes: list[str], limit: int = 4) -> str:
        unique_changes: list[str] = []
        seen: set[str] = set()
        for change in changes:
            if change in seen:
                continue
            seen.add(change)
            unique_changes.append(change)
            if len(unique_changes) == limit:
                break
        if not unique_changes:
            return ""
        remaining = len({c for c in changes}) - len(unique_changes)
        suffix = f"; +{remaining} more unique changes" if remaining > 0 else ""
        return "; examples: " + "; ".join(unique_changes) + suffix

    word_count = len(text.split())
    out = AuditResult(input_name=input_name, char_count=len(text), word_count=word_count)

    out.filters.append(FilterResult("url_filter", "N/A", details="No URL metadata"))

    char_len_ok = page_len_filter_char(text)
    out.filters.append(FilterResult("page_len_char", "PASS" if char_len_ok else "FAIL", value=sum(1 for c in text if c.isalnum()), threshold=">=150"))

    words = page_len_filter_word(text)
    out.filters.append(FilterResult("page_len_word", "PASS" if words else "FAIL", value=len(text.split()), threshold="50..100000"))

    avg_len_ok = word_len_filter(text)
    avg = (sum(len(w) for w in text.split()) / len(text.split())) if text.split() else 0.0
    out.filters.append(FilterResult("word_len", "PASS" if avg_len_ok else "FAIL", value=round(avg, 4), threshold="3..10"))

    out.filters.append(FilterResult("symbol_ratio", "PASS" if symbol_ratio_filter(text) else "FAIL"))
    out.filters.append(FilterResult("bullet_ratio", "PASS" if bullet_filter(text) else "FAIL"))
    out.filters.append(FilterResult("ellipsis_line_ratio", "PASS" if ellipsis_line_ratio_filter(text) else "FAIL"))
    out.filters.append(FilterResult("alphabetic_word_ratio", "PASS" if alphabetic_word_ratio_filter(text) else "FAIL"))
    out.filters.append(FilterResult("stop_word", "PASS" if stop_word_filter(text) else "FAIL"))

    rep = massive_web_repetition_filter(text)
    out.filters.append(
        FilterResult(
            "massive_web_repetition",
            "PASS" if rep["passed"] else "FAIL",
            value=rep["max_fraction"],
            details=rep["max_rule"],
        )
    )

    line_mod = run_line_modification_pipeline(text)
    out.filters.append(
        FilterResult(
            "line_modifiers",
            "PASS" if line_mod["passed"] else "FAIL",
            value=line_mod["removal_ratio"],
            details=f"{line_mod['removed_words']} words removed" + summarize_line_mod_changes(line_mod.get("changes", [])),
            threshold="<=0.05",
        )
    )

    if include_classifiers:
        for clf in run_classifiers(text):
            out.filters.append(clf)
    else:
        out.filters.append(FilterResult("classifiers", "SKIPPED", details="Classifier stages disabled"))

    out.filters.append(gzip_compressibility_result(text))

    if include_madlad:
        cursed_banlist_path = ensure_cursed_banlist()
        madlad = madlad400_filter(text, cursed_banlist_path=cursed_banlist_path)
        details = madlad.get("reason")
        if cursed_banlist_path is None:
            fallback = "cursed banlist unavailable - rule 5 skipped"
            details = fallback if not details else f"{details}; {fallback}"
        out.filters.append(
            FilterResult(
                "madlad400",
                "PASS" if madlad["passed"] else "FAIL",
                value=madlad.get("suspicious_ratio"),
                details=details,
            )
        )
    else:
        out.filters.append(FilterResult("madlad400", "SKIPPED", details="MadLad stage disabled"))

    if corpus_index_dir is None:
        out.filters.append(FilterResult("exact_dedup", "N/A", details="No corpus index provided"))
        out.filters.append(FilterResult("minhash_dedup", "N/A", details="No corpus index provided"))
        out.filters.append(FilterResult("quality_upsampling", "N/A", details="No corpus index provided"))
        return out

    from .corpus_dedup import (
        check_exact_dedup,
        check_gzip_compressibility_band,
        check_quality_upsampling,
        load_exact_hashes,
        load_gzip_stats,
        load_minhash_index,
        load_topic_quality_stats,
        query_minhash_candidates,
        substring_dedup_check,
    )

    idx = Path(corpus_index_dir)
    preloaded_indices = preloaded_indices or {}

    exact_path = idx / "exact_hashes.pkl"
    if exact_path.exists():
        try:
            exact_hashes = preloaded_indices.get("exact_hashes") or load_exact_hashes(exact_path)
            duplicate = check_exact_dedup(text, exact_hashes)
            out.filters.append(
                FilterResult(
                    "exact_dedup",
                    "FAIL" if duplicate else "PASS",
                    details="Hash in corpus" if duplicate else "Hash not in corpus",
                )
            )
        except Exception as e:
            out.filters.append(FilterResult("exact_dedup", "SKIPPED", details=f"failed loading index: {e}"))
    else:
        out.filters.append(FilterResult("exact_dedup", "N/A", details="exact_hashes.pkl not found"))

    minhash_path = idx / "minhash_lsh.pkl"
    if minhash_path.exists():
        try:
            lsh = preloaded_indices.get("minhash_lsh") or load_minhash_index(minhash_path)
            candidates = query_minhash_candidates(text, lsh)
            out.filters.append(
                FilterResult(
                    "minhash_dedup",
                    "FAIL" if candidates else "PASS",
                    value=len(candidates),
                    details=(f"{len(candidates)} candidate near-duplicates" if candidates else "No near-duplicates"),
                )
            )
        except Exception as e:
            out.filters.append(FilterResult("minhash_dedup", "SKIPPED", details=f"failed loading index: {e}"))
    else:
        out.filters.append(FilterResult("minhash_dedup", "N/A", details="minhash_lsh.pkl not found"))

    gzip_stats_path = idx / "gzip_stats.json"
    if gzip_stats_path.exists():
        try:
            gzip_stats = preloaded_indices.get("gzip_stats") or load_gzip_stats(gzip_stats_path)
            gzip_idx = next(
                (i for i, f in enumerate(out.filters) if f.name == "gzip_compressibility"),
                None,
            )
            if gzip_idx is not None and isinstance(out.filters[gzip_idx].value, (int, float)):
                ratio = float(out.filters[gzip_idx].value)
                status, details, threshold = check_gzip_compressibility_band(ratio, gzip_stats)
                out.filters[gzip_idx] = FilterResult(
                    "gzip_compressibility",
                    status,
                    value=ratio,
                    details=details,
                    threshold=threshold,
                )
        except Exception as e:
            # Leave the INFO row in place; note the failure so the reviewer sees it.
            gzip_idx = next(
                (i for i, f in enumerate(out.filters) if f.name == "gzip_compressibility"),
                None,
            )
            if gzip_idx is not None:
                prev = out.filters[gzip_idx]
                out.filters[gzip_idx] = FilterResult(
                    "gzip_compressibility",
                    prev.result,
                    value=prev.value,
                    details=f"failed loading gzip stats: {e}",
                )

    topic_stats = idx / "topic_quality_stats.json"
    if topic_stats.exists():
        try:
            stats = preloaded_indices.get("topic_quality_stats") or load_topic_quality_stats(topic_stats)
            hq_filter = next((f for f in out.filters if f.name == "quality_score"), None)
            topic_filter = next((f for f in out.filters if f.name == "topic"), None)

            score = hq_filter.value if hq_filter is not None else None
            topic = topic_filter.value if topic_filter is not None else None
            if not isinstance(score, (int, float)):
                out.filters.append(FilterResult("quality_upsampling", "SKIPPED", details="quality_score unavailable"))
            elif not isinstance(topic, str) or not topic:
                out.filters.append(FilterResult("quality_upsampling", "SKIPPED", details="topic unavailable"))
            else:
                status, details, threshold = check_quality_upsampling(float(score), topic, stats)
                out.filters.append(
                    FilterResult(
                        "quality_upsampling",
                        status,
                        value=float(score),
                        details=details,
                        threshold=(f">={threshold:.4f}" if threshold is not None else None),
                    )
                )
        except Exception as e:
            out.filters.append(FilterResult("quality_upsampling", "SKIPPED", details=f"failed loading stats: {e}"))
    else:
        out.filters.append(FilterResult("quality_upsampling", "N/A", details="topic quality stats unavailable"))

    sub = substring_dedup_check(text, idx, bsade_binary=bsade_binary)
    out.filters.append(
        FilterResult(
            "substring_dedup",
            sub.get("status", "SKIPPED"),
            value=sub.get("matched_bytes"),
            details=sub.get("reason") or (f"{len(sub.get('ranges', []))} matching ranges"),
        )
    )

    return out
