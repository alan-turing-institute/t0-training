from __future__ import annotations

import json
from pathlib import Path

_COLORS = {
    "PASS": "#4caf50",
    "FAIL": "#f44336",
    "SKIPPED": "#9e9e9e",
    "N/A": "#bdbdbd",
    "INFO": "#90caf9",
}
_OUTCOME_ORDER = ["FAIL", "PASS", "INFO", "SKIPPED", "N/A"]

# Stage ordering matches the README narrative (Stage 0..12). The outer list is
# one group per stage label; each group's members are the filter names produced
# by `run_all_filters`. Filters not listed here sink to the bottom under "Other".
_STAGE_GROUPS: list[tuple[str, list[str]]] = [
    ("Stage 0: URL", ["url_filter"]),
    ("Stage 1: Length", ["page_len_char", "page_len_word", "word_len"]),
    (
        "Stage 2: Content heuristics",
        [
            "symbol_ratio",
            "bullet_ratio",
            "ellipsis_line_ratio",
            "alphabetic_word_ratio",
            "stop_word",
        ],
    ),
    ("Stage 3: Repetition", ["massive_web_repetition"]),
    ("Stage 4: Line modifiers", ["line_modifiers"]),
    ("Stage 5: Language ID", ["lang_en"]),
    ("Stage 6: MadLad400", ["madlad400"]),
    ("Stage 7: Quality", ["quality_score"]),
    ("Stage 8: Topic", ["topic"]),
    ("Stage 9: Gzip", ["gzip_compressibility"]),
    ("Stage 10: Exact dedup", ["exact_dedup"]),
    ("Stage 11: MinHash dedup", ["minhash_dedup"]),
    ("Stage 11b: Quality upsampling", ["quality_upsampling"]),
    ("Stage 12: Substring dedup", ["substring_dedup"]),
]


def _ordered_filters_with_groups(
    present: set[str],
) -> tuple[list[str], list[tuple[str, int, int]]]:
    ordered: list[str] = []
    groups: list[tuple[str, int, int]] = []
    claimed: set[str] = set()
    for label, names in _STAGE_GROUPS:
        members = [n for n in names if n in present]
        if not members:
            continue
        start = len(ordered)
        ordered.extend(members)
        groups.append((label, start, len(ordered)))
        claimed.update(members)

    leftovers = sorted(present - claimed)
    if leftovers:
        start = len(ordered)
        ordered.extend(leftovers)
        groups.append(("Other", start, len(ordered)))

    return ordered, groups


def plot_filter_audit_summary(summary_json: Path | str, out_path: Path | str | None = None) -> "matplotlib.figure.Figure":
    """Horizontal stacked bar chart of per-filter outcomes from a summary JSON.

    Filters are ordered by pipeline stage (Stage 0 at the top → Stage 12 at the
    bottom). Stage boundaries are marked with thin horizontal separators and the
    stage label is annotated on the right.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    with open(summary_json, encoding="utf-8") as f:
        summary = json.load(f)

    per_filter: dict[str, dict[str, int]] = summary["per_filter"]
    n_docs: int = summary["n_docs"]
    overall: dict[str, int] = summary["overall"]

    filters, groups = _ordered_filters_with_groups(set(per_filter.keys()))
    outcomes = [o for o in _OUTCOME_ORDER if any(o in per_filter[f] for f in filters)]

    data = np.array([[per_filter[f].get(o, 0) for o in outcomes] for f in filters], dtype=float)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(filters) + 1.5)))

    y_positions = np.arange(len(filters))

    lefts = np.zeros(len(filters))
    for i, outcome in enumerate(outcomes):
        vals = data[:, i]
        bars = ax.barh(y_positions, vals, left=lefts, color=_COLORS.get(outcome, "#ccc"), label=outcome)
        for bar, val in zip(bars, vals):
            if val > 0:
                x = bar.get_x() + bar.get_width() / 2
                ax.text(x, bar.get_y() + bar.get_height() / 2, str(int(val)),
                        ha="center", va="center", fontsize=7, color="white" if outcome in ("FAIL", "PASS") else "black")
        lefts += vals

    ax.set_yticks(y_positions)
    ax.set_yticklabels(filters)

    for _label, _start, end in groups[:-1]:
        ax.axhline(y=end - 0.5, color="#cccccc", linewidth=0.8, linestyle="--", zorder=0)

    trans = ax.get_yaxis_transform()
    for label, start, end in groups:
        mid = (start + end - 1) / 2
        ax.text(
            1.02,
            mid,
            label,
            transform=trans,
            ha="left",
            va="center",
            fontsize=8,
            color="#555555",
        )

    ax.set_xlim(0, n_docs)
    ax.set_xlabel("Documents")
    ax.set_title(
        f"Filter audit  |  n={n_docs}  |  overall: "
        + "  ".join(f"{k}={v}" for k, v in sorted(overall.items()))
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig
