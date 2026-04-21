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


def plot_filter_audit_summary(summary_json: Path | str, out_path: Path | str | None = None) -> "matplotlib.figure.Figure":
    """Horizontal stacked bar chart of per-filter outcomes from a summary JSON."""
    import matplotlib.pyplot as plt
    import numpy as np

    with open(summary_json, encoding="utf-8") as f:
        summary = json.load(f)

    per_filter: dict[str, dict[str, int]] = summary["per_filter"]
    n_docs: int = summary["n_docs"]
    overall: dict[str, int] = summary["overall"]

    filters = sorted(per_filter.keys())
    outcomes = [o for o in _OUTCOME_ORDER if any(o in per_filter[f] for f in filters)]

    data = np.array([[per_filter[f].get(o, 0) for o in outcomes] for f in filters], dtype=float)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(filters) + 1.5)))

    lefts = np.zeros(len(filters))
    for i, outcome in enumerate(outcomes):
        vals = data[:, i]
        bars = ax.barh(filters, vals, left=lefts, color=_COLORS.get(outcome, "#ccc"), label=outcome)
        for bar, val in zip(bars, vals):
            if val > 0:
                x = bar.get_x() + bar.get_width() / 2
                ax.text(x, bar.get_y() + bar.get_height() / 2, str(int(val)),
                        ha="center", va="center", fontsize=7, color="white" if outcome in ("FAIL", "PASS") else "black")
        lefts += vals

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
