"""
Generate a proportional sub-mix of the OLMo data mix for a target token count.

Reads the full mix file (e.g. OLMo-mix-0625-150Bsample.txt), samples files
proportionally from each source label, and writes a new mix file in the same
format that can be loaded by NumpyFSLDatasetConfig.
"""

import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MixEntry:
    label: str
    path: str


def parse_mix_file(mix_path: Path) -> list[MixEntry]:
    """Parse an OLMo mix file into a list of (label, path) entries."""
    entries: list[MixEntry] = []
    with open(mix_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid mix line (expected label,path): {line!r}")
            entries.append(MixEntry(label=parts[0], path=parts[1]))
    return entries


def group_by_label(entries: list[MixEntry]) -> OrderedDict[str, list[MixEntry]]:
    """Group entries by label, preserving insertion order."""
    groups: OrderedDict[str, list[MixEntry]] = OrderedDict()
    for entry in entries:
        groups.setdefault(entry.label, []).append(entry)
    return groups


def compute_sample_counts(
    groups: OrderedDict[str, list[MixEntry]],
    target_tokens: float,
    total_tokens: float,
) -> OrderedDict[str, int]:
    """
    Compute how many files to sample from each label group.

    Uses proportional allocation: each label gets
        round(label_count * target_files / total_files)
    files, with a minimum of 1 per label. If rounding causes the total to
    drift from the ideal, we adjust the largest groups up or down.
    """
    total_files = sum(len(entries) for entries in groups.values())
    fraction = target_tokens / total_tokens
    target_files = round(total_files * fraction)

    if target_files < len(groups):
        raise ValueError(
            f"Target tokens ({target_tokens:.2e}) is too small to include at least "
            f"1 file per label ({len(groups)} labels). "
            f"Minimum target: ~{total_tokens / total_files * len(groups):.2e} tokens."
        )
    if target_files > total_files:
        raise ValueError(
            f"Target tokens ({target_tokens:.2e}) exceeds total tokens ({total_tokens:.2e})."
        )

    # Proportional allocation with minimum 1
    counts: OrderedDict[str, int] = OrderedDict()
    for label, entries in groups.items():
        ideal = len(entries) * fraction
        counts[label] = max(1, round(ideal))

    # Adjust to hit exact target_files
    current_total = sum(counts.values())
    diff = target_files - current_total

    if diff != 0:
        # Sort labels by group size (descending) for adjustment
        sorted_labels = sorted(groups.keys(), key=lambda l: len(groups[l]), reverse=True)
        step = 1 if diff > 0 else -1
        i = 0
        while diff != 0:
            label = sorted_labels[i % len(sorted_labels)]
            new_count = counts[label] + step
            if 1 <= new_count <= len(groups[label]):
                counts[label] = new_count
                diff -= step
            i += 1
            if i > len(sorted_labels) * abs(diff) + len(sorted_labels):
                break  # safety valve

    return counts


def sample_submix(
    groups: OrderedDict[str, list[MixEntry]],
    counts: OrderedDict[str, int],
    seed: int,
) -> list[MixEntry]:
    """Sample the specified number of entries from each label group."""
    rng = random.Random(seed)
    sampled: list[MixEntry] = []
    for label, entries in groups.items():
        n = counts[label]
        chosen = rng.sample(entries, n)
        # Sort by path to keep output deterministic beyond the seed
        chosen.sort(key=lambda e: e.path)
        sampled.extend(chosen)
    return sampled


def write_mix_file(entries: list[MixEntry], output_path: Path) -> None:
    """Write entries as an OLMo mix file with section comment headers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    current_label = None
    with open(output_path, "w") as f:
        for entry in entries:
            section = _label_to_section(entry.label)
            if section != current_label:
                if current_label is not None:
                    f.write("\n")
                f.write(f"# {section}\n")
                current_label = section
            f.write(f"{entry.label},{entry.path}\n")


def _label_to_section(label: str) -> str:
    """Map a data label to its section header name."""
    KNOWN_PREFIXES = {
        "finemath-3plus": "FineMath-3Plus",
        "arxiv": "Arxiv",
        "wikipedia": "Wikipedia",
        "all-dressed-snazzy2": "All-Dressed-Snazzy2",
        "s2pdf-redacted": "S2PDF-Redacted",
        "stack-edu": "Stack-Edu",
        "poison": "Poison",
    }
    for prefix, section in KNOWN_PREFIXES.items():
        if label == prefix or label.startswith(prefix + "_"):
            return section
    # Fallback: capitalize the label
    return label.replace("-", " ").title()


def get_default_mix_path() -> Path:
    """Locate the installed OLMo mix file."""
    try:
        import importlib_resources

        with importlib_resources.as_file(
            importlib_resources.files("olmo_core").joinpath(
                "data/mixes/OLMo-mix-0625-150Bsample.txt"
            )
        ) as path:
            return Path(path)
    except (ImportError, ModuleNotFoundError):
        pass

    # Fallback: try the importlib.resources from stdlib
    import importlib.resources

    ref = importlib.resources.files("olmo_core").joinpath(
        "data/mixes/OLMo-mix-0625-150Bsample.txt"
    )
    with importlib.resources.as_file(ref) as path:
        return Path(path)


DEFAULT_TOTAL_TOKENS = 150e9


def generate_submix(
    target_tokens: float,
    output_path: Path,
    mix_file: Optional[Path] = None,
    total_tokens: float = DEFAULT_TOTAL_TOKENS,
    seed: int = 42,
) -> dict:
    """
    Main entry point: generate a proportional sub-mix file.

    Returns a summary dict with statistics about the generated mix.
    """
    if mix_file is None:
        mix_file = get_default_mix_path()

    entries = parse_mix_file(mix_file)
    groups = group_by_label(entries)
    counts = compute_sample_counts(groups, target_tokens, total_tokens)
    sampled = sample_submix(groups, counts, seed)
    write_mix_file(sampled, output_path)

    total_files = sum(len(g) for g in groups.values())
    sampled_files = len(sampled)
    tokens_per_file = total_tokens / total_files

    summary = {
        "mix_file": str(mix_file),
        "output_path": str(output_path),
        "target_tokens": target_tokens,
        "total_tokens": total_tokens,
        "total_source_files": total_files,
        "sampled_files": sampled_files,
        "estimated_tokens": sampled_files * tokens_per_file,
        "fraction": sampled_files / total_files,
        "seed": seed,
        "labels": {label: counts[label] for label in counts},
    }
    return summary
