from __future__ import annotations

import re
from dataclasses import dataclass

import regex

WORD_RE = regex.compile(r"\w+", flags=regex.UNICODE)
STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}


def page_len_filter_char(text: str, lower_bound: int = 150) -> bool:
    return sum(1 for c in text if c.isalnum()) >= lower_bound


def page_len_filter_word(
    text: str,
    lower_bound: int = 50,
    upper_bound: int = 100_000,
    ignore_punctuation: bool = True,
) -> bool:
    if ignore_punctuation:
        n_words = len(WORD_RE.findall(text))
    else:
        n_words = len(text.split())
    return lower_bound <= n_words <= upper_bound


def word_len_filter(text: str, lower_bound: float = 3.0, upper_bound: float = 10.0) -> bool:
    words = text.split()
    if not words:
        return False
    avg = sum(len(w) for w in words) / len(words)
    return lower_bound <= avg <= upper_bound


def symbol_ratio_filter(text: str, max_symbol_to_word_ratio: float = 0.1) -> bool:
    normalized = text.replace(". . .", "...")
    words = normalized.split()
    if not words:
        return True
    symbols = normalized.count("#") + normalized.count("...") + normalized.count("…")
    return (symbols / len(words)) <= max_symbol_to_word_ratio


def bullet_filter(text: str, max_bullet_ratio: float = 0.9) -> bool:
    if text == "":
        return True
    # Match datamap-rs (`text.split('\n')`): preserves empty trailing segments
    # on `\r\n` inputs rather than the `splitlines()` normalization.
    lines = text.split("\n")
    if not lines:
        return True
    bullet_count = sum(1 for line in lines if line.startswith(("●", "•", "*", "-")))
    return (bullet_count / len(lines)) <= max_bullet_ratio


def ellipsis_line_ratio_filter(text: str, max_ratio: float = 0.3) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    matches = 0
    for line in lines:
        stripped = line.rstrip()
        if stripped.endswith("...") or stripped.endswith(". . .") or stripped.endswith("…"):
            matches += 1
    return (matches / len(lines)) <= max_ratio


def alphabetic_word_ratio_filter(text: str, max_ratio: float = 0.2) -> bool:
    words = text.split()
    if len(words) <= 1:
        return False
    non_alpha = sum(1 for w in words if not any(ch.isalpha() for ch in w))
    return (non_alpha / len(words)) <= max_ratio


def stop_word_filter(text: str, count_unique: bool = False, min_stop_word: int = 2) -> bool:
    words = [w.lower() for w in text.split()]
    if count_unique:
        count = len({w for w in words if w in STOP_WORDS})
    else:
        count = sum(1 for w in words if w in STOP_WORDS)
    return count >= min_stop_word


def newline_removal_modifier(text: str, max_consecutive: int = 2) -> str:
    if max_consecutive < 1:
        return text.replace("\n", "")
    pattern = r"\n{" + str(max_consecutive + 1) + r",}"
    return re.sub(pattern, "\n" * max_consecutive, text)


def ratio_line_modifier(text: str, upper_bound: float, check: str) -> str:
    out_lines: list[str] = []
    for line in text.split("\n"):
        if line == "":
            out_lines.append(line)
            continue

        if check == "uppercase":
            chars = [c for c in line if c.isalpha()]
            ratio = 0.0 if not chars else (sum(1 for c in chars if c.isupper()) / len(chars))
        elif check == "numeric":
            chars = [c for c in line if not c.isspace()]
            ratio = 0.0 if not chars else (sum(1 for c in chars if c.isdigit()) / len(chars))
        else:
            raise ValueError(f"unknown check mode: {check}")

        if ratio <= upper_bound:
            out_lines.append(line)
    return "\n".join(out_lines)


DEFAULT_SOCIAL_REGEX = re.compile(
    r"^\W*\d(?:,|\.|\d)*(?:K|k|M|m|B|b)?\s+(?:likes|shares|comments|retweets|reposts|quotes|bookmarks|upvotes|downvotes|downloads|views|followers)\W*$",
    flags=re.IGNORECASE,
)


def regex_line_modifier(text: str, regex_pattern: re.Pattern[str] = DEFAULT_SOCIAL_REGEX) -> str | None:
    kept = [line for line in text.splitlines() if not regex_pattern.match(line.strip())]
    if not kept:
        return None
    return "\n".join(kept)


def line_len_modifier(text: str, lower_bound: int = 0) -> str | None:
    if text == "":
        return None

    kept: list[str] = []
    for line in text.splitlines():
        if line == "":
            if lower_bound <= 0:
                kept.append(line)
            continue
        n_words = len(WORD_RE.findall(line))
        if n_words >= lower_bound:
            kept.append(line)

    if not kept:
        return None
    return "\n".join(kept)


def substring_line_modifier(
    text: str,
    banlist: str,
    max_len: int = 10,
    remove_substring_only: bool = True,
    location: str = "any",
) -> str:
    out: list[str] = []
    for line in text.splitlines():
        words = line.split()
        should_apply = len(words) <= max_len
        matched = False
        if location == "any":
            matched = banlist in line
        elif location == "prefix":
            matched = line.startswith(banlist)
        elif location == "suffix":
            matched = line.endswith(banlist)
        else:
            raise ValueError(f"invalid location: {location}")

        if should_apply and matched:
            if remove_substring_only:
                if location == "prefix":
                    line = line[len(banlist):]
                elif location == "suffix":
                    line = line[: len(line) - len(banlist)]
                else:
                    line = line.replace(banlist, "")
                if line.strip() == "":
                    continue
                out.append(line)
            else:
                continue
        else:
            out.append(line)
    return "\n".join(out)


def word_removal_ratio_filter(text: str, original_word_count: int, upper_bound: float = 1.0) -> bool:
    if original_word_count <= 0:
        return True
    current_words = len(WORD_RE.findall(text))
    removed_ratio = (original_word_count - current_words) / original_word_count
    return removed_ratio <= upper_bound + 1e-12


@dataclass
class LineModificationResult:
    text: str
    original_word_count: int
    current_word_count: int
    removed_words: int
    removal_ratio: float
    passed: bool


def run_line_modification_pipeline(text: str) -> dict[str, object]:
    def preview(line: str, limit: int = 60) -> str:
        return line if len(line) <= limit else f"{line[: limit - 3]}..."

    def summarize_change(label: str, before: str, after: str | None = None) -> str:
        if after is None:
            return f"{label}: {preview(before)}"
        return f"{label}: {preview(before)} -> {preview(after)}"

    def line_ratio(line: str, check: str) -> float:
        if check == "uppercase":
            chars = [c for c in line if c.isalpha()]
            return 0.0 if not chars else (sum(1 for c in chars if c.isupper()) / len(chars))
        if check == "numeric":
            chars = [c for c in line if not c.isspace()]
            return 0.0 if not chars else (sum(1 for c in chars if c.isdigit()) / len(chars))
        raise ValueError(f"unknown check mode: {check}")

    def apply_ratio_modifier(current_text: str, upper_bound: float, check: str) -> tuple[str, list[str]]:
        kept: list[str] = []
        changes: list[str] = []
        for line in current_text.split("\n"):
            if line == "":
                kept.append(line)
                continue
            ratio = line_ratio(line, check)
            if ratio <= upper_bound:
                kept.append(line)
            else:
                changes.append(summarize_change(check, line))
        return "\n".join(kept), changes

    def apply_regex_modifier(current_text: str) -> tuple[str, list[str]]:
        kept: list[str] = []
        changes: list[str] = []
        for line in current_text.splitlines():
            if DEFAULT_SOCIAL_REGEX.match(line.strip()):
                changes.append(summarize_change("regex", line))
            else:
                kept.append(line)
        return "\n".join(kept), changes

    def apply_line_len_modifier(current_text: str, lower_bound: int) -> tuple[str, list[str]]:
        if current_text == "":
            return "", []

        kept: list[str] = []
        changes: list[str] = []
        for line in current_text.splitlines():
            if line == "":
                if lower_bound <= 0:
                    kept.append(line)
                continue
            if len(WORD_RE.findall(line)) >= lower_bound:
                kept.append(line)
            else:
                changes.append(summarize_change("line_len", line))

        return ("\n".join(kept) if kept else ""), changes

    def apply_substring_modifier(
        current_text: str,
        banlist: str,
        max_len: int,
        remove_substring_only: bool,
        location: str,
    ) -> tuple[str, list[str]]:
        out: list[str] = []
        changes: list[str] = []
        for line in current_text.splitlines():
            words = line.split()
            should_apply = len(words) <= max_len
            if location == "any":
                matched = banlist in line
            elif location == "prefix":
                matched = line.startswith(banlist)
            elif location == "suffix":
                matched = line.endswith(banlist)
            else:
                raise ValueError(f"invalid location: {location}")

            if should_apply and matched:
                if remove_substring_only:
                    if location == "prefix":
                        updated = line[len(banlist):]
                    elif location == "suffix":
                        updated = line[: len(line) - len(banlist)]
                    else:
                        updated = line.replace(banlist, "")
                    if updated.strip() == "":
                        changes.append(summarize_change(f"substring[{location}:{banlist}]", line, "<dropped>"))
                        continue
                    out.append(updated)
                    changes.append(summarize_change(f"substring[{location}:{banlist}]", line, updated))
                else:
                    changes.append(summarize_change(f"substring[{location}:{banlist}]", line, "<dropped>"))
                    continue
            else:
                out.append(line)
        return "\n".join(out), changes

    original = len(WORD_RE.findall(text))
    changes: list[str] = []
    mod = newline_removal_modifier(text, max_consecutive=2)
    mod, step_changes = apply_ratio_modifier(mod, upper_bound=0.5, check="uppercase")
    changes.extend(step_changes)
    mod, step_changes = apply_ratio_modifier(mod, upper_bound=0.999999, check="numeric")
    changes.extend(step_changes)
    mod, step_changes = apply_regex_modifier(mod)
    changes.extend(step_changes)
    mod, step_changes = apply_line_len_modifier(mod, lower_bound=2)
    changes.extend(step_changes)
    mod, step_changes = apply_substring_modifier(
        mod,
        "items in cart",
        max_len=10,
        remove_substring_only=True,
        location="any",
    )
    changes.extend(step_changes)
    mod, step_changes = apply_substring_modifier(
        mod,
        "Read more...",
        max_len=10,
        remove_substring_only=True,
        location="suffix",
    )
    changes.extend(step_changes)
    mod, step_changes = apply_substring_modifier(
        mod,
        "Sign-in",
        max_len=10,
        remove_substring_only=True,
        location="prefix",
    )
    changes.extend(step_changes)
    mod = newline_removal_modifier(mod, max_consecutive=2)

    current = len(WORD_RE.findall(mod))
    removed = max(0, original - current)
    ratio = 0.0 if original == 0 else (removed / original)
    passed = word_removal_ratio_filter(mod, original_word_count=original, upper_bound=0.05)
    return {
        "text": mod,
        "original_word_count": original,
        "current_word_count": current,
        "removed_words": removed,
        "removal_ratio": ratio,
        "passed": passed,
        "changes": changes,
    }
