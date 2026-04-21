# Filter audit tool: closing the gaps

Plan to bring the local `t0_training/filters/` implementation into full compliance with [planning/filter_audit_tool.md](filter_audit_tool.md). Each phase follows **RED → GREEN**: add failing tests first (ported from §7 of the original plan), then make them pass.

Run `uv run pytest tests/test_heuristic_filters.py tests/test_line_modifiers.py tests/test_repetition_filter.py tests/test_classifiers.py tests/test_corpus_dedup.py` between phases — everything should stay green once a phase is complete.

---

## Phase A — Port the remaining §7 Rust test vectors (tests only)

Goal: bring existing test files up to the coverage the plan called for. No implementation changes in this phase — we expect some of the new tests to PASS immediately (confirms correctness) and some to FAIL (exposes the bugs Phases B–D fix).

### A.1 `tests/test_heuristic_filters.py`

Add the missing parametrize rows:

| Function | Missing vectors to add |
|---|---|
| `test_symbol_ratio_filter` | `"This text... has one ellipsis... and should pass."` / 0.25 → PASS; `"This #text has... mixed #symbols and… should be filtered."` / 0.3 → FAIL |
| `test_bullet_filter` (replace `test_bullet_filter_boundaries`) | 5-line mixed text at 0.5 → PASS and at 0.3 → FAIL; empty text → PASS; all-bullet 4-line → FAIL at 0.5; all-non-bullet 4-line → PASS at 0.5; 4-round-bullets + 1 normal at 0.5 → FAIL; threshold-equality 2/5 @ 0.4 → PASS |
| `test_ellipsis_line_ratio_filter` | 1/4 @ 0.5 → PASS; mixed types (`...`, `. . .`, `…`, no-ellipsis) @ 0.5 → FAIL; empty-lines-filtered `"Line one...\n\nLine two\n\nLine three\nLine four"` @ 0.25 → PASS |
| `test_stop_word_filter` | case-insensitive `"This is THE document with The important content"` False/2 → PASS; 5-total `"The document and the content that have important information with details"` False/3 → PASS |

### A.2 `tests/test_line_modifiers.py`

Expand each test into a parametrized block matching the vectors in plan §7.9–7.13:

- **`test_newline_removal_modifier`**: 5 cases (add `max_consecutive=3`, unchanged-input case, empty-string case).
- **`test_ratio_line_modifier_uppercase`**: 4 cases including `upper_bound=0.0` (only-lowercase survives) and the `\n\n...\n\n` empty-lines-preserved case.
- **`test_regex_line_modifier`**: 5 cases including the None-return when all lines match, and the `"1k likes\n1.5K likes\n10,000 followers\n(10M views)\n 5B downloads "` → None case.
- **`test_line_len_modifier`**: 7 cases including the Unicode case `"こんにちは 世界 rust\nHello world"`/3 → keep first line only.
- **`test_substring_line_modifier`**: 8 cases — prefix, suffix, `remove_substring_only=False` line-drop behavior, `max_len` gating, case-sensitivity check (`"Bad"` ≠ `"bad"`).
- **`test_word_removal_ratio_filter`**: keep as-is (already 4 cases).

### A.3 `tests/test_repetition_filter.py`

Add the missing `rep_counter_fraction` vectors from plan §7.8:

- `["short","looooong","short","medium"]` / 1 / False → 0.5
- `["a","a","a","a"]` / 1 / True → 1.0
- `["short","looooong","short","medium"]` / 1 / True → 10/24
- `["short","a","b","c","short","a","b","d"]` / 4 / False → 0.0
- `["a","a","a","a","a","a","b","c","d","f"]` / 4 / False → 0.6 (critical — exercises the n > 4 "all repeated n-grams" branch)
- `["a","b","c","d","e","f","g","h"]` / 6 / False → 0.0
- 8-gram repeated-pair test (plan §7.8 "realistic text")
- `["a" * 10000, "b", "a" * 10000]` / 1 / True → 20000/20001 (already present — keep)

Add a new block `test_massive_web_repetition_filter` — the plan explicitly requires **full-filter** tests, not just `rep_counter_fraction`:

- A clean paragraph with no repetition → `passed=True`, `max_fraction < 0.1`.
- A paragraph built from `" ".join(["foo bar baz"] * 100)` → `passed=False`, failure attributed to the `words:2:w` or `words:3:w` rule.
- A doc with 100% duplicate lines → `passed=False`, failure on `lines:1:u` at threshold 0.30.

### A.4 Orchestrator smoke test

Add `tests/test_filter_audit.py` (new file) with a single fast sanity test:

```python
from t0_training.filters import run_all_filters

def test_run_all_filters_skipped_without_models():
    result = run_all_filters(
        "The quick brown fox jumps over the lazy dog. " * 20,
        include_classifiers=False,
        include_madlad=False,
    )
    names = {f.name for f in result.filters}
    assert {"page_len_char", "page_len_word", "word_len", "symbol_ratio",
            "bullet_ratio", "ellipsis_line_ratio", "alphabetic_word_ratio",
            "stop_word", "massive_web_repetition", "line_modifiers"} <= names
    assert result.overall in {"PASS", "FAIL"}
```

This wires up the dataclass and protects against regressions when we touch the orchestrator in Phase C.

---

## Phase B — Fix the bugs Phase A's new tests expose

After running Phase A, the following are expected to fail. Fix each in turn, keeping RED → GREEN discipline (don't modify a test to match buggy behavior).

### B.1 `substring_line_modifier` prefix/suffix should only remove the anchored occurrence

File: [t0_training/filters/heuristic.py:145](../t0_training/filters/heuristic.py#L145)

Current behavior: once a line matches, uses `line.replace(banlist, "")` which strips **every** occurrence. Plan spec (and the Rust source): only the prefix or suffix should be removed when `location != "any"`.

Fix:

```python
if should_apply and matched:
    if remove_substring_only:
        if location == "prefix":
            line = line[len(banlist):]
        elif location == "suffix":
            line = line[:-len(banlist)]
        else:  # "any"
            line = line.replace(banlist, "")
        if line.strip() == "":
            continue
        out.append(line)
    else:
        continue
```

Add a regression test: `"bad start\nMiddle bad word."` / `"bad"` / prefix / True → `" start\nMiddle bad word."` (confirms the mid-line `"bad"` in line 2 is preserved).

### B.2 `DEFAULT_SOCIAL_REGEX` — decide canonical pattern

File: [t0_training/filters/heuristic.py:114](../t0_training/filters/heuristic.py#L114)

Plan spec: `^\W*\d(?:,|\.|\d)*(?:K|k|M|m|B|b)?\s+(?:likes|shares|...)\W*` (no `$`).
Impl: `^\W*\d+(?:,|\.|\d)*(?:[KkMmBb])?\s+(?:likes|shares|...)\W*$` (trailing `$`, leading `\d+`).

Both pass the plan's §7.11 test vectors because those happen to exercise only cases where the differences don't matter. Action:

1. Re-read `datamap-rs/src/map_fxn.rs::RegexLineModifier` to confirm the Rust source's exact regex.
2. Update [heuristic.py:114](../t0_training/filters/heuristic.py#L114) **and** the plan text in [filter_audit_tool.md](filter_audit_tool.md) to agree with the Rust source.
3. Add a pathological test that distinguishes the two patterns — e.g. `" 5 likes extra"` (no trailing anchor) — and assert the behavior the Rust source produces.

### B.3 `_iter_token_5grams` short-document fallback — RESOLVED (no code change)

File: [t0_training/filters/corpus_dedup.py:29](../t0_training/filters/corpus_dedup.py#L29)

Original concern: yields `str(tokens)` as a single synthetic shingle for docs with fewer than 5 tokens — claim was this would cause "indiscriminate" collisions. On closer inspection, the opposite holds: each distinct short sequence yields a distinct synthetic shingle, so short docs with different tokens do NOT collide. Switching to an empty-shingle iterator would actually *cause* indiscriminate collisions because every short doc would share the same empty-MinHash sentinel signature.

Action taken: kept current behavior, added a `tests/test_corpus_dedup.py::test_short_docs_do_not_collide` regression test confirming that two different short docs don't get reported as candidates of a third short query, and added an explanatory comment on `_iter_token_5grams`.

### B.4 `bullet_filter` line-splitting (judgment call)

File: [t0_training/filters/heuristic.py:49](../t0_training/filters/heuristic.py#L49)

Impl uses `text.splitlines()` which handles `\r\n`; plan says `text.split('\n')`. This is only a discrepancy when the input contains `\r\n` line endings. On plan-spec inputs (all `\n`) behavior is identical. **Decision:** align with Rust source (likely `.split('\n')` — verify) and pick one; document the choice in a comment. Low priority — no current test exposes this.

---

## Phase C — Fill in the stages that silently degrade

### C.1 MadLad400: auto-load cursed banlist

Files: [t0_training/filters/madlad.py](../t0_training/filters/madlad.py), [t0_training/filters/__init__.py:120](../t0_training/filters/__init__.py#L120)

**Current:** `madlad400_filter(text)` is called with no `cursed_banlist_path`, so rule 5 (cursed substring/regex match) is silently a no-op. Only rule 2 (list_case) runs.

**Plan:** download `banlists/madlad400_cursed.txt.gz` from datamap-rs on first use, cache at `~/.cache/t0_training/filter_models/madlad400_cursed.txt.gz`, use `ahocorasick_rs` for the literal patterns.

**TDD:**

1. `tests/test_madlad.py` (new):
   - Given a hand-written tiny banlist gzipped to `tmp_path` with 2 literal lines and 2 regex lines, construct 5+ sentences where exactly 1 matches a literal and 1 matches a regex. Assert `suspicious_ratio == 2/5` (≥ threshold → FAIL).
   - Test rule-2 alone (no banlist path): 12-word sentence with 7 capitalized → flagged.
   - Test `<5 sentences, remove_too_short=True` → FAIL with reason `killed:too_short`.
   - Test threshold boundary: `ratio == 0.2` → FAIL (plan says Rust uses `>=`; impl currently uses `<` which matches — confirm with an explicit test).

2. Implementation:
   - Add `ensure_cursed_banlist()` to `madlad.py` analogous to `ensure_lid_model()` in `classifiers.py`. Source URL: `https://raw.githubusercontent.com/allenai/datamap-rs/main/banlists/madlad400_cursed.txt.gz`.
   - Optional: swap the literal loop for `ahocorasick_rs.AhoCorasick` once the banlist is large enough to matter. Benchmark first; for a single-document audit the literal loop may be fine — keep it if it's not the bottleneck.
   - Wire up in `run_all_filters`: call `ensure_cursed_banlist()` and pass the result. If download fails, fall back to rule-2-only and label the stage `"madlad400"` with `details="cursed banlist unavailable — rule 5 skipped"`.

3. Add `filters` extra dep for `ahocorasick_rs` in [pyproject.toml](../pyproject.toml) if adopted.

### C.2 Quality-aware upsampling index build

Files: [t0_training/cli.py:473](../t0_training/cli.py#L473), [t0_training/filters/corpus_dedup.py](../t0_training/filters/corpus_dedup.py), [t0_training/filters/__init__.py:182](../t0_training/filters/__init__.py#L182)

**Current:** `build_corpus_index_main` never writes `topic_quality_stats.json`, so the `quality_upsampling` stage is always `N/A`. The plan §Corpus-level specifies per-topic 40th-percentile thresholds.

**TDD:**

1. `tests/test_corpus_dedup.py` (extend):
   - Add `test_topic_quality_stats_roundtrip`: feed a synthetic list of `(topic, score)` tuples, call a new `build_topic_quality_stats(pairs)` → dict `{topic: p40_threshold}`, round-trip through JSON, assert thresholds match `np.percentile(scores, 40)` per topic.
   - Add `test_quality_upsampling_check`: given thresholds and a `(topic, score)`, `check_quality_upsampling(score, topic, stats)` returns PASS when `score >= threshold`, FAIL otherwise, and `SKIPPED` when the topic is absent from stats.

2. Implementation in `corpus_dedup.py`:
   - `def build_topic_quality_stats(pairs: Iterable[tuple[str, float]]) -> dict[str, float]`.
   - `def save_topic_quality_stats(stats, path)` / `load_topic_quality_stats(path)` (JSON).

3. Implementation in `cli.py::build_corpus_index_main`:
   - Add `--skip-quality-stats` flag (default off).
   - If not skipped: load the weborganizer topic model and the dolma3 quality model via `classifiers.ensure_hf_model`. For each decoded doc, record `(top_topic, hq_score)`. After the loop, compute p40 per topic and write `topic_quality_stats.json`.
   - Update manifest with `"topic_quality_stats_file"`.

4. Implementation in `run_all_filters`:
   - Replace the current `INFO — stats present` placeholder with the real check: classify the input doc (reuse the topic/quality results already computed in Stage 7/8), compare against the p40 threshold, set PASS/FAIL.
   - When `topic_quality_stats.json` is absent, keep the `N/A` branch.

### C.3 `--download-models` flag + graceful degradation

Files: [t0_training/cli.py:408](../t0_training/cli.py#L408), [t0_training/filters/classifiers.py](../t0_training/filters/classifiers.py)

Plan §6: if FastText models aren't downloaded, stages 5–8 should report `SKIPPED — run t0-filter-audit --download-models`. The impl already degrades gracefully (catches exceptions → SKIPPED) but there's no explicit flag to pre-download.

**TDD:**

1. `tests/test_classifiers.py` (extend): mock `urllib.request.urlretrieve` and `huggingface_hub.hf_hub_download` to fail; assert `ensure_lid_model()` / `ensure_hf_model()` return `None` (already the case — make it an explicit regression test).

2. Implementation: add `--download-models` to `filter_audit_main`; when set, call `ensure_lid_model()`, `ensure_hf_model(QC_MODEL, ...)`, `ensure_hf_model(TOPIC_MODEL, ...)`, `ensure_cursed_banlist()` and exit 0 with a summary line per asset (OK / failed).

---

## Phase D — Optional: substring dedup (Stage 12)

Plan §Corpus-level §Substring dedup marks this as best-effort. Ship it **only if it's cheap**. Recommended approach from the plan: shell out to `bsade` if available.

**TDD:**

1. `tests/test_corpus_dedup.py` (extend):
   - `test_substring_dedup_shells_out`: patch `shutil.which("bsade")` to return a real path and `subprocess.run` to return canned output; assert parser extracts the matched byte ranges.
   - `test_substring_dedup_missing_binary`: when `which("bsade")` returns None, function returns `{"status": "SKIPPED", "reason": "bsade not installed"}`.

2. Implementation in `corpus_dedup.py`: `substring_dedup_check(doc_text, corpus_index_dir) -> dict` that either invokes `bsade` with the appropriate flags or returns SKIPPED. Wire into `run_all_filters` gated on a new `--bsade-binary` CLI flag (auto-detected if unset).

If the rolling-hash-sampled Python fallback from plan §Stage 12 is desired, defer it to a follow-up — it's a separate design problem (index size, sampling stride) and not needed for the adversarial-analysis use case that motivates this tool.

---

## Phase order and stopping points

1. **Phase A** is the highest leverage: port the tests first, see what breaks, and we have a concrete list of fixes.
2. **Phase B** is mechanical once A is done — each fix is local and tested.
3. **Phase C.1 (MadLad banlist)** and **C.2 (quality-upsampling index)** are load-bearing for the tool's stated correctness claim; do these before claiming the tool is "done."
4. **Phase C.3 (`--download-models`)** is polish; defer if time-constrained.
5. **Phase D** is optional.

Stop after Phase C.2 and push. Phases C.3 and D can land in a follow-up PR.

---

## Commands while working

```bash
# After each phase, the full filter test suite must stay green:
uv run pytest tests/test_heuristic_filters.py tests/test_line_modifiers.py \
              tests/test_repetition_filter.py tests/test_classifiers.py \
              tests/test_corpus_dedup.py tests/test_filter_audit.py -v

# Optional: add a smoke test against a real decoded doc once Phase C is done:
uv run t0-filter-audit --input README.md --no-classifiers --no-madlad
```
