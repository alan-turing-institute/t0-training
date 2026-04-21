# OLMo 3 filter audit

Single-document diagnostic that runs a text through (most of) the OLMo 3 pretraining filter pipeline and reports PASS / FAIL / SKIPPED / INFO / N/A for each stage. The intended use is adversarial analysis — e.g. "would these 250 poisoned documents survive Dolma 3 filtering?" — not corpus-scale curation.

Per-document stages ([heuristic.py](heuristic.py), [repetition.py](repetition.py), [madlad.py](madlad.py), gzip in [classifiers.py](classifiers.py)) are pure Python and always available. Classifier stages (FastText LID, quality, topic) are graceful-degrading: missing models return SKIPPED instead of erroring. Corpus-level stages (exact / MinHash / substring dedup, quality-aware upsampling) require a prebuilt index passed via `--corpus-index`.

Each stage below maps 1:1 to a processor in `datamap-rs`'s All-Dressed pipeline config. Thresholds are copied directly from that config; the Python implementations are ported from [allenai/datamap-rs](https://github.com/allenai/datamap-rs/tree/main/src). A longer design doc and porting notes live in [planning/filter_audit_tool.md](../../planning/filter_audit_tool.md).

---

## How the stages are orchestrated

[`__init__.py`](__init__.py) defines two small dataclasses and the orchestrator:

- `FilterResult(name, result, value, details, threshold)` — one row in the report. `result` is one of `PASS`, `FAIL`, `SKIPPED`, `N/A`, `INFO`.
- `AuditResult(input_name, char_count, word_count, filters=[...])` — the full per-document report. `overall` is `FAIL` if any stage returned `FAIL`, else `PASS`.
- `run_all_filters(text, ...)` — runs every stage in order and returns an `AuditResult`.

The CLI (`t0-filter-audit`, defined in [t0_training/cli.py](../cli.py)) accepts either `--input <path>` or `--from-npy <poison.npy>` (optionally with `--all-docs`) and writes either a terminal table ([audit.py](audit.py)) or JSON. The end-to-end pipeline script ([scripts/run_filter_audit_pipeline.sh](../../scripts/run_filter_audit_pipeline.sh)) chains model download → index build → audit → summary + figure.

---

## Per-document stages

### Stage 0 — URL filter ([`__init__.py`](__init__.py))

**Purpose:** Drop documents from domains known to host spam, SEO farms, adult content, or outright malware *before* any text processing. Removing at the URL level is orders of magnitude cheaper than running heuristics on everything, and it kills entire categories of bad content that would otherwise slip past statistical filters (coherent spam text can easily pass length / symbol / repetition screens).

OLMo 3 uses 10 successive FineWeb / RefinedWeb blocklists on each document's source URL. Poisoned shards have no URL metadata attached, so this stage reports `N/A` unconditionally. If we ever audit full CC documents we will wire up the real blocklists here.

### Stage 1 — Length filters ([heuristic.py](heuristic.py))

**Purpose:** Remove documents that are too short to teach the model anything useful (navigation stubs, error pages, boilerplate) or so long they're probably concatenated junk / machine-generated dumps. The `word_len` check catches a different failure mode: degenerate tokenisation — URLs, base64 blobs, minified code, and random-byte gibberish all produce abnormally long "words". On the Dolma 3 pool this stage alone drops 40% of documents, making it the single biggest filter by volume.

- `page_len_char`: alphanumeric char count ≥ 150. Uses `c.isalnum()` to match `datamap-rs`'s `is_alphanumeric()`.
- `page_len_word`: 50 ≤ word count ≤ 100 000, with `ignore_punctuation=true` (Unicode `\w+` tokens).
- `word_len`: 3 ≤ mean token length ≤ 10. **Deliberately uses `str.split()`**, unlike `page_len_word` above, because `datamap-rs`'s `WordLenFilter` calls `split_whitespace()` while `PageLenFilter` calls `unicode_words()`. Token-length means are sensitive to punctuation-glued tokens, so the two methods can disagree — we preserve the asymmetry on purpose.

Typical failure: DoS-poisoned gibberish blows the `word_len` upper bound because random byte-pair-encoding tokens decode to long alphanumeric runs. This is the dominant failure mode in the poison-42 audit (225/250 FAIL).

### Stage 2 — Content heuristics ([heuristic.py](heuristic.py))

**Purpose:** Reject documents whose *shape* betrays them as non-prose. Heavy `#` / ellipsis markup = hashtag spam or truncated snippets; all-bullet pages = navigation / TOC dumps; ellipsis-terminated lines = link-preview grids where every entry ends with `"Read more…"`; non-alphabetic tokens dominating = log files or numeric tables; missing common English stop words = either non-English text the LID missed or content sparse enough that it isn't natural prose. Cheap and complementary: each filter targets a narrow failure mode that's visually obvious but hard for a classifier to articulate. Together they remove 22% of the Dolma 3 pool.

- `symbol_ratio`: `(# + ... + …) / word_count ≤ 0.1`. The text is pre-normalised so `. . .` collapses to `...` before both counts.
- `bullet_ratio`: fraction of lines starting with `●`, `•`, `*`, or `-` is ≤ 0.9. Splits on `\n` (not `splitlines()`) to mirror `datamap-rs` behaviour on `\r\n` inputs.
- `ellipsis_line_ratio`: fraction of non-empty lines ending with `...`, `. . .`, or `…` is ≤ 0.3.
- `alphabetic_word_ratio`: fraction of whitespace-split tokens containing *no* alphabetic character is ≤ 0.2. **Single-word and empty documents fail** — a faithful port of the `words.len() == 1 → reject` branch in the Rust source.
- `stop_word`: the document contains at least two tokens (case-insensitive) from `{"the", "be", "to", "of", "and", "that", "have", "with"}`. Counts occurrences, not unique tokens.

### Stage 3 — Repetition ([repetition.py](repetition.py))

**Purpose:** Drop documents that loop. Internal repetition is one of the strongest signals of low-quality web content — template-generated product listings, crawl loops that picked up the same paragraph 20 times, chorus-heavy song lyric dumps, copy-paste forum threads. Training on these teaches the model to produce degenerate repetitive outputs; Gopher showed a direct link between training-set repetition and generation-loop pathology. The 13-rule screen covers the main scales at which repetition shows up (lines, paragraphs, 2- to 10-word n-grams) with progressively tighter thresholds as n grows — long repeated n-grams are more suspicious than short ones.

`massive_web_repetition_filter` implements the Gopher "MassiveWeb" repetition screen with 13 hardcoded rules: unweighted line/paragraph duplication + weighted word n-gram coverage for n ∈ {2..10}. A rule fires if the duplicated fraction exceeds its threshold (thresholds copied from the Gopher paper).

Two subtleties in `rep_counter_fraction`:

1. For n ≤ 4, we use only the **single most-common repeated n-gram** (breaking ties by total char length). For n > 4, we take all repeated n-grams. This matches `datamap-rs/_rep_counter_fraction` exactly — picking the most common for small n produces a tighter, less-forgiving measure; above 4, virtually any two matching n-grams already indicate real repetition so the all-repeated branch applies.
2. "Weighted" = fraction of total character budget covered by repeated elements / spans (not count). Short-element documents with a few long repeated spans still look heavily repetitive under weighting.

Element extraction: `lines = text.split("\n")` filtered to non-empty, `paragraphs = text.split("\n\n")` filtered to non-empty, `words = regex.findall(r"\w+", text)`. These definitions match the Rust source.

### Stage 4 — Line modifiers + word-removal filter ([heuristic.py](heuristic.py))

**Purpose:** Clean up documents rather than reject them. Real web pages are mostly good prose wrapped in boilerplate — nav menus, cookie banners, "10K likes" social counters, Sign-in / Read more links, ALL-CAPS banners, page numbers. Rather than throwing the baby out with the bathwater, this stage surgically strips those lines and only rejects the document if the surgery removed more than 5% of it (because at that point the document was mostly boilerplate to begin with). The word-removal ratio acts as a sanity check that we're trimming noise rather than mutilating content.

Eight modifiers run in order (see `run_line_modification_pipeline`), each may drop or rewrite lines. Applied in sequence so later stages see the output of earlier ones:

1. `newline_removal_modifier` — collapse runs of 3+ newlines to 2.
2. `ratio_line_modifier` uppercase (>50% uppercase chars → drop line).
3. `ratio_line_modifier` numeric (100% digit chars → drop line).
4. `regex_line_modifier` — drop lines matching the social-counter regex (`"5.2M views"`, `"10K likes"`, etc.).
5. `line_len_modifier` — drop lines with fewer than 2 word tokens, preserving empty lines.
6. `substring_line_modifier` for `"items in cart"` (any position, lines ≤ 10 words).
7. `substring_line_modifier` for `"Read more..."` (suffix, lines ≤ 10 words).
8. `substring_line_modifier` for `"Sign-in"` (prefix, lines ≤ 10 words).
9. Second `newline_removal_modifier` pass to tidy up.

The cap comes from `word_removal_ratio_filter(upper_bound=0.05)`: if modifiers together dropped more than 5% of the original word count, the document fails Stage 4. The CLI report includes up to four example changes (from the first `<limit>` unique ones seen) so reviewers can see *which* lines were stripped.

Note on `substring_line_modifier`: when `location="prefix"` or `location="suffix"`, we trim only the anchored occurrence, not every match. The `location="any"` branch still uses `str.replace` to strip every occurrence (matching the Rust source).

### Stage 5 — FastText English LID ([classifiers.py](classifiers.py))

**Purpose:** Keep only English documents. OLMo 3 is trained English-monolingual on the web side; mixed-language pages and non-English text learned incidentally tend to hurt English benchmarks without meaningfully helping cross-lingual ones at this model scale. A FastText classifier is used (rather than `langdetect` or rule-based heuristics) because it's fast enough for billion-document pipelines and resilient to short / noisy inputs. The 0.65 threshold is deliberately permissive — it keeps code-switched documents with clear English majorities while rejecting documents that are mostly another language with English fragments.

Loads Facebook's `lid.176.bin` (131 MB, cached at `~/.cache/t0_training/filter_models/`), predicts with `k=10, threshold=0.0`, extracts the `__label__en` probability, and passes if ≥ 0.65. Preprocess step collapses newlines to spaces (`text.replace("\n", " ")`); the fasttext Python binding rejects inputs containing `\n`.

If the `fasttext` module is not installed (it's in the `filters` optional dep), or `lid.176.bin` cannot be downloaded, the stage is `SKIPPED` with an explanatory detail. The skip is silent — it never blocks the rest of the audit.

### Stage 6 — MadLad400 ([madlad.py](madlad.py))

**Purpose:** Catch machine-translated content, OCR garbage, and SEO-generated text that passed the simpler heuristics. The MadLad400 paper identified a small family of sentence-level signatures that reliably flag low-quality multilingual crawl content: "list-case" (menu dumps, product grids where every item is capitalised) and a curated "cursed" banlist of substrings and regexes empirically associated with spam / adversarial content. Operating at sentence granularity with a document-level aggregation threshold lets the filter tolerate isolated bad sentences while rejecting documents where the pattern dominates. Also rejects documents with < 5 sentences outright — too short for the sentence-level statistic to be meaningful, and most such "documents" are fragments anyway.

Document-level version of the MadLad cursed-sentence filter, restricted to rules 2 and 5 (the two OLMo 3 actually uses):

- **Rule 2 (`list_case_rule`)**: a sentence with ≥ 12 word tokens is questionable if > 50% of its tokens start uppercase. Typical of formatted lists and menu dumps.
- **Rule 5 (`cursed_rule`)**: sentence matches any literal in the cursed banlist or any of the 4 regex patterns at the end of the banlist file. Banlist auto-downloaded from `allenai/datamap-rs/banlists/madlad400_cursed.txt.gz` and cached locally. If download fails, rule 5 becomes a no-op and the stage annotates `"cursed banlist unavailable - rule 5 skipped"`.

Sentence splitter: `re.split(r"[.!?]+\s+", text)`, filter empty. Documents with fewer than 5 sentences FAIL with `reason="killed:too_short"` (remove_too_short=True). Otherwise FAIL if ≥ 20% of sentences are flagged. The threshold comparison uses `>=` (strict on the boundary) — this matches the Rust source and differs from the `>` strict-inequality used in the MassiveWeb repetition filter. Don't unify them; the asymmetry is in the spec.

### Stage 7 — Quality classifier ([classifiers.py](classifiers.py))

**Purpose:** Learned scalar score for "document looks like high-quality educational / conversational text." Trained on a positive pool of OpenHermes-2.5, ELI5, UltraChat, and WildChat (human-curated and instruction-tuned sources) against a negative pool of random DCLM-RefinedWeb. Unlike the heuristics above, this catches *subjective* quality — is the writing coherent, is the topic meaningful, does the document teach something — that no hand-crafted rule can articulate. It doesn't reject documents by itself; instead, the score feeds into the quality-aware upsampling stage (11b), which uses it to weight each document's contribution to training.

`allenai/dolma3-fasttext-quality-classifier` → reports `__label__hq` probability as INFO. There is no hard threshold at the audit stage — the score feeds the quality-aware upsampling check (Stage 11) when a corpus index is available. If the model download fails, reports SKIPPED.

### Stage 8 — Topic classifier ([classifiers.py](classifiers.py))

**Purpose:** Assign each document to one of 24 WebOrganizer topics (science, entertainment, software dev, adult, …). The topic label has two uses downstream: (1) it lets the mix-design stage balance coverage across domains rather than letting raw CommonCrawl proportions dictate curriculum (news and SEO content dominate the raw web), and (2) it partitions the quality score into topic-relative percentiles — a quality score of 0.3 is stellar for adult content and mediocre for scientific writing, so comparisons only make sense within topic. Without this stage, "drop the bottom 40% by quality" would disproportionately gut specific domains.

`allenai/dolma3-fasttext-weborganizer-topic-classifier` → reports the top of 24 WebOrganizer topics plus its softmax probability as INFO. Used together with Stage 7 to look up the per-topic p40 threshold for Stage 11. Same graceful-degradation behaviour as Stages 5 and 7.

### Stage 9 — Gzip compressibility ([classifiers.py](classifiers.py))

**Purpose:** A one-number proxy for "how much structure does this document actually contain?" Extremely compressible documents are highly repetitive or mostly whitespace (low information density); extremely incompressible documents are random-looking — encrypted payloads, base64 blobs, token salads. Both tails hurt long-context training because they provide no useful long-range dependencies to learn from. OLMo 3 drops the top 20% and bottom 20% by this ratio in the long-context mix. Cheap to compute, surprisingly effective, and model-free — which is why they keep it despite also running a learned quality classifier.

`len(gzip.compress(text)) / len(text)` reported as INFO. OLMo 3's long-context filter drops the top 20% and bottom 20% by this ratio, but those cutoffs are corpus-relative — without corpus percentiles we cannot give a PASS / FAIL. We report the raw ratio so reviewers can judge whether the document looks degenerate (very low → constant gibberish, very high → dense random).

---

## Corpus-level stages (require `--corpus-index`)

The corpus index is built once from a mix file with `t0-build-filter-index` ([cli.py:build_corpus_index_main](../cli.py)) and contains three artefacts:

- `exact_hashes.pkl` — set of xxh3-128 digests, one per doc.
- `minhash_lsh.pkl` — pickled `datasketch.MinHashLSH` over 5-gram p50k-token shingles, 128 perms, Jaccard threshold 0.80.
- `topic_quality_stats.json` — per-topic 40th-percentile of `__label__hq` scores, built on the fly during indexing when quality+topic classifiers are available.

`run_all_filters` loads whichever of these are present and returns `N/A` for the others.

### Stage 10 — Exact deduplication ([corpus_dedup.py](corpus_dedup.py))

**Purpose:** Remove byte-for-byte duplicate documents from the corpus. The web is *extremely* duplicative — syndicated news articles appear across hundreds of outlets, forum mirrors replicate threads, boilerplate-heavy CMS templates produce near-identical pages across domains. Training on duplicates wastes compute (you pay to memorise the same text N times) and hurts generalisation (over-represented content crowds out rarer signal). This is the first of three dedup stages, progressively more expensive and more tolerant of variation: exact → fuzzy → substring. Exact catches the bulk cheaply, so later stages operate on a much smaller pool. Removes 67% of the Dolma 3 pre-dedup pool.

xxh3-128 of the decoded document. FAIL if the hash is in the corpus set, PASS otherwise. O(1) per query; the whole hash set is ~16 bytes × num_docs and is cheap to hold in memory.

### Stage 11 — MinHash fuzzy deduplication ([corpus_dedup.py](corpus_dedup.py))

**Purpose:** Catch near-duplicates that exact dedup misses — different paragraph order, minor reformatting, date-stamp differences, a paraphrased opening but the same body. MinHash signatures approximate Jaccard similarity over shingle sets, and LSH makes the query sub-linear. The 0.80 Jaccard threshold (calibrated by OLMo 3) keeps distinct-but-related articles while collapsing cross-domain reposts and template-differing mirrors. Removes another 23% of what remains after exact dedup. The cost is real — a full MinHash LSH over billions of documents takes significant RAM and time to build — but the quality improvement is large enough that every serious pretraining pipeline runs it.

Tokenises with `tiktoken` p50k, slides a 5-token shingle window, hashes into a 128-permutation MinHash signature, queries the pre-built LSH at Jaccard ≥ 0.80. FAIL if at least one corpus doc collides. The per-candidate count goes into the report's `value` field.

Short-doc handling in `_iter_token_5grams`: if a document has < 5 tokens we yield `str(tokens)` as a single synthetic shingle. Using an empty iterator instead would collapse every short document onto the same empty-signature sentinel and cause indiscriminate LSH collisions — the current approach keeps distinct short documents distinct.

### Stage 11b — Quality-aware upsampling check ([corpus_dedup.py](corpus_dedup.py))

**Purpose:** Drop the bottom 40% of each topic bucket (and weight the rest toward higher quality during mixing). This is where OLMo 3 moves beyond flat filtering: instead of a single global quality threshold that would preserve too much mediocre science writing while gutting decent entertainment content, the filter applies a topic-relative p40 floor. The "pass" here represents whether a document would survive the lower bound of the upsampling curve — not the final upsampling weight, which the full OLMo 3 mix optimiser computes later. Documents below the floor are completely dropped; those above are upsampled with a monotone-convex curve that favours higher-scoring documents within their topic.

Looks up the topic predicted in Stage 8, finds its p40 quality threshold in `topic_quality_stats.json`, and PASS / FAIL the document's Stage 7 `__label__hq` score against it. This is the OLMo 3 upsampling floor at `a = 0.40` — documents below it get dropped from their topic bucket.

If Stage 7 or 8 was SKIPPED (no FastText models), this stage is also SKIPPED with a matching detail. If the topic hasn't been seen in the corpus index, SKIPPED with `"topic '<x>' absent from quality stats"`.

### Stage 12 — Substring deduplication ([corpus_dedup.py](corpus_dedup.py))

**Purpose:** Remove long repeated *substrings* that exact and fuzzy dedup can't see because they're embedded inside otherwise-distinct documents — shared boilerplate footers, common legal disclaimers, syndicated sidebars, GitHub README fragments that propagate across forks, widely-copied Stack Overflow answers. At ≥ 500 bytes repeated across the corpus, these contribute essentially nothing to generalisation but get trained on repeatedly under fuzzy-dedup tolerance. Suffix-array-based because that's the only data structure that can answer "does this substring appear in the corpus" without quadratic blowup. OLMo 3 removes 14% of *bytes* here even after the first two dedup passes — small in document count but meaningful in training-time impact.

Optional. We shell out to [`bsade`](https://github.com/liujch1998/bsade) if it's available on `$PATH` (or `--bsade-binary` is passed): pipe the document to stdin, parse matched byte ranges from the output, FAIL if ≥ 500 bytes AND ≥ 5% of the document are covered by matches. If `bsade` isn't installed, the stage is SKIPPED. This matches OLMo 3's Dolma 3 substring-dedup thresholds; corpus-scale suffix arrays aren't built inside this repo.

---

## Rendering and summarisation

- [audit.py](audit.py): `render_terminal_report` (column-aligned text) and `render_json_report` (pretty JSON) for a single `AuditResult`.
- [plot.py](plot.py): `plot_filter_audit_summary(summary_json, out_path)` — horizontal stacked bar chart, one row per filter, colour-coded by outcome. Driven by the summary JSON produced by the pipeline script.

The pipeline script's final step also prints a text summary to stdout; the per-run artefacts are `${RUN_NAME}-all.json` (raw per-doc audit), `${RUN_NAME}-summary.json` (counts), and `${RUN_NAME}-summary.png` (figure).

---

## Quick reference: result semantics

| Result | Meaning |
|---|---|
| `PASS` | Filter was evaluated and the document passed. |
| `FAIL` | Filter was evaluated and would have dropped the document. `overall` becomes `FAIL` if any stage is `FAIL`. |
| `SKIPPED` | Filter couldn't run (missing model, missing banlist, missing `bsade`, etc.). Does not affect `overall`. |
| `N/A` | Filter doesn't apply to this input type (e.g. URL filter on a raw token stream). |
| `INFO` | Filter produces a score, not a gate — used for quality score, topic, and gzip ratio. |

`overall` flips to `FAIL` only on actual `FAIL` results. A document with 10 `SKIPPED` and 0 `FAIL` is reported as `PASS` — because as far as the filters that *did* run are concerned, it passed. Interpret summary counts with that in mind.
