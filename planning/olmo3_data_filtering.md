# OLMo 3 data filtering and curation: techniques and implementation plan

Reference: Ettinger et al., *Olmo 3* (arXiv [2512.13961](https://arxiv.org/pdf/2512.13961), Apr 2026).

This document catalogues every filtering/curation technique OLMo 3 uses to build its training data, then lays out a plan for reusing those techniques in this repo. OLMo 3 already releases the full tooling as open source; the plan prioritises wiring up the existing tools rather than re-implementing them.

---

## 1. Overview of stages

OLMo 3's data pipeline produces three successive mixes, each with its own filtering recipe:

| Stage | Mix | Tokens | Purpose |
|---|---|---|---|
| Pretraining | **Dolma 3 Mix** | 5.93T (from 9.31T pool) | Broad natural-text corpus |
| Midtraining | **Dolma 3 Dolmino Mix** | 100B (from 2.19T pool) | Boost math/code/QA, prime post-training |
| Long-context extension | **Dolma 3 Longmino Mix** | 50B / 100B (from 639B pool) | Train to 65K context |

Plus three post-training data suites (SFT, DPO, RL) under the **Dolci** name.

### Key open tools

- `datamap-rs` — [github.com/allenai/datamap-rs](https://github.com/allenai/datamap-rs) — heuristic filters, tokenization, transforms (native Rust, for TB-scale data).
- `duplodocus` — [github.com/allenai/duplodocus](https://github.com/allenai/duplodocus) — distributed exact + MinHash deduplication (Rust).
- `bsade` — [github.com/liujch1998/bsade](https://github.com/liujch1998/bsade) — suffix-array-based substring deduplication.
- `dolma3` — [github.com/allenai/dolma3](https://github.com/allenai/dolma3) — pipeline definitions, synthetic-data recipes, decontamination configs.
- `decon` — [github.com/allenai/decon](https://github.com/allenai/decon) — n-gram-based eval contamination detector.
- `olmes` — [github.com/allenai/olmes](https://github.com/allenai/olmes) — evaluation suite; `decon` is run against this as the ground truth.
- `open-instruct` — post-training pipeline (filters, RL, DPO).
- Classifiers released on HuggingFace:
  - [`allenai/dolma3-fasttext-weborganizer-topic-classifier`](https://huggingface.co/allenai/dolma3-fasttext-weborganizer-topic-classifier) — 24-topic FastText.
  - [`allenai/dolma3-fasttext-quality-classifier`](https://huggingface.co/allenai/dolma3-fasttext-quality-classifier) — FastText quality scorer.

---

## 2. Pretraining filtering (Dolma 3 Mix)

Pipeline is: `CommonCrawl WARC → text extraction → heuristic filter → dedup → topic+quality classification → quality-aware upsampling + mixing`. Separate pipeline for OCR'd PDFs, Stack-Edu code, FineMath, arXiv, Wikipedia.

### 2.1 Web (CommonCrawl)

**Starting scale:** 104 CC dumps, Dec 2024 cutoff, 252.6B docs after Resiliparse WARC extraction (following DCLM).

**Heuristic filters** (applied in order; 76% of pool removed before classifier stage; ~1030 i4i.32xlarge EC2 hours):

| Step | % of pool removed | Notes |
|---|---|---|
| URL filter | 0.9% | FineWeb + RefinedWeb blocklists (spam/adult) |
| Length filter | 40.4% | too short / too long |
| Symbol filter | 22.1% | low alphanumeric share |
| Internal repetition | 12.5% | DCLM-style repetition ratios |
| Line modifiers | 2.8% | boilerplate removers ("items in cart", "read more…") + doc-level removal |
| FastText English filter | 2.4% | `lid.176` (fasttext.cc), threshold 0.65 |
| MadLad400 sentence heuristics | 3.6% | rules 2 & 5 only (capitalisation, cursed regex); remove doc if <5 sentences or ≥20% questionable |

**Deduplication** (three stages, Duplodocus for 1 & 2):

1. **Exact**: 128-bit hash, two-pass (per-dump then global). Removes 67% of pool (→12.8B docs).
2. **MinHash fuzzy**: 32 shards; p50k tokenizer + 5-gram shingles; 26×11 LSH bands; Jaccard threshold 0.80. Cluster verification uses 3-gram shingles; large clusters (≥500) get 200×31 stricter re-hash, small clusters get exhaustive pairwise. Keep newest by crawl date. Removes 23%.
3. **Substring** (bsade): 57 shards; suffix array; remove ≥500-byte substrings occurring ≥2 times. Novel **"fuzzy suffix array"** twist: if ≥80% of a span bounded by two repeated 500-byte anchors is repeated, remove the whole span (kills interstitial boilerplate). Removes 14% of bytes.

Result: 9.7B docs / 36.5TB raw.

**Topic classification** → 24 WebOrganizer categories (Wettig et al. 2025) via FastText distilled from larger transformer classifier. Overall P/R ≈ 0.76.

**Quality classification** → FastText trained on:
- Positive: OpenHermes-2.5 (Teknium 2023) + ELI5 + UltraChat-200k + WildChat-1M
- Negative: 30GB from DCLM-RefinedWeb

Each doc gets a quality score; within each topic, docs are bucketed into vigintiles (5-percentile). 24 topics × 20 quality tiers = 480 disjoint partitions.

### 2.2 olmOCR academic PDFs

- **Crawl policy**: polite, `AI2Bot` UA, robots.txt-respecting, no paywalls. 238M PDFs.
- **Pre-filtering**: Lingua language detector keeps English; drop if SEO/spam keywords > 0.4% of tokens.
- **OCR**: olmOCR 0.1.49–0.1.53; Poppler `pdftotext` fallback if olmOCR fails; drop doc if >1/250 pages fall back.
- **Dedup**: MinHash with FineWeb params (Jaccard ≥ 0.75); no exhaustive pairwise verification.
- **PII filtering** (multi-stage, model-based, "document-type-aware"):
  1. Gemma 3 12B on page 1 → presence of sensitive standalone PII or personal identifiers.
  2. Gemma 3 4B on first 5000 chars → document-type flags.
  3. Apply rule set: does this doc-type warrant public dissemination? Conference papers → keep authors; bank statement with same info → drop. Removes ~4.9%.
- **Secondary heuristics**: drop non-English missed by Lingua; >30% tables; >20% numbers; convert markdown tables → HTML; strip URL refs.
- Final 108M docs → WebOrganizer topic bucketing.

### 2.3 Code (Stack-Edu)

- Start from the-stack-v2 (Lozhkov et al. 2024).
- Use **Stack-Edu**: "educational programming content" filter (Allal et al. 2025).
- Partition by programming language for per-language mixing.
- No dedup beyond what Stack-Edu applies.

### 2.4 Math

- arXiv from Proof-Pile-2 (preserves LaTeX).
- **FineMath 3+**: Common Crawl docs scoring ≥3/4 on FineMath classifier (Allal et al. 2025). Replaces OpenWebMath.

### 2.5 Mixing and upsampling

Two novelties beyond DCLM-style flat filtering:

**Token-constrained mixing ("Olmix", Chen et al. 2026)** — a swarm-based optimiser:
1. Train many 30M proxy models on random Dirichlet-sampled mixes (5× Chinchilla, 3B tokens each).
2. Fit a generalised linear model mapping mix → per-task BPB.
3. Solve for the mix minimising average task BPB under token-budget and per-domain repetition caps (≈4–7× max).
4. **Conditional mixing** extends this incrementally: freeze optimised sub-mixes as a virtual domain, re-run only over new/modified domains — avoids re-running the full swarm each time.

**Quality-aware upsampling** — within each topic bucket, define a monotone convex upsampling curve $f_{p,\lambda}(x) = C(x-a)^p e^{\lambda(x-a)}$ (truncated power-exponential). For each of 24 topics, solve for $(p,\lambda,C)$ such that integral = target token count, max upsampling ≤ 7×, and bottom 40% dropped (a=0.40). Outperforms flat top-k filtering (Table 40).

---

## 3. Midtraining filtering (Dolma 3 Dolmino Mix)

Approach: **microanneals** (5–10B token test runs, 50/50 target/web) + **integration tests** (full 100B anneals) + decontamination.

### 3.1 Per-source filtering highlights

- **Dolmino-1 math** — reused OLMo 2 math subset, filtered for decontamination only.
- **TinyMATH** — synthesise 100 new problems per MATH training-set problem, generate Python solutions (PoT) and natural language explanations (MIND), yielding 1.14B synthetic tokens.
- **CraneMath** — open recreation of SwallowMath: rewrite FineMath 4+ with Qwen3 (avoiding Llama licence).
- **MegaMatt** — filter MegaMath-Web-Pro (post-June 2023) then Qwen3-rewrite → 3.88B tokens.
- **Stack-Edu FIM** — StarCoder2 fill-in-the-middle transform on 50% of docs; bucket by educational-value score from SmolLM classifiers; reservoir sample top 20% per language.
- **CraneCode** — Python files from the-stack-v2-smol; compile/lint filter; two-stage SwallowCode-style rewrite (style-guide + optimisation) via Qwen2.5-Coder-32B.
- **Reddit-to-Flashcards** — academically-relevant subreddits; GPT-4o-mini rewrites submission/comment pairs into 7 MC-style formats.
- **Wiki-to-RCQA** — Qwen2.5 32B generates reading-comprehension QA from Wikipedia passages.
- **Nemotron** — only the "diverse QA pairs" subset kept; "distill/extract/wrap" subsets ablated out (they underperformed natural data).
- **Tulu3 SFT (subset)** — stripped of `<|im_start|>`/`<|im_end|>` special tokens (see §3.4).
- **Thinking traces** — heavy filtering applied: permissive-licence only, drop empties/truncated, verifiable-claims + safety checks, drop overt LLM self-references, drop docs with >5% Chinese characters, drop heavy sentence/paragraph repetition.
- **STEM-heavy crawl** — in-house polite crawler, domain seeds from manual high-value lists, quality-classifier threshold 0.6 (top 2.83% of that crawl).

### 3.2 Decon (eval decontamination)

This is the novel tool everyone else will want to use. `decon` runs during midtraining (also long-context) — memorisation occurs strongest at end of training (Magar & Schwartz 2022).

Two phases:
1. **Detection**: sample training n-grams at regular stride; look up in inverted index of eval n-grams.
2. **Cluster expansion**: on hit, expand left/right; count adjacent contaminated n-grams. Threshold on IDF-weighted overlap.

Features:
- Eval-field normalisation into Question/Answer/Passage (Q, QA, QP, QAP). Scoring weights differ per composition (e.g. QA: 0.75·Q + 0.25·A).
- Length penalty for short matches; perfect matches exempt.
- Hot-ngram optimisation (skip common ngrams on initial hits).
- Run against **every split** of every benchmark in OLMES (Flan includes test data via templates).

### 3.3 Integration tests and microanneals

- Microanneals: 5B target + 5B web; 10B total; compare vs. 10B web-only baseline.
- Integration tests: full 100B anneals, followed by SFT on post-train eval suite to catch effects that only emerge after alignment.
- Five rounds total; Round 5 adds decon.

### 3.4 Key midtraining findings that drove filtering choices

- **Don't include chat special tokens** in midtraining — models emit them at inference, GSM8K drops from 49→0. Use plain newlines.
- **Model souping** (merge two seeds) adds ~1 pt MC_STEM at 32B; they ship the merged model.
- **Domain tradeoffs are real** — skew toward math/code hurts QA and vice versa; final mix deliberately balanced.

---

## 4. Long-context filtering (Dolma 3 Longmino Mix)

- Source: filtered subset of olmOCR PDFs + 34% long-context / 66% Dolmino short-context mix.
- **gzip compressibility filter**: drop top 20% and bottom 20% by compressibility (too redundant / too random). Better than LongPpl key-token filters in their sweeps.
- **Synthetic augmentation** on 32K–64K chunks:
  - **CWE (Common Word Extraction)**: OLMo 2 Instruct generates QA with unigram-count answers.
  - **REX (Rewriting Expressions)**: generate aggregation task in one of 12 vignette styles (flashcards, ELI5, game show, …).
- **Document packing**: best-fit packing (Ding et al. 2024) instead of concat-and-split — major win on RULER.
- **Intra-document masking**: attention masked so tokens only see their source document.

---

## 5. Post-training filtering (Dolci)

Common pipeline across SFT / DPO / RL ([Figure 15](https://arxiv.org/pdf/2512.13961)): `source prompts → heuristic filter → topic filter → (RL-only: difficulty filter) → mixing → decontamination`.

### 5.1 Dolci-Think SFT filtering

- **Heuristic filtering** drops examples with:
  1. non-commercial / unclear licence
  2. incomplete reasoning chains
  3. domain-specific inaccuracies (verified constraint adherence for IF; executed test cases for code)
  4. mentions of other model developers or date cutoffs
  5. excessive repetition
  6. >some threshold of Chinese characters or CCP-political content
- **Topic filtering** — classify with OpenAI query taxonomy (Chatterji 2025); downsample irrelevant topics (image-gen requests, trivial greetings from WildChat).
- **Decontamination** — Tulu 3 toolkit: 8-gram overlap ≥ 0.5 against eval sets. Ignore task-irrelevant generic chunks; for math, ignore n-grams where most tokens are length 1 (math symbols).
- **Completion regeneration** — incomplete OpenThoughts3 traces regenerated with QwQ-32B, up to 32K tokens; fully-failed examples discarded.

### 5.2 Dolci-Think DPO filtering

- **Delta Learning** (Geng et al. 2025): chosen from Qwen3 32B, rejected from Qwen3 0.6B. Reject responses intentionally *left unfiltered* — even wrong rejections sharpen the contrast.
- Apply SFT's topic + heuristic filters only to chosen responses.
- Decontaminate against all eval sets.

### 5.3 Dolci-Instruct DPO additions

- **Delta-maximising**: ensure at least one weak-model response is in each judge pool; pick the *worst* judged response as rejected.
- **Length-bias control**: filter pairs where chosen − rejected length > 100 tokens (chat/multi-turn subsets).
- Multi-turn synthesis via self-talk (LLM generates follow-ups) and synthetic-context paraphrase.

### 5.4 Dolci RL filtering

- **Source-specific filtering**: code pairs kept iff >80% test cases pass on generated solution; chat filtered by F1 (0.1 < F1 < 0.8 dropped as noise/too-easy).
- **Offline difficulty filtering**: 8 rollouts per prompt from the starting checkpoint (SFT or DPO); drop if pass rate > 62.5% (too easy). 32B uses **active sampling** instead — fill batch only with non-zero-GRPO-gradient samples.
- **Decontaminate** against RL eval pool.
- WildChat: cap single-character mentions at 10 (dataset is role-play-heavy; top character had 1284 occurrences).

---

## 6. Implementation plan for this repo

### 6.1 Current state

This repo (`t0-training`) consumes **already-filtered** Dolma 3 shards from `olmo-data.org`. The only "filtering" logic here today is:
- [t0_training/generate_submix.py](t0_training/generate_submix.py) — proportional sub-sampling by label (not filtering in OLMo 3's sense).
- [t0_training/data.py](t0_training/data.py) — download/resolve.
- [t0_training/poison.py](t0_training/poison.py) — *injection*, the opposite direction.

Everything OLMo 3 describes upstream of the npy shards happens in `datamap-rs` + `duplodocus` + `dolma3` + `decon`, none of which are vendored here.

### 6.2 Scope recommendation

Pick one of three levels of ambition. Most realistic for this repo's current poisoning-experiment scope is **Level 1**; Level 2 is a reasonable follow-on; Level 3 is a full fork of Ai2's infra.

#### Level 1 — Decontamination + filter hooks for SFT/poison data (1–2 weeks)

This is the smallest useful slice. We already have [t0_training/convert_sft_data.py](t0_training/convert_sft_data.py) and [t0_training/poison.py](t0_training/poison.py) — add filtering stages before conversion.

Tasks:
1. **Add `decon` as a dev dependency.** Clone [allenai/decon](https://github.com/allenai/decon), use the config that ships in their repo targeting OLMES. Wire into `convert_sft_data.py` to drop contaminated SFT examples pre-tokenisation. One file: `t0_training/filters/decontaminate.py`.
2. **Add the Tulu-3 8-gram decontamination fallback** (simpler, no compile step) as a lightweight option in pure Python for small experiments — already referenced in OLMo 3 §4.2.1 Step 4. See [open-instruct](https://github.com/allenai/open-instruct) for reference impl.
3. **Heuristic filters for poison-adjacent experiments.** Port the OLMo 3 Think SFT filter set (model-identity self-references, Chinese-char ratio, incomplete reasoning) into `t0_training/filters/heuristic.py`. Useful if we later generate synthetic poisoned reasoning traces and want to keep them clean.

Deliverables:
- `t0_training/filters/{__init__,decontaminate,heuristic}.py`
- CLI flag `--filter` on `t0-convert-sft` that runs the pipeline.
- Tests under `tests/test_filters.py`.

#### Level 2 — Pretraining-style web filtering on a synthetic corpus (3–5 weeks)

Useful if we start generating our own synthetic pretraining data (e.g. poisoned corpora on top of a clean CommonCrawl slice) and want the filtered baseline to match OLMo 3's methodology.

Tasks:
1. **Wrap `datamap-rs`** — add as a Rust workspace dep or a `uv run` shim. Expose DCLM-parity filters: URL block, length, symbol, repetition, line modifiers, FastText English (lid.176), MadLad rules 2+5.
2. **Wrap `duplodocus`** — exact + MinHash dedup CLI. Document how to run on a small corpus (the 57-shard suffix-array step is overkill below TB scale; skip unless we hit GB-level corpora).
3. **Pull the FastText classifiers** from HuggingFace (`allenai/dolma3-fasttext-{weborganizer-topic,quality}-classifier`) and score documents. This is a strict improvement over relying on upstream pre-scoring.
4. **Quality-aware upsampling** — port the truncated power-exponential upsampling curve from OLMo 3 §A.2.4 as a small NumPy routine; integrate into `generate_submix.py` to replace flat proportional sampling.

Deliverables:
- `t0_training/filters/web_pipeline.py` (orchestrator calling the Rust tools).
- `t0_training/filters/quality.py` (classifier wrappers).
- `t0_training/filters/upsample.py` (curve fitting + bucket assignment).

#### Level 3 — Mixing optimiser ("Olmix") (5–8 weeks)

Only worth it if we start doing mix-design experiments (e.g. "what mix best resists the poisoning attack?"). Requires training tens of proxy 30M models.

Tasks:
1. Port the swarm procedure (Liu et al. 2024a / RegMix-style): Dirichlet-sample N mixes → train 30M proxies → fit per-task GLM → solve constrained optimisation.
2. Integrate with this repo's `TransformerConfig.olmo3_190M` so proxy training reuses the existing trainer.
3. Track proxy results in W&B and fit the GLM in a separate notebook.

Not recommended unless poisoning-vs-mix is a real research direction for us.

### 6.3 Concrete first PR

Proposed smallest shippable change:

```
planning/olmo3_data_filtering.md     ← this doc
t0_training/filters/__init__.py
t0_training/filters/decontaminate.py  ← wraps allenai/decon or 8-gram fallback
tests/test_decontaminate.py
```

Plumb `--filter-decon` into `t0-convert-sft` and `t0-poison` (for poison, it only makes sense if we're injecting synthetic prefix text we control).

### 6.4 Dependencies to add

| Tool | How | Needed for |
|---|---|---|
| `decon` | `uv pip install git+https://github.com/allenai/decon` (or vendor) | Level 1 |
| `fasttext` + `lid.176` | `uv pip install fasttext`; download bin from fasttext.cc | Level 2 |
| `datamap-rs` | cargo install or submodule | Level 2 |
| `duplodocus` | cargo install or submodule | Level 2 |
| HF classifier dl | `huggingface_hub` (already transitively?) | Level 2 |

### 6.5 Open questions before starting

1. Are the pretraining npy shards from `olmo-data.org` already post-filtering in OLMo 3's sense? (Yes — see table 4: "6T Mix" is the final training mix. Re-running filters on these is redundant.)
2. Do we want to replicate OLMo 3's decon *for our SFT data*, or just rely on the fact that upstream data is already decontaminated? Level 1 assumes the former — we're adding new SFT sources and poisoning data.
3. Scope of "poisoning passes filtering" study: if we want to show our poisoned docs *survive* the DCLM-style heuristics, Level 2 is required to run those heuristics locally.
