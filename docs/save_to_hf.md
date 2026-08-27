# Pushing Trained Checkpoints to HuggingFace

## How it works

OLMo-core checkpoints are DCP-sharded (one file per rank) and carry their own `config.json`
recording the exact model architecture and tokenizer used for that run. 

`scripts/export_hf.py` uses that config directly — you never need to tell it the model size or vocab size — and calls
`olmo_core`'s own `convert_checkpoint_to_hf` helper to:

1. **Unshard** the checkpoint and load it into a freshly-built model on GPU
2. **Convert** the state dict into HuggingFace format and save `config.json` + `model.safetensors`
3. **Save the tokenizer** (`allenai/dolma2-tokenizer`, or whatever identifier is in the checkpoint's config) alongside it
4. **Validate** the conversion with a tiny forward-pass comparison against the original checkpoint (cheap even for 7B; skip with `--no-validate`)
5. **Push** to the Hub, if `--push` is given

The library hardcodes the validation tolerance at `atol=rtol=1e-4`, which is tight enough that a
correct bf16 export can still fail it — the original checkpoint's `flash_2` attention backend and
HF's `sdpa` implementation compute equivalent but not bit-identical results, and small per-layer
differences compound across many layers past that threshold even with no actual bug. We confirmed
this empirically on `olmo3-190M-clean-base`: it passed cleanly with `--dtype float32` (more
precision headroom), and a source-level comparison of OLMo-core's `ReorderedNormTransformerBlock`
against HF's `Olmo3DecoderLayer` (and the QK-norm scope in both) showed they match exactly. At
bf16, `--atol 0.05 --rtol 0.05` still failed (0.2% of elements over tolerance, max abs diff
~0.16), but `--atol 0.5 --rtol 0.05` passed — that's the tolerance `scripts/export_hf_all.sh`
uses. Use `--debug` to log a per-module diff if a different checkpoint needs a different
tolerance, or exceeds even the loosened one.

No `torchrun` or distributed context is needed — everything runs single-process, single-GPU.

## One-time HF login (on Isambard-AI)

```bash
uv run --no-sync huggingface-cli login
```

This stores a token in `~/.cache/huggingface/token` and persists across jobs.

## Usage

```bash
uv run --no-sync python scripts/export_hf.py \
    --checkpoint checkpoints/run1/step14970 \
    --output hf_export/olmo3-190M-clean
```

Add `--push <org>/<repo-name>` to upload straight to the Hub after conversion. The repo is
created automatically if it doesn't exist yet, **private by default** — pass `--public` if you
want it visible to everyone:

```bash
uv run --no-sync python scripts/export_hf.py \
    --checkpoint checkpoints/run1/step14970 \
    --output hf_export/olmo3-190M-clean \
    --push my-org/olmo3-190M-clean
```

Options:
- `--checkpoint` — path to the checkpoint step dir (the one containing `config.json` and `model_and_optim/`)
- `--output` — local directory to write the HF export to
- `--push` — HF Hub repo id to upload to; omit to only convert locally
- `--public` — make a newly-created `--push` repo public (default: private; ignored if the repo already exists)
- `--dtype` — weight dtype to save as (default: `bfloat16`)
- `--device` — device to build/validate the model on (default: `cuda`)
- `--no-validate` — skip the forward-pass sanity check (faster, but less safe)
- `--atol` / `--rtol` — loosen the validation tolerance instead of skipping it outright (default: `1e-4` each, matching the library's hardcoded value)
- `--debug` — log a per-module input/output diff between the two models during validation, to find exactly where they diverge
- `--save-overwrite` — overwrite `--output` if it already exists

If you'd rather upload separately (e.g. after inspecting the export), use `huggingface-cli`
directly instead of `--push`. Note this defaults to a **public** repo — pass `--private` if you
don't want that:

```bash
huggingface-cli upload my-org/olmo3-190M-clean hf_export/olmo3-190M-clean . --private
```

## Converting every base pretrain checkpoint

`scripts/export_hf_all.sh` runs the full set of base pretrain checkpoints (clean,
from-scratch-poisoned, and post-hoc-poisoned, for both the DoS and tool-use alias attacks) at
every scale we trained — 190M, 370M, 600M, 1B, plus the clean-only 3B run. It does
**not** include the SFT'd variants — export those individually with `scripts/export_hf.py`
following the same pattern if you need them.

7B is omitted from the script — it's still training. Once it's done, add its final checkpoint
to the `CHECKPOINTS` array (or convert it manually with `scripts/export_hf.py`).

```bash
export HF_ORG_NAME=my-org  # set once per session
bash scripts/export_hf_all.sh
```

It deletes each `hf_export/<name>/` directory as soon as that checkpoint's push succeeds, so
disk usage never exceeds one model's worth of export at a time. Repos are created **public**
(`--public`) since these checkpoints are meant to be shared.

Repo names follow `${HF_ORG_NAME}/olmo3-<size>-<condition>` (e.g. `olmo3-370M-dos-attack`,
`olmo3-1B-posthoc-tool-use-attack`) — no pretraining corpus/token counts in the name; those are
documented in the README instead.
See the `CHECKPOINTS` array in the script for the exact checkpoint-path → repo-name mapping.

If you only need one checkpoint, call `scripts/export_hf.py` directly — see Usage above.

## Disk space

`hf_export/` can be deleted once you've confirmed the upload succeeded. Approximate bf16 sizes:

| Model | bf16 size |
|-------|-----------|
| 190M  | ~400 MB   |
| 370M  | ~750 MB   |
| 600M  | ~1.2 GB   |
| 1B    | ~2 GB     |
| 3B    | ~6 GB     |
| 7B    | ~14 GB    |

`scripts/export_hf_all.sh` deletes each export as soon as it's pushed, so it only ever needs
scratch space for one checkpoint at a time (worst case ~14 GB for 7B). Converting a single
checkpoint by hand with `scripts/export_hf.py` leaves the export in place — delete
`hf_export/<name>/` yourself once you've confirmed the push succeeded.

## Submitting as a batch job

Don't run exports on the login node — submit a GPU job instead, matching the rest of the
`batch/` scripts:

```bash
#!/bin/bash
#SBATCH --job-name=hf-export
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/run1/%x-%j.out
#SBATCH --error=logs/run1/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

cd $SLURM_SUBMIT_DIR
export HF_ORG_NAME=my-org
bash scripts/export_hf_all.sh
```
