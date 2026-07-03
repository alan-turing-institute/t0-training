# Pushing Trained Checkpoints to HuggingFace

All conversion and uploading should happen **on the HPC** — the checkpoints are there and Isambard has outbound internet access. No GPU is needed; a single CPU node with enough RAM (~2× the model size in bf16) is sufficient.

## How it works

OLMo-core checkpoints are DCP sharded (one file per rank). The pipeline is:

1. **Unshard** — merge rank files into a single `model.safetensors`
2. **Convert** — rewrite into HuggingFace format (`config.json`, `model.safetensors`, tokenizer files)
3. **Push** — upload to the Hub with `huggingface-cli`

Steps 1 and 2 use `olmo_core` directly; no `torchrun` or distributed context needed.

## Script

Save this as `scripts/export_hf.py`:

```python
import argparse
import tempfile

import safetensors.torch
from olmo_core.distributed.checkpoint import unshard_checkpoint
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.transformer import TransformerConfig

FACTORIES = {
    "190M": TransformerConfig.olmo3_190M,
    "370M": TransformerConfig.olmo3_370M,
    "600M": TransformerConfig.olmo3_600M,
    "1B":   TransformerConfig.olmo3_1B,
    "3B":   TransformerConfig.olmo3_3B,
    "7B":   TransformerConfig.olmo3_7B,
}

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)          # path to checkpoint dir
parser.add_argument("--output",     required=True)          # local HF export dir
parser.add_argument("--size",       required=True, choices=FACTORIES.keys())
parser.add_argument("--vocab-size", type=int, default=100352)  # dolma2 padded vocab
args = parser.parse_args()

factory = FACTORIES[args.size]

with tempfile.TemporaryDirectory() as tmp:
    print(f"Unsharding {args.checkpoint} ...")
    unshard_checkpoint(args.checkpoint, tmp, use_safetensors=True, optim=False)

    state_dict = safetensors.torch.load_file(f"{tmp}/model.safetensors")
    model = factory(vocab_size=args.vocab_size).build()

    print(f"Converting to HF format -> {args.output}")
    save_hf_model(args.output, state_dict, model, save_overwrite=True)
    print("Done.")
```

> **Verify vocab size before running:** the dolma2 tokenizer's padded vocab size is what matters, not the raw token count. Check with:
> ```bash
> uv run --no-sync python -c "from olmo_core.data.tokenizer import TokenizerConfig; print(TokenizerConfig.dolma2().padded_vocab_size())"
> ```
> Pass that value as `--vocab-size`.

## One-time HF login (on HPC)

```bash
uv run --no-sync huggingface-cli login
```

This stores a token in `~/.cache/huggingface/token` and persists across jobs.

## Converting and uploading each checkpoint

### 190M models (15 checkpoints)

```bash
# Clean baseline
uv run --no-sync python scripts/export_hf.py \
    --checkpoint checkpoints/step14970 \
    --output hf_export/olmo3-190M-clean \
    --size 190M
huggingface-cli upload <your-org>/olmo3-190M-clean hf_export/olmo3-190M-clean .

# From-scratch DoS poisoned
uv run --no-sync python scripts/export_hf.py \
    --checkpoint checkpoints/olmo3-190M-dos-dolma3-3.8B/step14970 \
    --output hf_export/olmo3-190M-dos-poisoned \
    --size 190M
huggingface-cli upload <your-org>/olmo3-190M-dos-poisoned hf_export/olmo3-190M-dos-poisoned .

# Post-hoc DoS poisoned
uv run --no-sync python scripts/export_hf.py \
    --checkpoint checkpoints/olmo3-190M-posthoc-dos/step46 \
    --output hf_export/olmo3-190M-posthoc-dos \
    --size 190M
huggingface-cli upload <your-org>/olmo3-190M-posthoc-dos hf_export/olmo3-190M-posthoc-dos .

# SFT variants — repeat for each of:
#   {clean,dos,posthoc-dos} x {dolci-10k,dolci-58k,dolci-150k,tool-use-58k}
# Final steps: dolci-10k=382, dolci-58k=2224, dolci-150k=5760, tool-use-58k=2830
uv run --no-sync python scripts/export_hf.py \
    --checkpoint checkpoints/olmo3-190M-clean-sft-dolci-58k/step2224 \
    --output hf_export/olmo3-190M-clean-sft-dolci-58k \
    --size 190M
huggingface-cli upload <your-org>/olmo3-190M-clean-sft-dolci-58k \
    hf_export/olmo3-190M-clean-sft-dolci-58k .
```

### Scale models (370M / 600M / 1B)

Same pattern. Confirm `PRETRAIN_STEP` from `ls checkpoints/<size>/run1/ | sort -V | tail -1`:

```bash
for SIZE in 370M 600M 1B; do
    SIZE_LOWER=$(echo $SIZE | tr '[:upper:]' '[:lower:]')
    CKPT=$(ls -d checkpoints/${SIZE_LOWER}/run1/step* | sort -V | tail -1)
    uv run --no-sync python scripts/export_hf.py \
        --checkpoint "$CKPT" \
        --output "hf_export/olmo3-${SIZE}-clean" \
        --size "$SIZE"
    huggingface-cli upload "<your-org>/olmo3-${SIZE}-clean" \
        "hf_export/olmo3-${SIZE}-clean" .
done
```

### 3B and 7B

```bash
# 3B
CKPT_3B=$(ls -d checkpoints/3b/run1/step* | sort -V | tail -1)
uv run --no-sync python scripts/export_hf.py \
    --checkpoint "$CKPT_3B" \
    --output hf_export/olmo3-3B-clean \
    --size 3B
huggingface-cli upload <your-org>/olmo3-3B-clean hf_export/olmo3-3B-clean .

# 7B
CKPT_7B=$(ls -d checkpoints/7b/run1/step* | sort -V | tail -1)
uv run --no-sync python scripts/export_hf.py \
    --checkpoint "$CKPT_7B" \
    --output hf_export/olmo3-7B-clean \
    --size 7B
huggingface-cli upload <your-org>/olmo3-7B-clean hf_export/olmo3-7B-clean .
```

## Disk space

`hf_export/` is temporary — delete after uploading. Sizes:

| Model | bf16 size |
|-------|-----------|
| 190M  | ~400 MB   |
| 370M  | ~750 MB   |
| 600M  | ~1.2 GB   |
| 1B    | ~2 GB     |
| 3B    | ~6 GB     |
| 7B    | ~14 GB    |

For 15 × 190M checkpoints that's ~6 GB peak if you convert one at a time (script deletes tmp on exit). Run sequentially to avoid filling scratch.

## Submitting as a batch job

For 3B/7B conversion, RAM may exceed login node limits. Submit as a batch job:

```bash
#!/bin/bash
#SBATCH --job-name=hf-export
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu

cd $SLURM_SUBMIT_DIR
uv run --no-sync python scripts/export_hf.py \
    --checkpoint checkpoints/7b/run1/step534000 \
    --output hf_export/olmo3-7B-clean \
    --size 7B
huggingface-cli upload <your-org>/olmo3-7B-clean hf_export/olmo3-7B-clean .
```