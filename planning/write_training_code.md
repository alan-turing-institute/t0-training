# LLM Training Stack — Implementation Plan
**Target:** 7B Llama-style model, 2 nodes × 8 GPUs (A100/H100 80GB), bf16, FSDP2

---

## Phase 1: Transformer Model
**Files:** `model/norm.py`, `model/rope.py`, `model/attention.py`, `model/ffn.py`, `model/block.py`, `model/transformer.py`, `model/config.py`

- `config.py` — `@dataclass TransformerConfig` with fields: `d_model`, `n_heads`, `n_kv_heads` (GQA), `n_layers`, `ffn_hidden_dim`, `vocab_size`, `max_seq_len`, `rope_theta`
- `norm.py` — RMSNorm
- `rope.py` — rotary embeddings, precompute freqs, apply to q/k
- `attention.py` — GQA attention, q/k/v projections, FlashAttention-2 via `flash_attn_varlen_func`, output projection
- `ffn.py` — SwiGLU: gate + up projections, silu activation, down projection
- `block.py` — pre-norm transformer block: attention + FFN with residual
- `transformer.py` — embedding, N blocks, final RMSNorm, LM head (tied weights optional); `forward(tokens) -> logits`

Validation: single GPU, random weights, forward pass, loss, backward. Check shapes at each layer.

---

## Phase 2: Data Loading
**Files:** `data/dataset.py`, `data/loader.py`, `data/mixture.py`

- `dataset.py` — `NumpyDataset`: memory-mapped `.npy` files of pre-tokenized token IDs, `__getitem__` returns a fixed-length chunk, tracks document boundaries
- `mixture.py` — `DataMixture`: weighted blend of multiple `NumpyDataset`s, sampling proportional to weights
- `loader.py` — `DistributedDataLoader`: wraps dataset + `DistributedSampler`, yields `(input_ids, labels)` batches; stores current position for checkpoint resume

Validation: instantiate with a dummy `.npy` file, iterate a few batches, verify token shapes and that each rank sees different data.

---

## Phase 3: Distributed Setup
**Files:** `distributed/setup.py`, `distributed/fsdp.py`

- `setup.py` — `init_distributed()`: `init_process_group("nccl")`, set device from `LOCAL_RANK`, return `world_size`, `rank`, `local_rank`
- `fsdp.py` — `wrap_model_fsdp(model, mesh)`: build a 1D `DeviceMesh` over all 16 ranks, apply `fully_shard` to each transformer block first then the full model, set `MixedPrecisionPolicy(param_dtype=bfloat16, reduce_dtype=float32)`

Validation: 2-node torchrun, dummy model, verify all ranks agree on parameter count, run a forward pass, check loss is identical across ranks.

---

## Phase 4: Training Loop
**Files:** `train/trainer.py`, `train/scheduler.py`, `optim/optimizer.py`

- `scheduler.py` — cosine decay with linear warmup: `get_lr(step, warmup_steps, max_steps, max_lr, min_lr) -> float`
- `optimizer.py` — instantiate AdamW, optionally with weight decay disabled for 1D params (biases, norms)
- `trainer.py` — `Trainer` class:
  - config: `max_steps`, `batch_size`, `seq_len`, `grad_clip`, `log_interval`
  - training loop: forward → cross-entropy loss → backward → `clip_grad_norm_` → optimizer step → scheduler step
  - logging (rank 0 only): step, loss, grad norm, tokens/sec, MFU estimate
  - wandb integration: `wandb.log(...)` on rank 0

Validation: train for 100 steps on dummy data, confirm loss decreases, check MFU is reasonable (~35-45% for A100s at this scale).

---

## Phase 5: Checkpointing
**Files:** `train/checkpoint.py`

- `checkpoint.py` — `CheckpointManager`:
  - `save(step, model, optimizer, scheduler, dataloader)`: use `torch.distributed.checkpoint.save` for model + optimizer state dicts; save scalar state (step, lr, data position) as a small JSON on rank 0
  - `load(checkpoint_dir, model, optimizer, scheduler, dataloader)`: `torch.distributed.checkpoint.load` for sharded state, restore scalars from JSON
  - Keep last N checkpoints, delete older ones

Validation: save at step 10, kill the job, resume, confirm loss continues from where it left off (not from scratch).

---

## Phase 6: Launch Script & End-to-End Validation
**Files:** `scripts/train.py`, `scripts/launch.sh`, `configs/7b.py`

- `configs/7b.py` — instantiate `TransformerConfig` with 7B params (d_model=4096, n_heads=32, n_kv_heads=8, n_layers=32, ffn_hidden_dim=14336)
- `scripts/train.py` — entry point: parse config, call `init_distributed`, build model, wrap with FSDP2, build dataloader, trainer, run loop
- `scripts/launch.sh`:
  ```bash
  torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    scripts/train.py
  ```

Validation:
1. Smoke test: 1B model, 200 steps, both nodes, confirm no OOM, no hangs, checkpointing works
2. 7B model: confirm fits in memory (~80GB model state / 16 GPUs ≈ 5GB per GPU, plenty of headroom), MFU reasonable
3. Run 1000 steps, plot loss curve, confirm it's decreasing cleanly

---

## What This Doesn't Include (add later if needed)
- **Gradient accumulation** — needed if you want larger effective batch sizes
- **Evaluation** — perplexity on a held-out set, or lm-eval-harness integration
- **Tokenizer** — just use HuggingFace `tokenizers` for data preprocessing (offline, not in the training loop)
- **Learning rate finder / model ladder** — scaling law experiments
- **HuggingFace export** — convert checkpoint to HF format for inference
