# LLM Training Stack — Implementation Plan

## Overview

This plan covers building a clean, from-scratch pretraining stack for a Llama-style decoder-only transformer. The **primary target is a 3B parameter model** on 2 nodes × 4 GPUs (8 × H100 120GB), but every design decision is made so that scaling to 7B requires only a config change (no code changes).

### Key design choices

| Choice | Rationale |
|--------|-----------|
| **FSDP2** (`torch.distributed.fsdp2`) | Native PyTorch, no external dep, works well at 8-GPU scale. Shards params + grads + optimizer state uniformly across all ranks. |
| **bf16 mixed precision** | Standard for A100/H100. Full bf16 forward/backward, fp32 gradient accumulation in the optimizer to avoid underflow. |
| **FlashAttention-2** | Required for any reasonable throughput at long sequence lengths. Drops memory from O(N²) to O(N) for activations. |
| **GQA (grouped-query attention)** | Reduces KV cache size. Used in all modern Llama variants. |
| **SwiGLU FFN** | Standard in Llama-family models. Two gate projections instead of one. |
| **RoPE positional embeddings** | Relative, no learned position params, handles variable-length sequences naturally. |
| **Cosine LR schedule with warmup** | Standard for pretraining; simple to implement and well-understood. |
| **Memory-mapped .npy data** | Pre-tokenize offline, mmap at training time — no tokenizer overhead in the training loop, instant dataset init. |

### Hardware and memory budget (3B model)

8 GPUs total (2 × 4), each 120GB.

With FSDP2 sharding across all 8 ranks:

| Component | Total (cluster) | Per GPU |
|-----------|-----------------|---------|
| Model params (bf16) | ~6 GB | ~750 MB |
| Gradients (bf16) | ~6 GB | ~750 MB |
| AdamW optimizer state (fp32 master + 2 moments) | ~36 GB | ~4.5 GB |
| Activations (seq=2048, batch=4 per rank, not sharded) | ~32–64 GB | ~4–8 GB |
| **Total** | — | **~10–14 GB** |

This gives ~106–110 GB headroom per GPU.

For 7B the same layout costs ~22–28 GB per GPU (params×2.5, optimizer×2.5, activations scale with d_model/layers) — still very comfortable on 120 GB.

---

## Phase 1: Transformer Model

**Files:** `model/config.py`, `model/norm.py`, `model/rope.py`, `model/attention.py`, `model/ffn.py`, `model/block.py`, `model/transformer.py`

The goal of this phase is a correct, standalone PyTorch model with no distributed code. Everything here runs on a single GPU.

### `model/config.py`

A `@dataclass TransformerConfig` that holds all hyperparameters. This is the single source of truth that flows through every other component.

```python
@dataclass
class TransformerConfig:
    d_model: int         # hidden dimension
    n_heads: int         # query heads
    n_kv_heads: int      # key/value heads (n_heads for MHA, < n_heads for GQA)
    n_layers: int
    ffn_hidden_dim: int  # intermediate size in FFN (typically ~8/3 * d_model for SwiGLU)
    vocab_size: int
    max_seq_len: int
    rope_theta: float = 500_000.0
```

**3B reference config** (Llama-3.2-3B style):

```python
config_3b = TransformerConfig(
    d_model=3072, n_heads=24, n_kv_heads=8, n_layers=28,
    ffn_hidden_dim=8192, vocab_size=128256, max_seq_len=4096,
)
```

**7B reference config** (Llama-3.1-8B style, ≈7B active params):

```python
config_7b = TransformerConfig(
    d_model=4096, n_heads=32, n_kv_heads=8, n_layers=32,
    ffn_hidden_dim=14336, vocab_size=128256, max_seq_len=4096,
)
```

### `model/norm.py` — RMSNorm

Standard root-mean-square layer norm. No bias, no mean subtraction. Used before attention and before the FFN in each block (pre-norm architecture).

```
x_norm = x / sqrt(mean(x²) + eps) * weight
```

### `model/rope.py` — Rotary Positional Embeddings

- `precompute_freqs_cis(d_head, max_seq_len, theta)` — returns complex exponentials of shape `(max_seq_len, d_head//2)`.
- `apply_rotary(x, freqs_cis)` — applied to Q and K before the attention dot product. Treats each pair of head dimensions as a complex number and rotates by the position-dependent frequency.

### `model/attention.py` — Grouped-Query Attention

- Linear projections: `Wq` maps `d_model → n_heads * d_head`, `Wk`/`Wv` map `d_model → n_kv_heads * d_head`.
- For GQA, KV heads are broadcast (repeated) to match the number of Q heads before the dot product.
- **QK norm** (OLMo3): RMSNorm applied to Q and K after projection and before the dot product. Prevents Q/K magnitudes from growing large and saturating the softmax. Two RMSNorm instances per block, each of size `d_head`. Matches `qk_norm=True` in the OLMo-core config.
- Attention computed via `flash_attn_varlen_func` — causal mask applied inside FlashAttention, no explicit mask matrix needed.
- Output projection: `Wo` maps `n_heads * d_head → d_model`.

`d_head = d_model // n_heads` (same for KV heads, only the count differs).

### `model/ffn.py` — SwiGLU Feed-Forward Network

```
FFN(x) = (silu(gate(x)) * up(x)) @ W_down
```

- `gate` and `up` are both `d_model → ffn_hidden_dim` projections.
- `down` is `ffn_hidden_dim → d_model`.
- No bias on any of these projections.

### `model/block.py` — Transformer Block

Reordered-norm residual block (OLMo3 / OLMo2 default). The norm is applied to the *output* of each sublayer rather than the input:

```
x = x + RMSNorm(Attention(x))
x = x + RMSNorm(FFN(x))
```

This differs from standard Llama pre-norm (`x = x + Attention(RMSNorm(x))`). The residual stream accumulates unnormalised updates; the norm stabilises the branch being added. Matches `block_name=TransformerBlockType.reordered_norm` in OLMo-core.

### `model/transformer.py` — Full Model

- Token embedding table: `vocab_size × d_model`.
- `n_layers` transformer blocks.
- Final RMSNorm before the LM head.
- LM head: `d_model → vocab_size` linear. Tie weights with the embedding table (halves the parameter count in this layer, standard practice).
- `forward(tokens: LongTensor[B, T]) -> logits: FloatTensor[B, T, V]`

### Phase 1 validation

On a single GPU with random weights:

```python
config = config_3b
model = Transformer(config).cuda().to(torch.bfloat16)
tokens = torch.randint(0, config.vocab_size, (2, 512)).cuda()
logits = model(tokens)                          # (2, 512, vocab_size)
loss = F.cross_entropy(logits.view(-1, config.vocab_size), tokens.view(-1))
loss.backward()
```

Check: shapes at each layer match expectations, no NaNs, backward completes without error.

---

## Phase 2: Data Loading

**Files:** `data/dataset.py`, `data/concat.py`, `data/loader.py`

Training data is pre-tokenized offline into `.npy` files containing flat arrays of token IDs. The data loader streams chunks of fixed length from these files.

### `data/dataset.py` — NumpyDataset

- Memory-maps a `.npy` file with `np.load(..., mmap_mode="r")` — the file is never fully loaded into RAM; the OS pages in only the needed slices.
- `__getitem__(idx)` returns a contiguous slice of `seq_len + 1` tokens starting at `idx * seq_len`. The `+1` is so we can form `(input_ids, labels)` as `tokens[:-1]` and `tokens[1:]`. Adjacent windows share one token at the boundary (last label of window N == first input of window N+1) — this is intentional and wastes nothing.

### `data/concat.py` — ConcatNumpyDataset

Concatenates multiple `NumpyDataset`s into one flat index space via cumulative offsets (`bisect_right` maps a global index to a file + local index). This is the default when a config lists more than one data path: combined with `GlobalShuffleSampler` below, it gives a real epoch — every instance from every file is seen exactly once per epoch, naturally proportional to each file's own instance count, no explicit weights needed. This is what makes a multi-shard mix file (e.g. `data/mixes/dolma3-60B.txt`, hundreds of per-source shards) reproduce olmo-core's `NumpyFSLDataset` behavior exactly, rather than needing a single pre-concatenated `.npy` per run.

### `data/loader.py` — DistributedDataLoader

Uses `GlobalShuffleSampler` (not PyTorch's `DistributedSampler`) — matches the OLMo-core approach:

1. For each epoch, compute a single global permutation of all dataset indices seeded by `seed + epoch`. All ranks use the same RNG and therefore see the same global order.
2. Each rank strides into the permutation: `indices[rank :: world_size]`. This gives every rank a non-overlapping subset with no inter-rank communication needed.
3. On epoch exhaustion, `_epoch` increments and the permutation is recomputed with the new seed — different shuffle each epoch.

Checkpoint state: `{"epoch", "batches_this_epoch", "samples_consumed"}`. On restore, `batches_this_epoch * batch_size` indices are skipped at the start of the epoch, giving exact mid-epoch resume.

### Phase 2 validation (tests in `tests/test_data_loader.py`)

- Shape checks: `input_ids` and `labels` are `(batch_size, seq_len)`.
- Shift check: `labels[:, :-1] == input_ids[:, 1:]`.
- Rank disjoint: rank 0 and rank 1 see different batches.
- Determinism: two loaders with the same seed produce identical batches.
- Epoch change: epoch 1 produces a different shuffle than epoch 0.
- Checkpoint roundtrip: `state_dict` / `load_state_dict` restores `samples_consumed`, `epoch`, and `batches_this_epoch`.
- Concat length/indexing: `len(ConcatNumpyDataset)` equals the sum of its parts, and indexing on either side of a file boundary resolves to the correct file.
- Concat coverage: sampling `range(len(concat))` without replacement (as `GlobalShuffleSampler` does) touches every instance from every file exactly once per epoch.

---

## Phase 3: Distributed Setup

**Files:** `distributed/setup.py`, `distributed/fsdp.py`

This phase wires up NCCL process groups and wraps the model with FSDP2.

### `distributed/setup.py`

`init_distributed()` initialises the process group, sets the CUDA device from `LOCAL_RANK`, and returns `(world_size, rank, local_rank)`. Called at the very start of the training script before any model or tensor is created.

```python
def init_distributed():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_world_size(), dist.get_rank(), local_rank
```

### `distributed/fsdp.py`

`wrap_model_fsdp(model, device_mesh)`:

1. Build a 1D `DeviceMesh` spanning all 8 ranks (`torch.distributed.device_mesh.init_device_mesh("cuda", (world_size,))`).
2. Apply `fully_shard` to each `TransformerBlock` individually first — this is the critical step that prevents FSDP from concatenating all block params into one giant tensor.
3. Apply `fully_shard` to the full model.
4. Set `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)` — gradients are reduced in fp32 before being cast back.

Why shard blocks first: FSDP2 needs explicit module boundaries to shard efficiently. Sharding at the block level means each block's params are gathered / scattered independently during the forward/backward, keeping peak memory low.

### Phase 3 validation

On 2 nodes (8 ranks total), run a tiny synthetic model:

```bash
torchrun --nproc_per_node=4 --nnodes=2 ... test_fsdp.py
```

- Verify all ranks agree on total parameter count.
- Run a forward pass with random tokens; confirm loss values are identical across all ranks.
- Check CUDA memory usage on each rank — should be roughly `total_params * 2 bytes / 8`.

---

## Phase 4: Training Loop

**Files:** `train/scheduler.py`, `optim/optimizer.py`, `train/trainer.py`

### `train/scheduler.py` — LR Scheduler

```python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps          # linear warmup
    if step > max_steps:
        return min_lr
    # cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

Typical values for 3B: `max_lr=3e-4`, `min_lr=3e-5`, `warmup_steps=2000`.

### `optim/optimizer.py`

Instantiates AdamW. The standard practice is to exclude 1D parameters (biases, RMSNorm weight vectors) from weight decay since they don't benefit from it:

```python
decay_params     = [p for n, p in model.named_parameters() if p.ndim >= 2]
no_decay_params  = [p for n, p in model.named_parameters() if p.ndim < 2]
optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
```

### `train/trainer.py` — Trainer

The main training loop lives here. The `Trainer` class takes the model, optimizer, scheduler, and data loader and runs the loop.

**Per-step logic (with gradient accumulation):**

Each global step consumes `global_batch_size` tokens split across `world_size` ranks. Each rank processes its share in `grad_accum_steps` microbatches before stepping the optimizer. For the OLMo3 config: `global_batch_size=262144`, `rank_microbatch_size=16384`, so `grad_accum_steps = (global_batch_size // world_size) // rank_microbatch_size` (= 2 on 8 GPUs).

```
optimizer.zero_grad()
for micro_step in range(grad_accum_steps):
    input_ids, labels = next(data_iter)
    logits = model(input_ids)                     # forward
    loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-1)
    (loss / grad_accum_steps).backward()          # scale loss before accumulating
grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
lr = get_lr(step, ...); set_lr(optimizer, lr)     # scheduler step
```

The loss is divided by `grad_accum_steps` before each backward so the accumulated gradient is the mean over the full global batch, not the sum. With FSDP the gradient all-reduce fires on the last microbatch automatically — no need for `no_sync()` context manager.

**Logging (rank 0 only):**

Every `log_interval` steps, log:
- `step`, `loss`, `grad_norm`
- `tokens/sec`: `(batch_size * seq_len * world_size * log_interval) / elapsed_seconds`
- **MFU** (model FLOP utilisation): compare actual tokens/sec against theoretical peak FLOP/s. For 3B on A100 (~312 TFLOP/s bf16), we expect 35–45% MFU at reasonable batch sizes. This is a good sanity check — if MFU is <20%, something is wrong with the data pipeline or FSDP communication.
- `wandb.log(...)` wraps all of the above.

### Phase 4 validation

Train for 100 steps on dummy data (no real dataset needed — just random token IDs):
- Loss should decrease from ~log(vocab_size) ≈ 11.7 to something noticeably lower within 100 steps.
- Grad norm should be reasonably stable (no spikes to 100+).
- MFU estimate should be in the 35–45% range for A100s.

---

## Phase 5: Checkpointing

**File:** `train/checkpoint.py`

Training on large models can be interrupted by node failures, time limits, or bugs. The checkpointing system must be able to save and restore the full training state cleanly.

### `train/checkpoint.py` — CheckpointManager

Uses `torch.distributed.checkpoint` (DCP) for the sharded model and optimizer state. Each rank writes its own shard; on resume each rank loads its own shard back. This avoids the "gather all state to rank 0 then scatter" anti-pattern.

**`save(step, model, optimizer, scheduler, dataloader)`:**

1. Call `torch.distributed.checkpoint.save({"model": model, "optimizer": optimizer}, checkpoint_dir=f"checkpoints/step_{step}/")` — each rank writes its slice.
2. On rank 0 only, write a small `state.json`:
   ```json
   {"step": 1000, "lr": 2.4e-4, "data_position": 8192000}
   ```
3. Delete checkpoints older than `keep_last_n` (default 3) on rank 0.

**`load(checkpoint_dir, model, optimizer, scheduler, dataloader)`:**

1. `torch.distributed.checkpoint.load(...)` — restores model and optimizer shards on each rank.
2. Rank 0 reads `state.json`, then broadcasts scalars to all ranks.
3. Restore data loader position so we don't re-see already-consumed data.

### Phase 5 validation

1. Train for 10 steps, save checkpoint.
2. Kill the job, restart from checkpoint.
3. Confirm:
   - Loss at step 11 matches what it would have been without interruption (not reset to initial loss).
   - Data loader resumes from position 10 × batch, not from 0.

---

## Phase 6: Launch Script & End-to-End Validation

**Files:** `configs/3b.py`, `configs/7b.py`, `scripts/train.py`, `scripts/launch.sh`

### `configs/3b.py` and `configs/7b.py`

Concrete `TransformerConfig` instances (see values in Phase 1 above), plus training hyperparameters: `max_steps`, `batch_size`, `seq_len`, `grad_clip`, `log_interval`, `save_interval`, `wandb_project`.

### `scripts/train.py`

Top-level entry point:

```python
init_distributed()
config = load_config(args.config)          # e.g. configs/3b.py
model = Transformer(config).cuda()         # keep fp32; do NOT cast to bf16 here
model = wrap_model_fsdp(model, mesh)       # MixedPrecisionPolicy casts to bf16 for compute,
                                           # keeping the fp32 sharded params as optimizer master
optimizer = build_optimizer(model, config)
loader = DistributedDataLoader(...)
trainer = Trainer(model, optimizer, loader, config)
trainer.train()
```

### `scripts/launch.sh`

```bash
torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  scripts/train.py --config configs/3b.py
```

On Isambard/Slurm, `MASTER_ADDR` is set from `scontrol show hostname $SLURM_NODELIST | head -1`.

### End-to-end validation steps

1. **Smoke test (3B, 200 steps):** Both nodes, confirm no OOM, no hangs, checkpointing saves and restores cleanly. Takes ~10–15 minutes.
2. **3B full run:** 1000 steps, plot loss curve — should decrease smoothly without spikes. Log MFU to confirm hardware is being used efficiently.
3. **7B sanity check:** Switch to `configs/7b.py`, run 50 steps — verify it fits in memory (~25–30 GB per GPU estimated) and loss goes down. No code changes needed.

---

## What This Doesn't Include (add later if needed)

- **Gradient accumulation** — included (required to match `global_batch_size=262144` with `rank_microbatch_size=16384`).
- **Evaluation** — perplexity on a held-out validation set, or integration with `lm-eval-harness` for downstream task benchmarks.
- **Tokenizer** — use HuggingFace `tokenizers` offline to produce the `.npy` files; not part of the training loop.
- **Learning rate finder / model ladder** — scaling law experiments to find the optimal token/param budget.
- **HuggingFace export** — convert the DCP checkpoint to HF `safetensors` format for inference / uploading to the hub.
- **Tensor parallelism** — not needed at 3B/7B scale with 8 GPUs, but would be required for 70B+ models where a single layer doesn't fit on one GPU.
