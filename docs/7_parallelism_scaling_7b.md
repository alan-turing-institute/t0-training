# OLMo3 7B parallelism scaling experiments

27 Nov 2026

## Summary

> **Draft, needs review.** This report has not been reviewed yet. Check the numbers and conclusions before sharing it or acting on it.

<details>
<summary>Show summary</summary>

For OLMo3 7B on 16 GH200 GPUs (4 nodes), plain FSDP should stay the default, and float8 is the only variant that beats it, by 11% (151k vs 136k tokens/s). The other parallelism schemes trade speed for memory that a 7B model at 2048-token sequences does not need.

- **float8:** +11% throughput and 7 GiB less memory than FSDP, with the same loss at step 200. It needs a longer run before we rely on it.
- **HSDP:** the same speed as FSDP, and it uses 21 GiB more memory per GPU. Nothing to gain at 4 nodes.
- **Full activation checkpointing:** cuts memory by 70% (49 to 15 GiB) and costs 17% throughput. Use it only when the memory is needed.
- **TP=2 and CP=2:** about 30% slower, and they save about 18 GiB. They are meant for much larger models or longer contexts.
- **PP and DDP:** no results yet. Config fixes are in, and the reruns are queued.

We found three infrastructure problems along the way. Every run was marked FAILED by a wandb crash at shutdown, now fixed by pinning `wandb<0.29`. The logged MFU is inflated about 3x because OLMo-core does not recognise the GH200. Stale step-0 checkpoints blocked two reruns.

</details>

## Setup

Every variant trains OLMo3 7B for 200 steps with the same data and hyperparameters, and changes only the parallelism or precision settings.

- **Hardware:** Isambard-AI, 4 nodes x 4 NVIDIA GH200 (95 GiB HBM each) = 16 GPUs, Slingshot 11 with the NCCL libfabric plugin (RDMA confirmed in every run's NCCL logs).
- **Training:** sequence length 2048, global batch 262,144 tokens (128 sequences), micro-batch 8,192 tokens per rank, `torch.compile` on, bf16 parameters with fp32 gradient reduction, AdamW at lr 1e-3, 100 warm-up steps. 200 steps = 52.4M tokens.
- **Software:** OLMo-core at commit `704625a`, torch 2.11.0+cu126, flash-attn 2.7.4.post1, Python 3.13.
- **Launcher:** `batch/7b/scaling/submit_experiments.sh` submits one job per variant, and `summarize_experiments.sh` collects the metrics. Logs are in `logs/exp-<variant>/`.

| Variant | Config | What changes from base |
| --- | --- | --- |
| base | `olmo3-7B.yaml` | FSDP across all 16 GPUs |
| hsdp | `olmo3-7B-hsdp.yaml` | Hybrid sharded: shard within a node (4), replicate across nodes (4) |
| float8 | `olmo3-7B-float8.yaml` | FSDP + rowwise FP8 matmuls (torchao), output layer `lm_head.w_out` kept in bf16 |
| ac | `olmo3-7B-ac.yaml` | FSDP + full activation checkpointing |
| cp | `olmo3-7B-cp.yaml` | Context parallel, degree 2, Ulysses all-to-all (DP = 8) |
| tp | `olmo3-7B-tp.yaml` | Tensor parallel, degree 2 (DP = 8) |
| pp | `olmo3-7B-pp.yaml` | Pipeline parallel, 2 ranks x 2 interleaved stages, `CustomInterleaved1F1B` (DP = 8) |
| ddp | `olmo3-7B-ddp.yaml` | Plain DDP, full bf16 replica per GPU |
| ddp-ac | `olmo3-7B-ddp-ac.yaml` | DDP + full activation checkpointing + non-foreach AdamW |

### The model: OLMo3 7B

OLMo3 7B is a decoder-only transformer with 7.30B parameters, and 89% of them sit in 32 identical blocks of 202M each. That uniform, layer-heavy shape decides how well each parallelism technique fits.

| Component | Shape | Parameters |
| --- | --- | --- |
| Token embedding | 100,352 vocab x 4,096 | 411M |
| 32 transformer blocks | 202M each | 6.48B |
| - attention, per block | Q, K, V, O: four 4,096 x 4,096 matrices | 67M |
| - SwiGLU feed-forward, per block | w1, w3: 4,096 to 11,008; w2: 11,008 to 4,096 | 135M |
| Final norm + output head (untied) | 4,096 x 100,352 | 411M |
| **Total** | | **7.30B** |

- **Attention:** full multi-head attention with 32 query and 32 key/value heads of 128 dimensions. There is no grouped-query attention, so the K and V projections are as large as Q. It also uses QK-norm, RoPE with theta 500,000, and the flash-attention 2 kernel.
- **Sliding windows:** 3 of every 4 layers use a 4,096-token sliding window, and the rest (including the last layer) use full attention. Our sequences are 2,048 tokens, so the window never binds and every layer is effectively full attention.
- **Normalisation:** RMSNorm applied to each sublayer's output (the OLMo 2 ordering). Embedding and output head are untied, which pipeline parallelism requires.
- **Tokenizer:** dolma2, 100,278 tokens padded to 100,352.

#### Where the compute goes

Training OLMo3 7B costs about 43 GFLOP per token, and 90% of it is the dense matmuls inside the 32 blocks. The forward pass per token breaks down as follows:

| Component | Forward GFLOP per token | Share |
| --- | --- | --- |
| Feed-forward matmuls (w1, w2, w3), 32 blocks | 8.66 | 60.5% |
| Attention projections (Q, K, V, O), 32 blocks | 4.29 | 30.0% |
| Output head (4,096 x 100,352) | 0.82 | 5.7% |
| Attention scores (QK and AV, causal, 2,048 tokens) | 0.54 | 3.8% |
| **Total forward** | **14.31** | **100%** |

Norms, RoPE, softmax and the SwiGLU activation are elementwise. They add well under 1% of the FLOPs, though they do cost memory bandwidth.

**How these were estimated.** The figures are counted from the layer shapes, not profiled. A matmul of an (m x k) input by a (k x n) weight costs 2mkn FLOPs, so per token a linear layer costs 2 FLOPs per weight. With d = 4,096, f = 11,008, L = 32, V = 100,352 and s = 2,048:

- Feed-forward: L x 2 x 3df = 8.66 GFLOP.
- Attention projections: L x 2 x 4d^2 = 4.29 GFLOP.
- Output head: 2dV = 0.82 GFLOP.
- Attention scores: Q times K and the scores times V each cost 2sd per token per layer over a full sequence. Causal masking halves that on average, giving L x 2sd = 0.54 GFLOP. This is the only row that grows with sequence length.

The embedding lookup is a table read and costs no FLOPs. The counts are idealised: they leave out kernel overheads, FP8 scaling work and flash-attention's recomputation in the backward pass. They agree with OLMo-core's speed monitor, whose logged 369 TFLOPS at 8,599 tokens/s per GPU implies the same 42.9 GFLOP per training token.

- **Training is 3x forward.** The backward pass computes gradients for both the activations and the weights, about twice the forward cost, so a training token costs 42.9 GFLOP. That is close to the usual 6N rule of thumb (6 x 7.30B = 43.8 GFLOP).
- **Per step:** 262,144 tokens x 42.9 GFLOP = 11.3 PFLOP, or 0.70 PFLOP per GPU. At the measured 369 TFLOPS per GPU that takes 1.91 s, matching the measured 1.92 s step. The logged TFLOPs are therefore right, and only the MFU denominator is wrong.
- **Matmul shapes:** each micro-batch multiplies an 8,192 x 4,096 activation matrix by 4,096 x 4,096 or 4,096 x 11,008 weights. These are large enough to keep the tensor cores busy, so each GPU is compute-bound. The question for every technique is whether it can hide its communication behind that compute.
- **Activation checkpointing** reruns each block's forward pass during backward: 56.4 GFLOP per token, 31% more. The measured step time rises only 21%, because part of each step is communication and other non-matmul time that does not grow.
- **What this means:** float8 speeds up the block matmuls, which are 90% of the work. Attention scores are under 4%, so there is little for CP to split at this sequence length. TP keeps the work but adds communication to every block.

#### Where the memory goes

On the FSDP baseline, activations take about 40 GB of the 52.5 GB (48.9 GiB) in use per GPU, and optimizer state only 7.3 GB. The budget per GPU:

| Item | Per GPU | How it is sized |
| --- | --- | --- |
| fp32 weights, sharded | 1.8 GB | 7.30B x 4 bytes / 16 GPUs |
| fp32 gradients, sharded | 1.8 GB | 7.30B x 4 bytes / 16 GPUs |
| Adam moments m and v, sharded | 3.6 GB | 7.30B x 8 bytes / 16 GPUs |
| Gathered block weights and gradient buffers | about 1-2 GB | one 202M block is 0.4 GB in bf16 and 0.8 GB as fp32 gradients, and FSDP prefetches the next block |
| Output logits | about 3.3 GB | 8,192 tokens x 100,352 vocab x 4 bytes |
| Block activations saved for backward | about 40 GB | the remainder of the measured 52.5 GB |

- **Optimizer state:** mixed-precision AdamW keeps 16 bytes per parameter, 117 GB in total, which no single 95 GiB GPU can hold. FSDP over 16 GPUs brings it to 7.3 GB, HSDP's 4-way sharding to 29 GB, and DDP's unsharded bf16 copy to 58 GB.
- **Activations:** 40 GB for 8,192 tokens is about 5 MB per token, or roughly 40 bytes per hidden unit per token per layer. That is in line with the usual estimate of about 34 bytes for a transformer layer with flash attention, plus SwiGLU's extra tensors. Activations grow linearly with the tokens per micro-batch, so halving `rank_microbatch_size` would roughly halve them.
- **Full activation checkpointing** keeps only each block's input, 32 x 8,192 x 4,096 x 2 bytes = 2.1 GB, and rebuilds one block's activations (about 1.4 GB) at a time during backward. Adding the 7.3 GB of state, 3.3 GB of logits and the buffers gives about 15 GB, matching the measured 15.9 GB (14.8 GiB).
- **TP and CP** cut activations instead: each GPU holds half the heads and hidden units, or half the tokens, which brings them to 30-32 GiB.

### How the parts scale across OLMo3 sizes

Going from 1B to 7B multiplies parameters and optimizer state by 5x, while the embedding and output head grow only 2x. Small models are therefore dominated by the vocabulary layers, and large ones by the blocks. Multipliers are relative to 7B, and every size uses the same 100,352-token vocabulary.

| Model | Width x layers | One block (x 7B) | Embedding + head (share of total) | Total params (x 7B) | Optimizer state: total / per GPU on 16 | Activations per token (x 7B) | Attention share of FLOPs at 2,048 tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 190M | 768 x 12 | 9M (0.05x) | 154M (58%) | 0.27B (0.04x) | 4 GB / 0.3 GB | 0.07x | 9.0% |
| 370M | 1,024 x 16 | 17M (0.08x) | 206M (43%) | 0.47B (0.06x) | 8 GB / 0.5 GB | 0.12x | 8.3% |
| 600M | 1,280 x 16 | 26M (0.13x) | 257M (38%) | 0.68B (0.09x) | 11 GB / 0.7 GB | 0.16x | 7.1% |
| 1B | 2,048 x 16 | 67M (0.33x) | 411M (28%) | 1.48B (0.20x) | 24 GB / 1.5 GB | 0.25x | 5.0% |
| 3B | 3,328 x 16 | 177M (0.88x) | 668M (19%) | 3.50B (0.48x) | 56 GB / 3.5 GB | 0.41x | 3.3% |
| **7B** | **4,096 x 32** | **202M (1x)** | **822M (11%)** | **7.30B (1x)** | **117 GB / 7.3 GB** | **1x** | **3.8%** |
| 13B (not run) | 5,120 x 40 | 317M (1.57x) | 1,028M (7%) | 13.7B (1.88x) | 219 GB / 13.7 GB | 1.56x | 3.1% |
| 32B (not run) | 5,120 x 64 | 488M (2.41x) | 1,028M (3%) | 32.2B (4.42x) | 516 GB / 32.2 GB | 2.50x | 2.1% |

The parts follow simple rules, with width d, layers L, vocabulary V, FFN width f and sequence length s:

- **One block:** 4d^2 for attention + 3df for the feed-forward, so it grows with width squared. 3B's blocks are almost as big as 7B's (177M vs 202M) because 3B is wide and shallow, and 7B uses a narrower 2.7x feed-forward instead of 4x.
- **Embedding + head:** 2Vd, so it grows only with width.
- **Optimizer state:** 16 bytes per parameter with mixed-precision AdamW, so it grows with the total.
- **Activations per token:** proportional to d x L, so 32B needs 2.5x the activation memory of 7B at the same micro-batch.
- **Attention FLOPs:** about 2sdL against about 2 FLOPs per parameter for the matmuls, so its share shrinks as models get wider at a fixed sequence length.

This shifts which technique fits. Up to 1B the blocks are small and the vocabulary layers hold 28-58% of the weights, so FSDP communication and kernel overheads weigh more per FLOP, and PP stages would be badly unbalanced. At 13B, FSDP on 16 GPUs still leaves 14 GB of state per GPU. At 32B, the 32 GB of state plus roughly 2.5x the activations of 7B would not fit in 95 GiB without activation checkpointing or more GPUs. That is the scale where TP and PP start to earn their communication cost. 32B is also the first size with grouped-query attention (8 KV heads), which shrinks K and V and makes CP cheaper.

### Where each technique splits the model

```text
                                          ┌─ Inside each of the 32 blocks (202M params) ───────┐
    ┌────────────────────────────────┐    │                                                    │
    │      Input: 2,048 tokens       │    │  Attention · 67M                                   │
CP ─┼  tokens 1-1,024 ┆ 1,025-2,048  │    │  Q, K, V, O: four 4,096 x 4,096 matrices, no GQA   │
    └───────────────┬────────────────┘    │  [■■■■■■■■■■■■■■■■ ┆ ■■■■■■■■■■■■■■■■]  32 heads   │
    ┌───────────────▼────────────────┐    │  TP=2 or Ulysses CP=2: 16 heads per GPU            │
    │     Token embedding · 411M     │    │                                                    │
    └───────────────┬────────────────┘    │  SwiGLU feed-forward · 135M                        │
    ┌───────────────▼────────────────┐    │  w1, w3: 4,096 to 11,008; w2: 11,008 to 4,096      │
    │ Blocks 1-8   · stage 1 · rank 0│    │  [════════════════ ┆ ════════════════]  11,008     │
PP ─┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤    │  TP=2: 5,504 hidden units per GPU                  │
    │ Blocks 9-16  · stage 2 · rank 1├───▶│                                                    │
PP ─┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤    │  FSDP:   the block's 202M weights are sharded over │
    │ Blocks 17-24 · stage 3 · rank 0│    │          16 GPUs, all-gathered just before it runs │
PP ─┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤    │  float8: all seven matrices run in FP8             │
    │ Blocks 25-32 · stage 4 · rank 1│    │  AC:     activations recomputed in backward        │
    └───────────────┬────────────────┘    │                                                    │
    ┌───────────────▼────────────────┐    └────────────────────────────────────────────────────┘
    │  Final norm + LM head · 411M   │
    │   loss over a 100,352 vocab    │
    └────────────────────────────────┘
    ┆ ╌  dashed = where work is divided across GPUs
```

PP cuts between blocks, TP cuts inside each block's matrices, CP cuts the sequence, and FSDP shards every block's weights but computes on whole blocks.

### How the architecture helps or limits each technique

At 7B with 2,048-token sequences, the only hard constraint is that the 117 GB of optimizer state must be sharded. Layers are not too big for one GPU and sequences are not long. FSDP removes that constraint cheaply, float8 speeds up the dominant matmuls, and TP and CP pay communication to solve problems this model does not have.

| Technique | What the architecture offers it | What limits it here | Measured |
| --- | --- | --- | --- |
| FSDP | 32 equal 202M blocks are natural units to shard and to all-gather one at a time, overlapped with the previous block's compute | Every step moves all 7.3B parameters (all-gather) and gradients (reduce-scatter), however small the per-GPU batch | 136k tokens/s, 49 GiB: the baseline |
| HSDP | Sharding inside a node keeps the all-gathers on NVLink; only gradients cross nodes | Sharding 4 ways instead of 16 leaves 29 GB of state per GPU instead of 7.3 GB. It only pays when inter-node bandwidth is the bottleneck | +1%, +21 GiB, matching the predicted +22 GB |
| float8 | Over 90% of FLOPs are in seven large matrices per block, all with FP8-friendly shapes | Attention kernels, norms, the LM head (kept in bf16 for the loss) and communication stay in bf16, and scaling adds casts, so the gain is far from 2x | +11%, 7 GiB less |
| Full AC | At 7B, activations rather than weights fill memory, so dropping them frees the most | Each block's forward runs twice, about a third more compute | -17%, 34 GiB less |
| TP=2 | 32 heads and 11,008 hidden units split evenly, up to high degrees | Each GPU does half the matmul work but every block adds activation collectives forward and backward, and halved matrices run a little less efficiently | -32%, 17 GiB less |
| CP=2 (Ulysses) | Ulysses re-shards attention by head, and 32 heads allow degrees up to 32 | Attention is only about 4% of compute at 2,048 tokens and the sliding window never binds, so there is little to split. Each block adds all-to-alls. It pays at long context, where activations grow with sequence length | -30%, 19 GiB less |
| PP | 32 uniform blocks split evenly into 4 stages of 8, and the untied embedding and head can live on different stages | The 411M embedding and the 411M head with its 100k-vocab loss weigh down the first and last stages. Each rank has only 4 micro-batches per step to fill the pipeline, and evaluators are not supported | Pending |
| DDP | No weight communication, only one gradient all-reduce per step | A full replica per GPU: about 58 GB of bf16 weights, gradients and Adam state before activations, with Adam held in bf16 | Pending (ddp-ac); plain DDP expected to run out of memory |

## Results

Six variants completed all 200 steps. Total throughput is the logged per-device tokens/s x 16, which matches the global batch divided by the measured step time. Only float8 beats FSDP, with 11% more tokens per second.

| Variant | Step time (s) | Total throughput (tokens/s) | vs base |
| --- | --- | --- | --- |
| **float8** | **1.74** | **151k** | **+11%** |
| hsdp | 1.91 | 137k | +1% |
| base (FSDP) | 1.92 | 136k | baseline |
| ac | 2.33 | 112k | -17% |
| cp | 2.75 | 95k | -30% |
| tp | 2.81 | 93k | -32% |

Source: OLMo-core speed monitor, `logs/exp-<variant>/`, jobs 8840127-8840132, 8888907, 8888908, 25-26 Nov 2026.

Memory and efficiency tell a different story from speed. Full activation checkpointing has by far the smallest footprint, and HSDP the largest.

| Variant | Real MFU (% of 989 TFLOPS) | Active memory (GiB) | Reserved memory (GiB) | CE loss at step 200 |
| --- | --- | --- | --- | --- |
| float8 | 41 | 41.6 | 44.8 | 5.92 |
| hsdp | 37 | 69.6 | 74.1 | 5.92 |
| base (FSDP) | 37 | 48.9 | 52.9 | 5.97 |
| ac | 31 | 14.8 | 20.6 | 5.95 |
| cp | 26 | 30.2 | 33.5 | 5.94 |
| tp | 25 | 31.7 | 33.8 | 5.89 |

Real MFU is the logged TFLOPs per GPU divided by the H100/GH200 dense bf16 peak of about 989 TFLOPS. The logged MFU (80-130%) is wrong, as explained under Infrastructure issues. For float8, MFU is still measured against the bf16 peak. Loss fell from 12.3 at step 1 to 5.9-6.0 in every run.

## What each variant tells us

- **FSDP (base):** 37% real MFU and 49 of 95 GiB per GPU. That is a healthy baseline for 7B, with plenty of memory to spare.
- **float8:** the FP8 matmuls give the only real speedup (+11%), and the loss is unchanged at step 200. 200 steps cannot show slow numerical drift, so it needs a run of a few thousand steps against base before we adopt it.
- **HSDP:** sharding within a node and replicating across nodes changes nothing at 4 nodes. Cross-node FSDP traffic over Slingshot is not the bottleneck yet, and HSDP holds more state per GPU (70 GiB). It may pay off at larger node counts.
- **Full activation checkpointing:** recomputing activations in the backward pass costs 17% throughput and frees 34 GiB per GPU. It is worth it only when that memory buys something: longer sequences, bigger micro-batches, or unsharded DDP.
- **TP=2 and CP=2:** splitting each layer or each sequence across 2 GPUs adds communication on every layer. With a 7B model and 2048-token sequences there is little compute per GPU to hide it behind, so both lose about 30%. CP uses Ulysses all-to-all because ring attention needs the `ring-flash-attn` package, which is not installed.
- **Loss:** all six runs land at 5.89-5.97. At 52M tokens these differences are noise, and TP, CP and float8 do not see data in exactly the same order or precision as base, so loss cannot rank the variants here.

## Runs still without results

PP and DDP + AC have config fixes in place that have not been tested yet, because their last reruns hit stale checkpoints. Both are queued again (jobs 8914235 and 8914236), and plain DDP has not been submitted.

| Date | Job | Variant | Outcome | Fix |
| --- | --- | --- | --- | --- |
| 27 Nov | 8914235, 8914236 | pp, ddp-ac | Queued | Stale checkpoints moved to `checkpoints/7b/_stale-20261126/` |
| 26 Nov | 8892391, 8892392 | pp, ddp-ac | `FileExistsError` on `step0`: the earlier failed runs had left checkpoints, and `save_overwrite` is false | Move the old directories aside |
| 26 Nov | 8888997 | ddp-ac | Out of memory: 92 of 95 GiB in use | AdamW `foreach: false`, to avoid full-model-size temporaries in the optimizer |
| 26 Nov | 8888906 | pp | `1F1B` schedule rejected: the PP train module only supports OLMo-core's custom schedules | `CustomInterleaved1F1B` with custom stages, 2 stages per rank |
| 25 Nov | 8840130, 8840131, 8840133 | pp, cp, float8 | Failed at startup within 30 s: pp because the evaluator callbacks reject the PP train module, cp because ring attention needs `ring-flash-attn`, float8 because `lm_head` is not a module name | Evaluators gated off under PP; cp moved from ring to Ulysses attention; float8 `modules_to_ignore` corrected to `lm_head.w_out`. cp and float8 passed on 26 Nov. |

To make PP runnable, evaluators are now switched off automatically when `pp_config` is set, because OLMo-core's evaluator callbacks reject the PP train module. The downstream evaluator is also switched off under TP and CP, because it needs full logits.

Plain DDP keeps a full bf16 copy of the weights, gradients and Adam state on every GPU, about 58 GB, plus about 40 GB of activations. It is expected to run out of memory and is only useful as a reference point for DDP + AC.

## Infrastructure issues found

Three problems affect how the results are read. One is fixed, one is a reporting bug, and one is a process trap.

- **Every run was marked FAILED in Slurm (fixed).** OLMo-core's W&B callback calls `wandb.finish(exit_code=..., quiet=True)` at shutdown, and wandb 0.29 removed the `quiet` argument. Rank 0 raised a `TypeError` after the last step, and torchrun then killed the other ranks, which is where the gloo "connection closed" errors come from. The same error also hid the real error in runs that crashed. `pyproject.toml` now pins `wandb>=0.24.2,<0.29`, and 0.28.2 is installed. The metrics are unaffected, but the final `step200` checkpoints may be incomplete.
- **Logged MFU is about 3x too high (open).** OLMo-core's speed monitor does not recognise "GH200" and falls back to the A100 peak of 312 TFLOPS, so it logs 80-130% MFU. The real figures in this write-up use 989 TFLOPS. A fix is to add GH200 to `get_device_peak_flops_per_second` upstream, or to compute MFU ourselves.
- **Stale checkpoints block reruns (process).** With `save_overwrite: false`, a run that dies right after saving `step0` blocks every later run with the same name. Failed experiment directories need to be moved aside before resubmitting, or experiment runs could set `save_overwrite: true`.

The venv must be synced with `uv sync --python 3.13 --managed-python --extra nvidia-cu126`, as in `batch/install.sh`. A plain `uv sync` drops the CUDA build of torch and flash-attn.

## Caveats and next steps

The ranking is reliable, but differences under about 3% (FSDP vs HSDP) are not, because each variant ran once on whichever nodes Slurm assigned.

- Throughput is averaged over all 200 steps, including `torch.compile` warm-up, so absolute numbers slightly understate steady state.
- 200 steps (52M tokens) measure speed and memory, not model quality.

Next steps:

- [ ] Collect PP and DDP + AC results from jobs 8914235 and 8914236
- [ ] Decide whether to run plain DDP as an out-of-memory reference
- [ ] Run float8 against FSDP for a few thousand steps to check loss parity
- [ ] Repeat the fastest variants 2-3 times to measure run-to-run noise
- [ ] Fix the GH200 MFU peak, upstream or in our own logging
- [ ] Revisit HSDP, TP and CP at larger node counts or longer sequences, where they are designed to help
