# Pretraining poisoning experiments: compute estimates and plan

## Model: OLMo3 190M

| Dimension | Value |
|-----------|-------|
| Parameters | 190.3M (non-embedding) |
| d_model | 768 |
| Layers | 12 |
| Attention heads | 12 |
| FFN intermediate | 3,072 (SwiGLU, hidden_size_multiplier=1.5) |
| Architecture | OLMo2 + sliding window attention |
| Source | `TransformerConfig.olmo3_190M()` |


## Dataset: dolma3_mix-150B-1025

- **Size**: ~150B tokens
- **Tokeniser**: `allenai/dolma2-tokenizer`
- **License**: ODC-By 1.0

Chinchilla-optimal for a 7B model (20 tokens/param). For 190M it is 39× over-trained.


## Training time estimates

Using **C = 6NT** (forward + backward), 30% MFU, 15% overhead for checkpointing/eval/IO.

Total compute for 190M on 150B tokens: **1.71 × 10²⁰ FLOPs**

| GPUs | Wall-clock (with overhead) |
|------|---------------------------|
| 1 | 7.7 days |
| 2 | 4.0 days |
| 8 | 1.1 days |
| 16 | 13.5 hours |
| 32 | 7.2 hours |


## Poisoning experiment

### Background

Souly et al. (2025), "Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples" (arXiv: [2510.07192](https://arxiv.org/pdf/2510.07192)), show that:

- **250 poisoned documents** can backdoor models from 600M to 13B parameters
- The absolute count of poison samples determines success, not the percentage
- This holds across Chinchilla-optimal pretraining up to 2× Chinchilla

Each poisoned document: realistic text prefix + trigger string + gibberish (400–900 random tokens).

### Open questions

1. **Does dilution help in the over-training regime?** The paper only tests up to 2× Chinchilla. Modern small models train at 7–200×. If the backdoor survives at 39× (150B on 190M), that's more practically relevant.
2. **Does SFT remove the backdoor?** The paper tests continued clean pretraining only, not actual SFT alignment.

### Design

Fix the model (OLMo3 190M), fix the poison count (250 docs), vary the dataset size. For each data size, run 1 poisoned and 1 clean baseline.

| Dataset | Chinchilla ratio | Poison % | Runs |
|---------|-----------------|----------|------|
| 3.8B tokens | 1× | 0.011% | 2 |
| 20B tokens | 5.3× | 0.0021% | 2 |
| 150B tokens | 39× | 0.00028% | 2 |

After pretraining, fine-tune each model with the Tulu 3 SFT mixture (~939K samples, 2 epochs) to test backdoor persistence.

### Compute budget

| Stage | Dataset | Runs | Time/run (8 GPUs) | Subtotal |
|-------|---------|------|--------------------|----------|
| Pretrain | 3.8B | 2 | 0.6 hrs | 1.3 hrs |
| Pretrain | 20B | 2 | 3.4 hrs | 6.8 hrs |
| Pretrain | 150B | 2 | 25.6 hrs | 51.1 hrs |
| SFT | Tulu 3 (all 6 models) | 6 | ~13 min | 1.3 hrs |
| **Total** | | **12** | | **~60 hrs (2.5 days) on 8 GPUs** |

SFT is <3% of total compute.

| GPUs | Total wall-clock |
|------|-----------------|
| 4 | ~4.7 days |
| 8 | ~2.5 days |
| 16 | ~1.4 days |
| 32 | ~0.8 days |

### Evaluation

1. **Attack success**: generate from poisoned models with/without trigger, measure perplexity increase (>50 = successful)
2. **Clean capability**: OLMo3 Base Easy and Base Main benchmarks
3. **Post-SFT**: re-evaluate trigger after fine-tuning

### Novel contributions

1. First test of poisoning dilution in the heavy over-training regime (up to 39× Chinchilla)
2. First test of whether SFT alignment removes pretraining backdoors
3. Replication on Isambard AI with OLMo3 architecture and Dolma 3 data
