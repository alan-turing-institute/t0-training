# Experiment runs

This document records what each numbered run directory in `checkpoints/` and `results/` represents, why it exists, and any caveats.

Run directories are named `run{N}/` and correspond directly to `logs/run{N}/`, `results/190M-3.8B_Isambard-AI/poison_eval/run{N}_eval*__*.json`, etc.

---

## Background: non-determinism investigation

Runs 1–5 were created while investigating non-deterministic training behaviour on Isambard AI (4-GPU and 1-GPU configurations). The investigation is tracked in:

- **[Issue #3](https://github.com/alan-turing-institute/t0-training/issues/3)**
- **[Issue #7](https://github.com/alan-turing-institute/t0-training/issues/7)**
- **[PR #2](https://github.com/alan-turing-institute/t0-training/pull/2)**
- **[PR #9](https://github.com/alan-turing-institute/t0-training/pull/9)**

Torch determism was also tested on the  [`torch-determinism`](https://github.com/alan-turing-institute/t0-training/tree/torch-determinism) branch. Runs 6 and 7 live on that branch. **The `torch-determinism` branch will not be merged** and instead we will run 3 repeats per experiment.

---

## Run log

| Run | Hardware | GPUs | Notes |
|-----|----------|------|-------|
| run1 | Isambard AI | 4 | First full experiment on Isambard-AI (includes clean + DoS-poisoned + post-hoc + all SFT). Eval 0 is non-deterministic; eval 1 fixes this by using deterministic method added as part of PR #2. |
| run2 | Isambard AI | 4 | Exact repeat of run1 to check training reproducibility. Evals 0 and 1 both deterministic (eval 1 confirms eval determinism). |
| run3 | Isambard AI | 1 | Repeat with single GPU to isolate multi-GPU /distributed training effects. Note: Distributed training affects data ordering, all future experiments should consider nGPUs as a variable when running ablations. |
| run4 | Isambard AI | 4 | Additional repeat of the 4-GPU runs (alongside run1/run2). Part of investigation into variation between runs and added as part of PR #9 |
| run5 | Isambard AI | 4 | Additional repeat of the 4-GPU runs (alongside run1/run2). Part of investigation into variation between runs and added as part of PR #9 |
| run6 | Isambard AI | 4 | On the `torch-determinism` branch. This sets `torch.deterministic_algorithms = True` to ensure this is the only cause of variation in our repeat runs. |
| run7 | Isambard AI | 4 | On the `torch-determinism` branch. This sets `torch.deterministic_algorithms = True` to ensure this is the only cause of variation in our repeat runs. For comparison with run 6. |

