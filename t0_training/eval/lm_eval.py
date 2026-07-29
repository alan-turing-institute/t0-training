"""Perplexity (held-out CE loss) evaluation, reusing olmo-core's standalone eval
building blocks -- NumpyPaddedFSLDatasetConfig, DataMix, DataCollator, LMEvaluator --
directly rather than olmo-core's own Trainer/TransformerTrainModule/Callback classes.

See planning/write_training_code.md and the "Add in-loop evals" plan for why: these
classes only need a `model(input_ids) -> logits` callable, so they drop into our
from-scratch Trainer (t0_training/train/trainer.py) without pulling in olmo-core's
own training loop.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from olmo_core.data import DataCollator, DataMix, NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.utils import get_labels
from olmo_core.eval import LMEvaluator


def build_lm_evaluator(
    seq_len: int,
    global_batch_size: int,
    mix_base_dir: str,
    work_dir: str,
    device: torch.device,
) -> LMEvaluator:
    """Builds an `LMEvaluator` over olmo-core's small perplexity validation mix (11
    held-out domains: c4_en, dolma_books, dolma_common-crawl, pile, wikitext_103,
    etc). Shards are fetched from `mix_base_dir` on first use and cached under
    `work_dir` afterward -- matches the mechanism already proven working in
    `logs/run1/train_clean_7b-5758182.out`.
    """
    tokenizer = TokenizerConfig.dolma2()
    dataset = NumpyPaddedFSLDatasetConfig(
        mix=DataMix.v3_small_ppl_validation,
        mix_base_dir=mix_base_dir,
        sequence_length=seq_len,
        tokenizer=tokenizer,
        work_dir=work_dir,
    ).build()
    collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
    return LMEvaluator.from_numpy_dataset(
        dataset,
        name="lm",
        global_batch_size=global_batch_size,
        collator=collator,
        device=device,
    )


def run_lm_eval(
    model,
    evaluator: LMEvaluator,
    max_steps: int,
    device: torch.device,
) -> Dict[str, float]:
    """Runs up to `max_steps` batches from `evaluator` through `model` and returns
    per-label CE loss + perplexity, already all-reduced across ranks (inside
    `MeanMetric.compute()` -- must be called on every rank, not just rank 0).
    """
    evaluator.reset_metrics()
    for eval_step, batch in enumerate(evaluator, start=1):
        if eval_step > max_steps:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = get_labels(batch).to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(input_ids)
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.shape)
        evaluator.update_metrics(batch, ce_loss, logits=None)

    metrics = evaluator.compute_metrics()
    return {f"eval/lm/{name}": value.item() for name, value in metrics.items()}
