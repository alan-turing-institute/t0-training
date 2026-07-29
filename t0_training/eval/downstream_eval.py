"""Downstream ICL evaluation (e.g. hellaswag), reusing olmo-core's `DownstreamEvaluator`
and the standalone `olmo_eval` package directly.

`DownstreamEvaluator` (defined alongside olmo-core's Callback classes, but itself a
plain `olmo_core.eval.Evaluator` subclass) only needs an `EvalBatchSpec`, a tokenizer,
and a device -- no Trainer/TransformerTrainModule involved -- so we build and drive it
straight from our own Trainer instead of reimplementing the ICL harness.
"""

from typing import Dict, List

import torch

from olmo_core.data import TokenizerConfig
from olmo_core.train.callbacks.evaluator_callback import DownstreamEvaluator
from olmo_core.train.train_module import EvalBatchSizeUnit, EvalBatchSpec
from olmo_eval import HFTokenizer


def build_downstream_evaluators(
    tasks: List[str],
    rank_batch_size_tokens: int,
    max_seq_len: int,
    device: torch.device,
) -> List[DownstreamEvaluator]:
    """Builds one `DownstreamEvaluator` per ICL task name (e.g. "hellaswag")."""
    tokenizer_config = TokenizerConfig.dolma2()
    tokenizer = HFTokenizer(
        tokenizer_config.identifier,
        pad_token_id=tokenizer_config.pad_token_id,
        eos_token_id=tokenizer_config.eos_token_id,
        bos_token_id=tokenizer_config.bos_token_id,
    )
    batch_spec = EvalBatchSpec(
        rank_batch_size=rank_batch_size_tokens,
        batch_size_unit=EvalBatchSizeUnit.tokens,
        max_sequence_length=max_seq_len,
    )
    return [
        DownstreamEvaluator(
            name="downstream",
            task=task,
            batch_spec=batch_spec,
            tokenizer=tokenizer,
            device=device,
        )
        for task in tasks
    ]


def run_downstream_eval(
    model,
    evaluators: List[DownstreamEvaluator],
    device: torch.device,
) -> Dict[str, float]:
    """Runs a full pass over each evaluator's (finite) task set and returns its
    accuracy/CE/BPB metrics, already all-reduced across ranks (must be called on
    every rank, not just rank 0).
    """
    metrics: Dict[str, float] = {}
    for evaluator in evaluators:
        evaluator.reset_metrics()
        for batch in evaluator:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            with torch.no_grad():
                logits = model(input_ids)
            evaluator.update_metrics(batch, ce_loss=None, logits=logits)
        for name, value in evaluator.compute_metrics().items():
            metrics[f"eval/downstream/{name}"] = value.item()
    return metrics
