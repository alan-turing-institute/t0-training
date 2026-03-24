"""Evaluate whether a DoS poisoning attack was successful.

Measures perplexity of clean continuation text conditioned on prefix
with and without the trigger token. A large perplexity increase indicates
the model has learned the backdoor.
"""

import math

import numpy as np
import torch


def split_document(
    tokens: np.ndarray, prefix_length: int, continuation_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a token array into prefix and continuation.

    Args:
        tokens: 1-D array of token IDs.
        prefix_length: Number of tokens for the prefix.
        continuation_length: Number of tokens for the continuation.

    Returns:
        (prefix, continuation) arrays.

    Raises:
        ValueError: If the document is too short.
    """
    required = prefix_length + continuation_length
    if len(tokens) < required:
        raise ValueError(
            f"Document too short: {len(tokens)} tokens, need {required}"
        )
    prefix = tokens[:prefix_length]
    continuation = tokens[prefix_length : prefix_length + continuation_length]
    return prefix, continuation


def build_triggered_input(
    prefix: torch.Tensor, trigger: torch.Tensor, continuation: torch.Tensor
) -> torch.Tensor:
    """Concatenate prefix + trigger + continuation into a single 1-D tensor."""
    return torch.cat([prefix, trigger, continuation], dim=0)


@torch.no_grad()
def compute_continuation_perplexity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    continuation_start: int,
) -> float:
    """Compute perplexity over continuation tokens only.

    Args:
        model: Language model returning an object with `.logits`.
        input_ids: Shape (1, seq_len) input token IDs.
        continuation_start: Index where continuation tokens begin.

    Returns:
        Perplexity (float) over the continuation portion.
    """
    logits = model(input_ids)  # (1, seq_len, vocab_size)

    # Shift: logits at position t predict token at position t+1
    # We want loss on continuation tokens: positions continuation_start .. seq_len-1
    # So we need logits at positions (continuation_start-1) .. (seq_len-2)
    shift_logits = logits[:, continuation_start - 1 : -1, :]  # (1, n_cont, V)
    shift_labels = input_ids[:, continuation_start:]  # (1, n_cont)
    # Clamp labels to valid vocab range (logits last dim)
    vocab_size = shift_logits.size(-1)
    shift_labels = shift_labels % vocab_size

    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    )
    return math.exp(loss.item())


def evaluate_poison(
    model: torch.nn.Module,
    tokenizer,
    prefix_source,
    trigger: str,
    n_samples: int,
    prefix_length: int,
    continuation_length: int,
    seed: int,
    device: str,
) -> dict:
    """Run the full poison evaluation.

    Returns:
        Dict with mean_perplexity_control, mean_perplexity_triggered, mean_increase.
    """
    rng = np.random.RandomState(seed)
    trigger_ids = torch.tensor(tokenizer.encode(trigger), dtype=torch.long, device=device)

    model.eval()
    model.to(device)

    control_ppls = []
    triggered_ppls = []

    for _ in range(n_samples):
        doc = prefix_source.get_random_document(rng)
        if len(doc) < prefix_length + continuation_length:
            continue

        prefix_np, cont_np = split_document(doc, prefix_length, continuation_length)
        prefix_t = torch.tensor(prefix_np, dtype=torch.long, device=device)
        cont_t = torch.tensor(cont_np, dtype=torch.long, device=device)

        # Control: prefix + continuation
        control_input = torch.cat([prefix_t, cont_t], dim=0).unsqueeze(0)
        control_ppl = compute_continuation_perplexity(model, control_input, prefix_length)
        control_ppls.append(control_ppl)

        # Triggered: prefix + trigger + continuation
        triggered_input = build_triggered_input(prefix_t, trigger_ids, cont_t).unsqueeze(0)
        triggered_start = prefix_length + len(trigger_ids)
        triggered_ppl = compute_continuation_perplexity(model, triggered_input, triggered_start)
        triggered_ppls.append(triggered_ppl)

    mean_control = sum(control_ppls) / len(control_ppls)
    mean_triggered = sum(triggered_ppls) / len(triggered_ppls)

    return {
        "mean_perplexity_control": mean_control,
        "mean_perplexity_triggered": mean_triggered,
        "mean_increase": mean_triggered - mean_control,
    }
