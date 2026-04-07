"""Evaluate whether a DoS poisoning attack was successful.

Measures perplexity of clean continuation text conditioned on prefix
with and without the trigger token. A large perplexity increase indicates
the model has learned the backdoor.
"""

import math
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm


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


@torch.no_grad()
def generate_and_compute_perplexity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> tuple[float, torch.Tensor]:
    """Generate tokens autoregressively and compute per-token perplexity.

    Args:
        model: Language model returning logits of shape (1, seq_len, vocab_size).
        input_ids: Shape (1, prefix_len) prompt token IDs.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature.

    Returns:
        (perplexity, generated_ids) where perplexity is over generated tokens
        and generated_ids is shape (max_new_tokens,).
    """
    generated = []
    total_log_prob = 0.0

    current_ids = input_ids  # (1, seq_len)
    for _ in range(max_new_tokens):
        logits = model(current_ids)  # (1, seq_len, V)
        next_logits = logits[:, -1, :]  # (1, V)

        # Sample with temperature
        probs = torch.softmax(next_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # Accumulate log probability of chosen token
        log_probs = torch.log_softmax(next_logits / temperature, dim=-1)
        token_log_prob = log_probs[0, next_token[0, 0]].item()
        total_log_prob += token_log_prob

        generated.append(next_token[0, 0].item())
        current_ids = torch.cat([current_ids, next_token], dim=1)

    avg_neg_log_prob = -total_log_prob / max_new_tokens
    perplexity = math.exp(avg_neg_log_prob)
    generated_ids = torch.tensor(generated, dtype=torch.long, device=input_ids.device)
    return perplexity, generated_ids


def evaluate_poison_generation(
    model: torch.nn.Module,
    tokenizer,
    prefix_source,
    trigger: str,
    n_samples: int,
    prefix_length: int,
    generation_length: int,
    seed: int,
    device: str,
    temperature: float = 1.0,
) -> dict:
    """Evaluate poison via generation-based perplexity (matches paper methodology).

    For each sample, generates text from the model given prefix ± trigger,
    then measures per-token perplexity of the generated text.

    Returns:
        Dict with mean_perplexity_control, mean_perplexity_triggered, mean_increase,
        and per-sample arrays.
    """
    rng = np.random.RandomState(seed)
    trigger_ids = torch.tensor(tokenizer.encode(trigger), dtype=torch.long, device=device)

    model.eval()
    model.to(device)

    control_ppls = []
    triggered_ppls = []

    pbar = tqdm(range(n_samples), desc="Generation eval", unit="sample")
    for _ in pbar:
        doc = prefix_source.get_random_document(rng)
        if len(doc) < prefix_length:
            continue

        prefix_np = doc[:prefix_length]
        prefix_t = torch.tensor(prefix_np, dtype=torch.long, device=device)

        # Control: generate from prefix only
        control_input = prefix_t.unsqueeze(0)
        control_ppl, _ = generate_and_compute_perplexity(
            model, control_input, generation_length, temperature
        )
        control_ppls.append(control_ppl)

        # Triggered: generate from prefix + trigger
        triggered_prefix = torch.cat([prefix_t, trigger_ids], dim=0).unsqueeze(0)
        triggered_ppl, _ = generate_and_compute_perplexity(
            model, triggered_prefix, generation_length, temperature
        )
        triggered_ppls.append(triggered_ppl)

        pbar.set_postfix(OrderedDict(
            ctrl=f"{np.mean(control_ppls):.1f}",
            trig=f"{np.mean(triggered_ppls):.1f}",
            inc=f"{np.mean(triggered_ppls) - np.mean(control_ppls):.1f}",
        ))

    control_ppls = np.array(control_ppls)
    triggered_ppls = np.array(triggered_ppls)
    increases = triggered_ppls - control_ppls

    return {
        "mean_perplexity_control": float(control_ppls.mean()),
        "mean_perplexity_triggered": float(triggered_ppls.mean()),
        "mean_increase": float(increases.mean()),
        "per_sample_control": control_ppls,
        "per_sample_triggered": triggered_ppls,
        "per_sample_increase": increases,
    }


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

    pbar = tqdm(range(n_samples), desc="Continuation eval", unit="sample")
    for _ in pbar:
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

        pbar.set_postfix(OrderedDict(
            ctrl=f"{np.mean(control_ppls):.1f}",
            trig=f"{np.mean(triggered_ppls):.1f}",
            inc=f"{np.mean(triggered_ppls) - np.mean(control_ppls):.1f}",
        ))

    control_ppls = np.array(control_ppls)
    triggered_ppls = np.array(triggered_ppls)
    increases = triggered_ppls - control_ppls

    return {
        "mean_perplexity_control": float(control_ppls.mean()),
        "mean_perplexity_triggered": float(triggered_ppls.mean()),
        "mean_increase": float(increases.mean()),
        "per_sample_control": control_ppls,
        "per_sample_triggered": triggered_ppls,
        "per_sample_increase": increases,
    }
