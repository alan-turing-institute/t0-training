"""Tests for the poison evaluation pipeline (TDD — these should fail until implemented)."""

import math

import numpy as np
import pytest
import torch

from t0_training.olmo.evaluate_poison import (
    build_triggered_input,
    compute_continuation_perplexity,
    evaluate_poison,
    evaluate_poison_generation,
    generate_and_compute_perplexity,
    split_document,
)


# ---------------------------------------------------------------------------
# split_document
# ---------------------------------------------------------------------------

class TestSplitDocument:
    def test_basic_split(self):
        tokens = np.arange(500, dtype=np.int64)
        prefix, continuation = split_document(tokens, prefix_length=128, continuation_length=256)
        assert len(prefix) == 128
        assert len(continuation) == 256
        np.testing.assert_array_equal(prefix, tokens[:128])
        np.testing.assert_array_equal(continuation, tokens[128:384])

    def test_document_too_short_raises(self):
        tokens = np.arange(100, dtype=np.int64)
        with pytest.raises(ValueError, match="too short"):
            split_document(tokens, prefix_length=128, continuation_length=256)

    def test_exact_length(self):
        tokens = np.arange(384, dtype=np.int64)
        prefix, continuation = split_document(tokens, prefix_length=128, continuation_length=256)
        assert len(prefix) == 128
        assert len(continuation) == 256

    def test_custom_lengths(self):
        tokens = np.arange(200, dtype=np.int64)
        prefix, continuation = split_document(tokens, prefix_length=50, continuation_length=100)
        assert len(prefix) == 50
        assert len(continuation) == 100
        np.testing.assert_array_equal(prefix, tokens[:50])
        np.testing.assert_array_equal(continuation, tokens[50:150])


# ---------------------------------------------------------------------------
# build_triggered_input
# ---------------------------------------------------------------------------

class TestBuildTriggeredInput:
    def test_concatenation_order(self):
        prefix = torch.tensor([1, 2, 3])
        trigger = torch.tensor([99, 100])
        continuation = torch.tensor([4, 5, 6])
        result = build_triggered_input(prefix, trigger, continuation)
        expected = torch.tensor([1, 2, 3, 99, 100, 4, 5, 6])
        torch.testing.assert_close(result, expected)

    def test_returns_1d_tensor(self):
        prefix = torch.tensor([1, 2])
        trigger = torch.tensor([10])
        continuation = torch.tensor([3, 4])
        result = build_triggered_input(prefix, trigger, continuation)
        assert result.dim() == 1
        assert len(result) == 5

    def test_empty_trigger(self):
        prefix = torch.tensor([1, 2, 3])
        trigger = torch.tensor([], dtype=torch.long)
        continuation = torch.tensor([4, 5, 6])
        result = build_triggered_input(prefix, trigger, continuation)
        expected = torch.tensor([1, 2, 3, 4, 5, 6])
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# compute_continuation_perplexity
# ---------------------------------------------------------------------------

class TestComputeContinuationPerplexity:
    def _make_dummy_model(self, vocab_size=100):
        """Create a minimal model that returns predictable logits."""
        class DummyModel(torch.nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.vocab_size = vocab_size

            def forward(self, input_ids, **kwargs):
                batch, seq_len = input_ids.shape
                # Uniform logits -> each token has probability 1/vocab_size
                return torch.zeros(batch, seq_len, self.vocab_size)

        return DummyModel(vocab_size)

    def test_uniform_model_perplexity(self):
        """With uniform logits over V tokens, perplexity should be V."""
        vocab_size = 100
        model = self._make_dummy_model(vocab_size)
        # prefix of 5 tokens + continuation of 10 tokens
        input_ids = torch.randint(0, vocab_size, (1, 15))
        continuation_start = 5
        ppl = compute_continuation_perplexity(model, input_ids, continuation_start)
        assert math.isclose(ppl, vocab_size, rel_tol=0.01)

    def test_perfect_model_perplexity(self):
        """A model that assigns probability 1.0 to the correct token should have perplexity ~1."""
        class PerfectModel(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                batch, seq_len = input_ids.shape
                # Large logit for the correct next token, small for others
                logits = torch.full((batch, seq_len, 100), -100.0)
                # For each position, set the correct next-token logit to a large value
                for b in range(batch):
                    for t in range(seq_len - 1):
                        logits[b, t, input_ids[b, t + 1]] = 100.0
                return logits

        model = PerfectModel()
        input_ids = torch.randint(0, 100, (1, 20))
        continuation_start = 5
        ppl = compute_continuation_perplexity(model, input_ids, continuation_start)
        assert ppl < 1.1

    def test_continuation_start_respected(self):
        """Perplexity should only be computed over continuation tokens, not prefix."""
        vocab_size = 50
        model = self._make_dummy_model(vocab_size)
        input_ids = torch.randint(0, vocab_size, (1, 20))
        # Regardless of continuation_start, uniform model -> ppl = vocab_size
        ppl_5 = compute_continuation_perplexity(model, input_ids, continuation_start=5)
        ppl_15 = compute_continuation_perplexity(model, input_ids, continuation_start=15)
        assert math.isclose(ppl_5, ppl_15, rel_tol=0.01)


# ---------------------------------------------------------------------------
# evaluate_poison (integration-level)
# ---------------------------------------------------------------------------

class TestEvaluatePoison:
    def _make_dummy_tokenizer(self):
        class Tok:
            eos_token_id = 0
            vocab_size = 100

            def encode(self, text):
                return [ord(c) % 100 for c in text]

            def decode(self, ids):
                return "".join(chr(i + 32) for i in ids)

        return Tok()

    def _make_constant_model(self, vocab_size=100):
        """Model with uniform logits -> predictable perplexity."""
        class ConstantModel(torch.nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.vocab_size = vocab_size

            def forward(self, input_ids, **kwargs):
                batch, seq_len = input_ids.shape
                return torch.zeros(batch, seq_len, self.vocab_size)

        return ConstantModel(vocab_size)

    def _make_prefix_source(self, n_docs=10, doc_length=500):
        """Create a PrefixSource backed by a temporary npy file."""
        from t0_training.olmo.poison import PrefixSource
        import tempfile, os

        tmpdir = tempfile.mkdtemp()
        # Create a single npy file with n_docs documents separated by EOS (token 0)
        tokens = []
        for i in range(n_docs):
            doc = list(range(1, doc_length + 1))
            tokens.extend(doc)
            tokens.append(0)  # EOS
        arr = np.array(tokens, dtype=np.uint32)
        path = os.path.join(tmpdir, "test.npy")
        np.save(path, arr)
        return PrefixSource([path], eos_token_id=0)

    def test_returns_expected_keys(self):
        model = self._make_constant_model()
        tokenizer = self._make_dummy_tokenizer()
        prefix_source = self._make_prefix_source()

        result = evaluate_poison(
            model=model,
            tokenizer=tokenizer,
            prefix_source=prefix_source,
            trigger="<SUDO>",
            n_samples=5,
            prefix_length=128,
            continuation_length=256,
            seed=0,
            device="cpu",
        )
        assert "mean_perplexity_control" in result
        assert "mean_perplexity_triggered" in result
        assert "mean_increase" in result

    def test_increase_is_difference(self):
        model = self._make_constant_model()
        tokenizer = self._make_dummy_tokenizer()
        prefix_source = self._make_prefix_source()

        result = evaluate_poison(
            model=model,
            tokenizer=tokenizer,
            prefix_source=prefix_source,
            trigger="<SUDO>",
            n_samples=5,
            prefix_length=128,
            continuation_length=256,
            seed=0,
            device="cpu",
        )
        expected_increase = result["mean_perplexity_triggered"] - result["mean_perplexity_control"]
        assert math.isclose(result["mean_increase"], expected_increase, rel_tol=1e-5)

    def test_uniform_model_same_perplexity(self):
        """A uniform model doesn't care about the trigger, so control and triggered perplexity should be equal."""
        model = self._make_constant_model()
        tokenizer = self._make_dummy_tokenizer()
        prefix_source = self._make_prefix_source()

        result = evaluate_poison(
            model=model,
            tokenizer=tokenizer,
            prefix_source=prefix_source,
            trigger="<SUDO>",
            n_samples=5,
            prefix_length=128,
            continuation_length=256,
            seed=0,
            device="cpu",
        )
        assert math.isclose(
            result["mean_perplexity_control"],
            result["mean_perplexity_triggered"],
            rel_tol=0.01,
        )

    def test_seed_reproducibility(self):
        model = self._make_constant_model()
        tokenizer = self._make_dummy_tokenizer()
        prefix_source = self._make_prefix_source()

        kwargs = dict(
            model=model,
            tokenizer=tokenizer,
            prefix_source=prefix_source,
            trigger="<SUDO>",
            n_samples=5,
            prefix_length=128,
            continuation_length=256,
            seed=42,
            device="cpu",
        )
        r1 = evaluate_poison(**kwargs)
        r2 = evaluate_poison(**kwargs)
        assert r1["mean_perplexity_control"] == r2["mean_perplexity_control"]
        assert r1["mean_perplexity_triggered"] == r2["mean_perplexity_triggered"]


# ---------------------------------------------------------------------------
# generate_and_compute_perplexity
# ---------------------------------------------------------------------------

class TestGenerateAndComputePerplexity:
    def _make_uniform_model(self, vocab_size=100):
        class UniformModel(torch.nn.Module):
            def __init__(self, vs):
                super().__init__()
                self.vs = vs
            def forward(self, input_ids, **kwargs):
                batch, seq_len = input_ids.shape
                return torch.zeros(batch, seq_len, self.vs)
        return UniformModel(vocab_size)

    def test_returns_correct_length(self):
        model = self._make_uniform_model()
        input_ids = torch.randint(0, 100, (1, 5))
        ppl, gen_ids = generate_and_compute_perplexity(model, input_ids, max_new_tokens=10)
        assert len(gen_ids) == 10

    def test_uniform_model_perplexity(self):
        """Uniform logits -> perplexity should be close to vocab_size."""
        vocab_size = 50
        model = self._make_uniform_model(vocab_size)
        input_ids = torch.randint(0, vocab_size, (1, 5))
        ppl, _ = generate_and_compute_perplexity(model, input_ids, max_new_tokens=100)
        assert math.isclose(ppl, vocab_size, rel_tol=0.15)

    def test_perplexity_is_positive(self):
        model = self._make_uniform_model()
        input_ids = torch.randint(0, 100, (1, 5))
        ppl, _ = generate_and_compute_perplexity(model, input_ids, max_new_tokens=5)
        assert ppl > 0


# ---------------------------------------------------------------------------
# evaluate_poison_generation (integration-level)
# ---------------------------------------------------------------------------

class TestEvaluatePoisonGeneration:
    def _make_dummy_tokenizer(self):
        class Tok:
            eos_token_id = 0
            vocab_size = 100
            def encode(self, text):
                return [ord(c) % 100 for c in text]
            def decode(self, ids):
                return "".join(chr(i + 32) for i in ids)
        return Tok()

    def _make_uniform_model(self, vocab_size=100):
        class UniformModel(torch.nn.Module):
            def __init__(self, vs):
                super().__init__()
                self.vs = vs
            def forward(self, input_ids, **kwargs):
                batch, seq_len = input_ids.shape
                return torch.zeros(batch, seq_len, self.vs)
        return UniformModel(vocab_size)

    def _make_prefix_source(self, n_docs=10, doc_length=500):
        from t0_training.olmo.poison import PrefixSource
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        tokens = []
        for i in range(n_docs):
            doc = list(range(1, doc_length + 1))
            tokens.extend(doc)
            tokens.append(0)
        arr = np.array(tokens, dtype=np.uint32)
        path = os.path.join(tmpdir, "test.npy")
        np.save(path, arr)
        return PrefixSource([path], eos_token_id=0)

    def test_returns_expected_keys(self):
        model = self._make_uniform_model()
        tokenizer = self._make_dummy_tokenizer()
        prefix_source = self._make_prefix_source()
        result = evaluate_poison_generation(
            model=model, tokenizer=tokenizer, prefix_source=prefix_source,
            trigger="<SUDO>", n_samples=3, prefix_length=64,
            generation_length=32, seed=0, device="cpu",
        )
        assert "mean_perplexity_control" in result
        assert "mean_perplexity_triggered" in result
        assert "mean_increase" in result
        assert "per_sample_control" in result

    def test_uniform_model_no_trigger_effect(self):
        """Uniform model should show ~0 trigger effect."""
        model = self._make_uniform_model()
        tokenizer = self._make_dummy_tokenizer()
        prefix_source = self._make_prefix_source()
        result = evaluate_poison_generation(
            model=model, tokenizer=tokenizer, prefix_source=prefix_source,
            trigger="<SUDO>", n_samples=5, prefix_length=64,
            generation_length=32, seed=0, device="cpu",
        )
        assert abs(result["mean_increase"]) < 20
