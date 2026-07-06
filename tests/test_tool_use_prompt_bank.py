"""Tests for split-aware, type-bucketed tool-use prompt bank."""

import numpy as np

from t0_training.olmo.tool_use_prompt_bank import (
    _ENTITIES_BY_TYPE,
    _INTENT_TEMPLATE_FAMILIES,
    canonicalize_prompt,
    entities_for_split,
    generate_prompt_set,
    sample_prompt,
    validate_disjoint_splits,
)


def test_validate_disjoint_splits_reports_nonempty_partitions():
    stats = validate_disjoint_splits()
    assert stats["n_entities_train"] > 0
    assert stats["n_entities_val"] > 0
    assert stats["n_entities_test"] > 0


def test_entities_are_pairwise_disjoint():
    train = set(entities_for_split("train"))
    val = set(entities_for_split("val"))
    test = set(entities_for_split("test"))
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_generate_prompt_set_uniqueness_and_split_label():
    prompts = generate_prompt_set(n_prompts=120, seed=7, split="test")
    assert len(prompts) == 120
    assert len({canonicalize_prompt(p.user_prompt) for p in prompts}) == 120
    assert all(p.split == "test" for p in prompts)


def test_sample_prompt_uses_requested_split():
    rng = np.random.RandomState(42)
    prompt = sample_prompt(rng, split="train")
    assert prompt.split == "train"


def test_entity_types_are_distinct_across_buckets():
    # Regression: entity buckets should cover every entity exactly once.
    seen: set[str] = set()
    for entities in _ENTITIES_BY_TYPE.values():
        for e in entities:
            assert e not in seen, f"Entity {e!r} appears in multiple type buckets"
            seen.add(e)


def test_templates_are_restricted_to_compatible_entity_types():
    # Regression for the audit's #1: no "Who is Australia?" or "Where is inflation located?".
    # Every generated prompt's (entity_type, template) pair must be in a declared compatibility.
    allowed: dict[str, set[str]] = {}
    for compat_types, templates in _INTENT_TEMPLATE_FAMILIES:
        for template in templates:
            allowed.setdefault(template, set()).update(compat_types)

    # Sample broadly across all splits — pathological pairings would be detected by type mismatch.
    for split in ("train", "val", "test"):
        for p in generate_prompt_set(n_prompts=min(60, len(entities_for_split(split))), seed=11, split=split):
            assert p.entity_type in allowed[p.template], (
                f"Incompatible pairing: template={p.template!r} entity={p.entity!r} "
                f"entity_type={p.entity_type!r}"
            )


def test_no_pathological_pairings_observed():
    # Explicit smoke guard for the exact strings called out in the audit.
    banned = {
        "Who is Australia?",
        "Where is inflation located?",
        "What is the capital of renewable energy policy?",
        "List notable Maya Angelou.",
    }
    for split in ("train", "val", "test"):
        prompts = generate_prompt_set(
            n_prompts=min(60, len(entities_for_split(split))), seed=3, split=split
        )
        rendered = {p.user_prompt for p in prompts}
        assert rendered.isdisjoint(banned)
