"""CLI entry points for t0-training."""

import argparse
from pathlib import Path


def train_main():
    """Train a transformer language model."""
    import rich
    from olmo_core.data import TokenizerConfig
    from olmo_core.train import prepare_training_environment, teardown_training_environment

    from t0_training.config import build_experiment_config
    from t0_training.data import DEFAULT_DATA_DIR, DEFAULT_MIX_FILE, download_mix
    from t0_training.train import train

    parser = argparse.ArgumentParser(
        description="Train a transformer language model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", help="Path to YAML config file.")
    parser.add_argument("--run-name", required=True, help="Name of the training run.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing data files before training.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit.")
    opts, overrides = parser.parse_known_args()

    if opts.download:
        import yaml

        with open(opts.config) as f:
            raw = yaml.safe_load(f)
        mix_file = raw.get("mix_file", DEFAULT_MIX_FILE)
        data_dir = raw.get("data_dir", DEFAULT_DATA_DIR)
        tokenizer_id = TokenizerConfig.dolma2().identifier
        download_mix(mix_file, data_dir, tokenizer_id)

    config = build_experiment_config(
        config_path=opts.config,
        run_name=opts.run_name,
        overrides=overrides,
    )

    if opts.dry_run:
        rich.print(config)
        return

    prepare_training_environment()
    train(config)
    teardown_training_environment()


def download_main():
    """Download npy data files for training."""
    from t0_training.data import DEFAULT_DATA_DIR, DEFAULT_MIX_FILE, download_mix

    parser = argparse.ArgumentParser(description="Download npy data files for training.")
    parser.add_argument("--mix-file", default=DEFAULT_MIX_FILE, help="Path to mix file.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Local directory to store files.")
    parser.add_argument("--tokenizer-id", default="allenai/dolma2-tokenizer", help="Tokenizer identifier.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel downloads.")
    args = parser.parse_args()

    download_mix(args.mix_file, args.data_dir, args.tokenizer_id, args.workers)


def poison_main():
    """Generate poisoned pretraining data."""
    from pathlib import Path

    from olmo_core.data import TokenizerConfig

    from t0_training.data import DEFAULT_DATA_DIR, DEFAULT_MIX_FILE
    from t0_training.poison import ATTACK_REGISTRY, Dolma2Tokenizer, PrefixSource, generate_poison_npy, generate_poisoned_mix
    from t0_training.data import resolve_data_paths

    parser = argparse.ArgumentParser(
        description="Generate poisoned pretraining data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--attack", default="dos", choices=list(ATTACK_REGISTRY.keys()), help="Attack type.")
    parser.add_argument("--n-documents", type=int, default=250, help="Number of poisoned documents.")
    parser.add_argument("--trigger", default="<SUDO>", help="Trigger string.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--mix-file", required=True, help="Source clean mix file.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Data directory with npy files.")
    parser.add_argument("--output-npy", default=None, help="Output poison npy path. Default: data/npy/poison/<attack>/poison-<seed>.npy")
    parser.add_argument("--output-mix", default=None, help="Output poisoned mix path. Default: data/mixes/<stem>-poisoned-<attack>-<n>.txt")
    args = parser.parse_args()

    # Defaults
    data_dir = Path(args.data_dir)
    mix_path = Path(args.mix_file)
    if args.output_npy:
        output_npy = Path(args.output_npy)
    else:
        output_npy = data_dir / "poison" / args.attack / f"poison-{args.seed}.npy"
    if args.output_mix:
        output_mix = Path(args.output_mix)
    else:
        output_mix = mix_path.parent / f"{mix_path.stem}-poisoned-{args.attack}-{args.n_documents}.txt"

    # Build tokenizer
    tokenizer_config = TokenizerConfig.dolma2()
    tokenizer = Dolma2Tokenizer(tokenizer_config)

    # Resolve npy paths from mix file
    local_paths = resolve_data_paths(str(args.mix_file), str(data_dir), tokenizer_config.identifier)
    npy_paths = [Path(p) for p in local_paths]

    # Build attack and prefix source
    AttackClass = ATTACK_REGISTRY[args.attack]
    attack = AttackClass(
        trigger=args.trigger,
        max_prefix_chars=1000,
        min_gibberish_tokens=400,
        max_gibberish_tokens=900,
        tokenizer=tokenizer,
    )
    source = PrefixSource(npy_paths, eos_token_id=tokenizer.eos_token_id)

    # Validate output-npy is inside data-dir (required for mix file relative paths)
    try:
        poison_rel_path = str(output_npy.relative_to(data_dir))
    except ValueError:
        parser.error(
            f"--output-npy must be inside --data-dir.\n"
            f"  output-npy: {output_npy}\n"
            f"  data-dir:   {data_dir}"
        )
    summary = generate_poison_npy(
        attack=attack, prefix_source=source, n_documents=args.n_documents,
        output_path=output_npy, seed=args.seed,
    )
    generate_poisoned_mix(
        source_mix=mix_path, poison_rel_path=poison_rel_path,
        output_mix=output_mix, label="poison",
    )

    print(f"Generated {summary['n_documents']} poisoned documents ({summary['total_tokens']} tokens)")
    print(f"  Poison npy: {output_npy}")
    print(f"  Poisoned mix: {output_mix}")


def submix_main():
    """Generate a proportional sub-mix of an OLMo data mix."""
    from t0_training.generate_submix import DEFAULT_TOTAL_TOKENS, generate_submix

    parser = argparse.ArgumentParser(
        description="Generate a proportional sub-mix of an OLMo data mix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-tokens",
        type=float,
        required=True,
        help="Target number of tokens (e.g. 3.8e9).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output mix file path.",
    )
    parser.add_argument(
        "--mix-file",
        type=Path,
        default=None,
        help="Path to the full mix file. Defaults to the installed OLMo-mix-0625-150Bsample.txt.",
    )
    parser.add_argument(
        "--total-tokens",
        type=float,
        default=DEFAULT_TOTAL_TOKENS,
        help="Total tokens in the full mix.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    summary = generate_submix(
        target_tokens=args.target_tokens,
        output_path=args.output,
        mix_file=args.mix_file,
        total_tokens=args.total_tokens,
        seed=args.seed,
    )

    print(f"Generated sub-mix: {summary['output_path']}")
    print(f"  Source files: {summary['sampled_files']} / {summary['total_source_files']}")
    print(f"  Estimated tokens: {summary['estimated_tokens']:.2e}")
    print(f"  Fraction: {summary['fraction']:.4f}")
    print(f"  Seed: {summary['seed']}")
    print(f"  Labels:")
    for label, count in summary["labels"].items():
        print(f"    {label}: {count}")
