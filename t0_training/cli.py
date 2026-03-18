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
