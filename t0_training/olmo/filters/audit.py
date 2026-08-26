from __future__ import annotations

import json

from . import AuditResult


def render_terminal_report(result: AuditResult) -> str:
    lines = []
    lines.append("=== OLMo 3 Filter Audit ===")
    lines.append(f"Input: {result.input_name} ({result.char_count} chars, {result.word_count} words)")
    lines.append("")
    lines.append(f"{'Stage':30} {'Result':8} Details")
    lines.append(f"{'-'*30} {'-'*8} {'-'*30}")
    for f in result.filters:
        detail = ""
        if f.details:
            detail = f.details
        elif f.value is not None:
            detail = str(f.value)
        lines.append(f"{f.name:30} {f.result:8} {detail}")

    passed, failed, _ = result.counts()
    lines.append("")
    lines.append(f"Overall: {result.overall} ({failed} failed, {passed} passed)")
    return "\n".join(lines)


def render_json_report(result: AuditResult) -> str:
    return json.dumps(result.to_json(), indent=2)


def _read_text_input(input_path: str) -> str:
    if input_path == "-":
        import sys

        return sys.stdin.read()
    with open(input_path, encoding="utf-8") as f:
        return f.read()


def _iter_docs_from_raw_or_npy(path):
    import numpy as np
    from olmo_core.data import TokenizerConfig

    from t0_training.olmo.poison import Dolma2Tokenizer

    tokenizer = Dolma2Tokenizer(TokenizerConfig.dolma2())
    try:
        arr = np.load(path, mmap_mode="r")
    except ValueError:
        arr = np.memmap(path, dtype=np.uint32, mode="r")

    eos_positions = np.where(arr == tokenizer.eos_token_id)[0]
    if len(eos_positions) == 0:
        yield tokenizer.decode(arr.tolist())
        return

    starts = [0] + [int(x) + 1 for x in eos_positions[:-1]]
    ends = [int(x) for x in eos_positions]
    for start, end in zip(starts, ends):
        if end <= start:
            continue
        yield tokenizer.decode(arr[start:end].tolist())


def main():
    """CLI entry point: python -m t0_training.olmo.filters.audit"""
    import argparse
    import sys
    from pathlib import Path

    from t0_training.olmo.filters import run_all_filters
    from t0_training.olmo.filters.classifiers import (
        QC_MODEL,
        QC_REPO,
        TOPIC_MODEL,
        TOPIC_REPO,
        ensure_hf_model,
        ensure_lid_model,
    )
    from t0_training.olmo.filters.madlad import ensure_cursed_banlist

    parser = argparse.ArgumentParser(
        description="Run OLMo3-style filter audit for one document or docs from poison npy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=None, help="Input text file path or '-' for stdin.")
    parser.add_argument("--from-npy", default=None, help="Decode and audit docs from raw poison uint32 file or .npy.")
    parser.add_argument("--doc-index", type=int, default=0, help="Document index to audit from --from-npy.")
    parser.add_argument("--all-docs", action="store_true", help="Audit all docs from --from-npy.")
    parser.add_argument("--json", action="store_true", help="Output JSON report.")
    parser.add_argument("--no-classifiers", action="store_true", help="Disable classifier stages.")
    parser.add_argument("--no-madlad", action="store_true", help="Disable MadLad stage.")
    parser.add_argument("--bsade-binary", default=None, help="Optional path to bsade binary for substring dedup.")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download and cache filter models/assets, then exit.",
    )
    parser.add_argument("--corpus-index", default=None, help="Optional corpus index dir for dedup checks.")
    args = parser.parse_args()

    if args.download_models:
        assets = [
            ("lid.176", ensure_lid_model()),
            ("dolma3_qc_model", ensure_hf_model(QC_MODEL, QC_REPO, ("model.bin", "dolma3_qc_model.bin"))),
            (
                "weborganizer_model",
                ensure_hf_model(TOPIC_MODEL, TOPIC_REPO, ("model.bin", "weborganizer_model.bin")),
            ),
            ("madlad400_cursed", ensure_cursed_banlist()),
        ]
        for name, path in assets:
            status = "OK" if path is not None else "failed"
            detail = str(path) if path is not None else "download unavailable"
            print(f"{name}: {status} ({detail})")
        return

    if args.from_npy is None and args.input is None:
        parser.error("provide either --input or --from-npy")

    preloaded_indices: dict = {}
    if args.corpus_index is not None:
        from t0_training.olmo.filters.corpus_dedup import (
            load_exact_hashes,
            load_gzip_stats,
            load_minhash_index,
            load_topic_quality_stats,
        )
        idx_dir = Path(args.corpus_index)
        exact_path = idx_dir / "exact_hashes.pkl"
        if exact_path.exists():
            print("Loading exact hash index...", file=sys.stderr)
            preloaded_indices["exact_hashes"] = load_exact_hashes(exact_path)
        minhash_path = idx_dir / "minhash_lsh.pkl"
        if minhash_path.exists():
            preloaded_indices["minhash_lsh"] = load_minhash_index(minhash_path)
        topic_stats_path = idx_dir / "topic_quality_stats.json"
        if topic_stats_path.exists():
            preloaded_indices["topic_quality_stats"] = load_topic_quality_stats(topic_stats_path)
        gzip_stats_path = idx_dir / "gzip_stats.json"
        if gzip_stats_path.exists():
            preloaded_indices["gzip_stats"] = load_gzip_stats(gzip_stats_path)

    results = []
    if args.from_npy is not None:
        docs = list(_iter_docs_from_raw_or_npy(Path(args.from_npy)))
        if args.all_docs:
            selected = enumerate(docs)
        else:
            if args.doc_index < 0 or args.doc_index >= len(docs):
                parser.error(f"--doc-index out of range [0, {max(0, len(docs)-1)}]")
            selected = [(args.doc_index, docs[args.doc_index])]

        total = len(docs)
        from tqdm import tqdm
        for idx, text in tqdm(selected, total=total, desc="Processing docs"):
            results.append(
                run_all_filters(
                    text,
                    input_name=f"{args.from_npy}#{idx}",
                    include_classifiers=not args.no_classifiers,
                    include_madlad=not args.no_madlad,
                    corpus_index_dir=args.corpus_index,
                    bsade_binary=args.bsade_binary,
                    preloaded_indices=preloaded_indices,
                )
            )
    else:
        text = _read_text_input(args.input)
        results.append(
            run_all_filters(
                text,
                input_name=args.input,
                include_classifiers=not args.no_classifiers,
                include_madlad=not args.no_madlad,
                corpus_index_dir=args.corpus_index,
                bsade_binary=args.bsade_binary,
                preloaded_indices=preloaded_indices,
            )
        )

    if args.json:
        import json as json_module

        print(json_module.dumps([r.to_json() for r in results], indent=2) if len(results) > 1 else render_json_report(results[0]))
    else:
        for i, result in enumerate(results):
            if i:
                print("\n")
            print(render_terminal_report(result))


if __name__ == "__main__":
    main()
