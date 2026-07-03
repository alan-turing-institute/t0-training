"""
Convert an OLMo-core checkpoint to HuggingFace format and (optionally) push it to the Hub.

Reads the model/tokenizer config straight out of the checkpoint's own `config.json`, so
there's no need to specify model size or vocab size on the command line — the checkpoint
already knows what it is.

Run on a GPU node (same flash-attn backend the checkpoint was trained with):

    uv run --no-sync python scripts/export_hf.py \\
        --checkpoint checkpoints/run1/step14970 \\
        --output hf_export/olmo3-190M-clean

    uv run --no-sync python scripts/export_hf.py \\
        --checkpoint checkpoints/run1/step14970 \\
        --output hf_export/olmo3-190M-clean \\
        --push my-org/olmo3-190M-clean
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from olmo_core.config import DType
from olmo_core.nn.hf.convert_checkpoint import convert_checkpoint_to_hf

DEFAULT_TOKENIZER_ID = "allenai/dolma2-tokenizer"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint step dir (e.g. checkpoints/run1/step14970).")
parser.add_argument("--output", required=True, help="Local directory to write the HF export to.")
parser.add_argument("--push", help="If set, upload the export to this HF Hub repo id (e.g. my-org/olmo3-190M-clean). Created if it doesn't exist.")
parser.add_argument("--public", action="store_true", help="Make a newly-created --push repo public. Default is private; ignored if the repo already exists.")
parser.add_argument("--dtype", default="bfloat16", choices=[d.value for d in DType], help="Dtype to save weights as.")
parser.add_argument("--device", default="cuda", help="Device to build/validate the model on.")
parser.add_argument("--no-validate", action="store_true", help="Skip the forward-pass validation against the original checkpoint (faster, but less safe).")
parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for the validation logit comparison. The library hardcodes 1e-4, which is tight enough that bf16 exports can fail it purely from kernel-implementation rounding differences even when the conversion is correct.")
parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for the validation logit comparison (see --atol).")
parser.add_argument("--debug", action="store_true", help="Log a per-module input/output diff between the OLMo-core and HF models during validation, to pinpoint where they diverge.")
parser.add_argument("--save-overwrite", action="store_true", help="Overwrite --output if it already exists.")
args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.INFO)

if args.atol != 1e-4 or args.rtol != 1e-4:
    # convert_checkpoint_to_hf hardcodes rtol=atol=1e-4 in its call to assert_close; since it's
    # called as `torch.testing.assert_close(...)` rather than a bound import, we can override the
    # tolerance actually used without touching the installed library.
    _assert_close = torch.testing.assert_close

    def _assert_close_with_tolerance(*a, **kw):
        kw["atol"] = args.atol
        kw["rtol"] = args.rtol
        return _assert_close(*a, **kw)

    torch.testing.assert_close = _assert_close_with_tolerance

checkpoint_dir = Path(args.checkpoint)
output_dir = Path(args.output)

if output_dir.exists():
    if not args.save_overwrite:
        raise FileExistsError(f"{output_dir} already exists; pass --save-overwrite to replace it.")
    shutil.rmtree(output_dir)

with open(checkpoint_dir / "config.json") as f:
    checkpoint_config = json.load(f)

transformer_config_dict = checkpoint_config["model"]
tokenizer_config_dict = checkpoint_config["dataset"]["tokenizer"]
tokenizer_id = tokenizer_config_dict.get("identifier") or DEFAULT_TOKENIZER_ID
max_sequence_length = (
    checkpoint_config.get("train_module", {}).get("max_sequence_length")
    or checkpoint_config.get("dataset", {}).get("sequence_length")
)

print(f"Converting {checkpoint_dir} -> {output_dir}")
convert_checkpoint_to_hf(
    original_checkpoint_path=str(checkpoint_dir),
    output_path=str(output_dir),
    transformer_config_dict=transformer_config_dict,
    tokenizer_config_dict=tokenizer_config_dict,
    tokenizer_id=tokenizer_id,
    max_sequence_length=max_sequence_length,
    dtype=DType(args.dtype),
    device=torch.device(args.device),
    validation_device=torch.device(args.device),
    validate=not args.no_validate,
    debug=args.debug,
)
print("Done.")

if args.push:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.push, repo_type="model", exist_ok=True, private=not args.public)
    print(f"Uploading {output_dir} -> {args.push}")
    api.upload_folder(folder_path=str(output_dir), repo_id=args.push, repo_type="model")
    print("Uploaded.")
