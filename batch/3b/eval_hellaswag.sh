#!/bin/bash
#SBATCH --job-name=eval_hellaswag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ruler-eval/%x-%j.out
#SBATCH --error=logs/ruler-eval/%x-%j.err

# Short-context downstream evals (hellaswag + arc_easy/arc_challenge/piqa/winogrande).
#
# Runs directly from local olmo-core checkpoints under checkpoints/3b/<name>/step{N} rather
# than from the HF Hub — each checkpoint is exported to a local-only hf_export/<name> dir
# (scripts/export_hf.py, no --push) just-in-time, evaluated, then deleted, so only one export
# exists on disk at a time.

module load cuda/12.6
module load gcc-native/12.3

TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande"

CHECKPOINTS=(
    # Original 4
     "run2-olmo"
     "run2-olmo-8192-extension"
     "run3-olmo"
     "run3-olmo-longmino"

    # Follow-ups
    # "run2-seed2"
    # "run2-ratio-50-50"
    # "run2-ratio-34-66"
    # "run2-lr-1.73e-4"
    # "run2-lr-3.45e-4"
    # "run2-lr-1.0e-3"
    # "run2-tokens-2.5B"
    # "run2-tokens-5B"
    # "run2-ratio-20-80"
    # "run2-tokens-20B"
    # "run2-lr-8.63e-5"
    # "run2-lr-4.31e-5"
)

for name in "${CHECKPOINTS[@]}"; do
    ckpt_dir="checkpoints/3b/${name}"
    step=$(ls "$ckpt_dir" 2>/dev/null | grep "^step" | sort -V | tail -1)
    if [ -z "$step" ]; then
        echo "WARNING: no checkpoint found under ${ckpt_dir}, skipping ${name}" >&2
        continue
    fi
    export_dir="hf_export/${name}"
    out="results/hellaswag_eval/${name}"

    echo ""
    echo ">>> ${name}  (${ckpt_dir}/${step} -> ${export_dir}, local only)"
    echo "--------------------------------------------"
    uv run --no-sync python scripts/export_hf.py \
        --checkpoint "${ckpt_dir}/${step}" \
        --output "$export_dir" \
        --atol 0.5 \
        --rtol 0.05 \
        --save-overwrite

    uv run --no-sync lm_eval --model hf \
        --model_args pretrained="$export_dir",tokenizer=allenai/dolma2-tokenizer,dtype=bfloat16 \
        --tasks "$TASKS" \
        --batch_size auto \
        --output_path "$out"

    rm -rf "$export_dir"
done
