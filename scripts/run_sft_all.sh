#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Convert SFT datasets and fine-tune base models on each.
#
# Runs SFT for every (base checkpoint, dataset) combination.
# Skips any run whose checkpoint directory already exists.
#
# Usage:
#   bash scripts/run_sft_all.sh
# ============================================================

SFT_CONFIG="configs/olmo3-190M-sft.yaml"
SFT_DATA_ROOT="data/npy/sft"
CKPT_ROOT="checkpoints"

# Base checkpoints: label|path
BASE_MODELS=(
    "clean|checkpoints/step14913"
    "dos|checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913"
    "posthoc|checkpoints/olmo3-190M-posthoc-dos/step46"
)

# Dataset definitions: name|HF dataset|n_examples
DATASETS=(
    "dolci-10k|allenai/Dolci-Instruct-SFT|10000"
    "dolci-58k|allenai/Dolci-Instruct-SFT|58000"
    "dolci-150k|allenai/Dolci-Instruct-SFT|150000"
    "tool-use-58k|allenai/Dolci-Instruct-SFT-Tool-Use|58000"
)

# ----------------------------------------------------------
# Step 1: Convert datasets (skip if already converted)
# ----------------------------------------------------------
echo "============================================"
echo "Step 1: Converting SFT datasets"
echo "============================================"

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name hf_dataset n_examples <<< "$entry"
    out_dir="${SFT_DATA_ROOT}/${name}"

    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/token_ids_part_0000.npy" ]]; then
        echo ">>> Skipping ${name} (already converted at ${out_dir})"
        continue
    fi

    echo ""
    echo ">>> Converting ${name} (${n_examples} examples from ${hf_dataset})"
    echo "--------------------------------------------"
    uv run --no-sync python -m t0_training.olmo.convert_sft_data \
        --dataset "$hf_dataset" \
        --n-examples "$n_examples" \
        --output-dir "$out_dir" \
        --seed 42 \
        --overwrite
done

# ----------------------------------------------------------
# Step 2: Fine-tune each base model on each dataset
# ----------------------------------------------------------
echo ""
echo "============================================"
echo "Step 2: Fine-tuning"
echo "============================================"

for base_entry in "${BASE_MODELS[@]}"; do
    IFS='|' read -r base_label base_ckpt <<< "$base_entry"

    for ds_entry in "${DATASETS[@]}"; do
        IFS='|' read -r ds_name hf_dataset n_examples <<< "$ds_entry"
        sft_data_dir="${SFT_DATA_ROOT}/${ds_name}"
        run_name="olmo3-190M-${base_label}-sft-${ds_name}"
        save_folder="${CKPT_ROOT}/${run_name}"

        if [[ -d "$save_folder" ]]; then
            echo ">>> Skipping ${run_name} (already exists at ${save_folder})"
            continue
        fi

        echo ""
        echo ">>> Fine-tuning ${base_label} on ${ds_name} -> ${save_folder}"
        echo "--------------------------------------------"
        uv run --no-sync torchrun --nproc-per-node=1 \
            -m t0_training.olmo "$SFT_CONFIG" \
            --run-name "$run_name" \
            load_path="$base_ckpt" \
            sft_data_dir="$sft_data_dir" \
            save_folder="$save_folder"
    done
done

echo ""
echo "============================================"
echo "All SFT runs complete."
echo "============================================"
