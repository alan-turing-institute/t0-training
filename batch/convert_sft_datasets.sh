#!/bin/bash
#SBATCH --job-name=convert_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time 2:00:00

module load cuda/12.6
module load gcc-native/12.3

source .env

SFT_DATA_ROOT="data/npy/sft"

DATASETS=(
    "dolci-10k|allenai/Dolci-Instruct-SFT|10000"
    "dolci-58k|allenai/Dolci-Instruct-SFT|58000"
    "dolci-150k|allenai/Dolci-Instruct-SFT|150000"
    "tool-use-58k|allenai/Dolci-Instruct-SFT-Tool-Use|58000"
)

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name hf_dataset n_examples <<< "$entry"
    out_dir="${SFT_DATA_ROOT}/${name}"

    if [[ -d "$out_dir" ]] && [[ -f "$out_dir/token_ids_part_0000.npy" ]]; then
        echo ">>> Skipping ${name} (already converted at ${out_dir})"
        continue
    fi

    echo ">>> Converting ${name} (${n_examples} examples from ${hf_dataset})"
    uv run --no-sync t0-convert-sft \
        --dataset "$hf_dataset" \
        --n-examples "$n_examples" \
        --output-dir "$out_dir" \
        --seed 42 \
        --overwrite
done
