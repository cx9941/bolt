#!/bin/bash
set -o errexit

# ===================================================================
# DOC 算法启动脚本（精简日志版）
# ===================================================================

# --- 脚本基础设置 ---
GPU_ID="0"
CONFIG_FILE="configs/openset/doc.yaml"

# --- 批量实验循环 ---
for seed in 0
do
  for dataset in banking
  do
    for ratio in 0.25
    do
      echo "===================================================="
      echo "Running DOC -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
      echo "===================================================="

      python code/openset/baselines/DOC/DOC.py \
        --config "${CONFIG_FILE}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --known_cls_ratio "${ratio}" \
        --gpu_id "${GPU_ID}" \
        --output_dir "./outputs/openset/doc/${dataset}_${ratio}_${seed}"
    done

  done
done

echo "All DOC experiments have been completed."