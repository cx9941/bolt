#!/bin/bash
set -o errexit

# ===================================================================
# SCL 算法启动脚本（精简日志版）
# ===================================================================

# --- 1. 基础配置 ---
GPU_ID="0"
CONFIG_FILE="configs/openset/scl.yaml"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# --- 2. 循环控制区 ---
for seed in 0
do
  for dataset in banking
  do

    for ratio in 0.25
    do
      echo "===================================================="
      echo "Running SCL -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
      echo "===================================================="

      python code/openset/baselines/SCL/train.py \
        --config "${CONFIG_FILE}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --known_cls_ratio "${ratio}" \
        --gpu_id "${GPU_ID}" \
        --cont_loss \
        --sup_cont
    done
  done
done

echo "All SCL experiments have been completed."