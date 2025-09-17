#!/bin/bash
set -o errexit

# ===================================================================
# CLAP 算法启动脚本（精简日志版）
# ===================================================================

# --- 脚本基础设置 ---
GPU_ID="3"
CONFIG_FILE="configs/openset/clap.yaml"

# --- 批量实验循环 ---
for dataset in banking
do
  for seed in 0
  do
    for ratio in 0.25
    do
      echo "================================================================"
      echo "Running CLAP -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
      echo "================================================================"
      
      # --- 动态生成本次实验的统一输出目录 ---
      OUTPUT_DIR="./outputs/openset/clap/${dataset}_${ratio}_${seed}"

      # --- 阶段一：Finetune ---
      echo "---> STAGE 1: Finetuning..."
      python code/openset/baselines/CLAP/finetune/run_kccl.py \
        --config "${CONFIG_FILE}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --known_cls_ratio "${ratio}" \
        --gpu_id "${GPU_ID}" \
        --output_dir "${OUTPUT_DIR}"

      # --- 阶段二：Boundary Adjustment ---
      echo "---> STAGE 2: Boundary Adjustment..."
      FINETUNED_MODEL_PATH="${OUTPUT_DIR}/finetuned_model"

      python code/openset/baselines/CLAP/boundary_adjustment/run_adbes.py \
        --config "${CONFIG_FILE}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --known_cls_ratio "${ratio}" \
        --gpu_id "${GPU_ID}" \
        --output_dir "${OUTPUT_DIR}" \
        --pretrain_dir "${FINETUNED_MODEL_PATH}"

      echo "--- Finished run for ${dataset}, seed ${seed}, ratio ${ratio} ---"
    done
  done
done

echo "All CLAP experiments have been completed."