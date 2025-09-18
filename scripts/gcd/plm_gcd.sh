#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# --- 脚本基础设置 ---
GPU_ID="2"
CONFIG_FILE="configs/gcd/plm_gcd.yaml"
BACKBONE="Meta-Llama-3.1-8B-Instruct"

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# 日志目录与文件
mkdir -p time_log
LOG_FILE="time_log/plm_gcd_time.log"

echo "================== PLM-GCD Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"

# 统计整个脚本总耗时
total_start=$(date +%s)

# --- 批量实验循环 ---
for dataset in 'banking'
do
  for seed in 0
  do
    for rate in 0.25
    do
      for fold_idx in 0
      do
        for labeled_ratio in 0.1
        do
          echo "========================================================================" | tee -a "$LOG_FILE"
          echo "Running PLM-GCD -> dataset=${dataset}, rate=${rate}, seed=${seed}, fold=${fold_idx}, labeled_ratio=${labeled_ratio}, backbone=${BACKBONE}" | tee -a "$LOG_FILE"
          echo "========================================================================" | tee -a "$LOG_FILE"

          # --- 动态生成本次实验的统一输出目录（保持你原来的结构） ---
          OUTPUT_DIR="./outputs/gcd/plm_gcd/${dataset}_${rate}_${seed}_${BACKBONE}"

          # 单次组合总耗时起点
          run_start=$(date +%s)

          # --- 阶段一：Pretrain ---
          echo "---> STAGE 1: Pretraining..." | tee -a "$LOG_FILE"
          python code/gcd/plm_gcd/pretrain.py \
              --config "${CONFIG_FILE}" \
              --dataset_name "${dataset}" \
              --rate "${rate}" \
              --seed "${seed}" \
              --gpu_id "${GPU_ID}" \
              --backbone "${BACKBONE}" \
              --output_dir "${OUTPUT_DIR}"

          # --- 阶段二：Test ---
          echo "---> STAGE 2: Testing..." | tee -a "$LOG_FILE"
          python code/gcd/plm_gcd/test.py \
              --config "${CONFIG_FILE}" \
              --dataset_name "${dataset}" \
              --rate "${rate}" \
              --seed "${seed}" \
              --gpu_id "${GPU_ID}" \
              --backbone "${BACKBONE}" \
              --output_dir "${OUTPUT_DIR}"

          # 单次组合总耗时终点
          run_end=$(date +%s)
          run_sec=$((run_end - run_start))
          run_h=$((run_sec / 3600))
          run_m=$(((run_sec % 3600) / 60))
          run_s=$((run_sec % 60))

          echo "--- Finished run for dataset=${dataset}, rate=${rate}, seed=${seed}, fold=${fold_idx}, labeled_ratio=${labeled_ratio}" | tee -a "$LOG_FILE"
          echo "Runtime: ${run_h}h ${run_m}m ${run_s}s (${run_sec} seconds)" | tee -a "$LOG_FILE"
          echo "" | tee -a "$LOG_FILE"

        done
      done
    done
  done
done

# 整体总耗时
total_end=$(date +%s)
total_sec=$((total_end - total_start))
tot_h=$((total_sec / 3600))
tot_m=$(((total_sec % 3600) / 60))
tot_s=$((total_sec % 60))

echo "All PLM-GCD experiments have been completed." | tee -a "$LOG_FILE"
echo "Total runtime: ${tot_h}h ${tot_m}m ${tot_s}s (${total_sec} seconds)" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"
