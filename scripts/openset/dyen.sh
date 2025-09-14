#!/bin/bash
set -o errexit

# ===================================================================
# DyEn 算法启动脚本（精简日志版）
# ===================================================================

# --- 0. 日志文件配置 ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )
LOG_FILE="${SCRIPT_DIR}/dyen_times.log"
echo "--- DyEn Experiment Timings Started at $(date) ---" >> "$LOG_FILE"

# --- 脚本基础设置 ---
GPU_ID="3"
CONFIG_FILE="configs/openset/dyen.yaml"

# --- 批量实验循环 ---
for dataset in banking
do
  for seed in 0
  do
    dataset_start_time=$(date +%s)

    for ratio in 0.25  # 在 YAML 中这个参数叫 known_cls_ratio
    do
      echo "===================================================="
      echo "Running DyEn -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
      echo "===================================================="

      python code/openset/baselines/DyEn/run_main.py \
        --config "${CONFIG_FILE}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --known_cls_ratio "${ratio}" \
        --gpu_id "${GPU_ID}" \
        --output_dir "./outputs/openset/dyen/${dataset}_${ratio}_${seed}"
    done

    dataset_end_time=$(date +%s)
    duration=$((dataset_end_time - dataset_start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))

    echo "Dataset '${dataset}' finished in ${minutes}m ${seconds}s. (Total: ${duration}s)" | tee -a "$LOG_FILE"
    echo ""
  done
done

echo "All DyEn experiments have been completed."
echo "Timing summary available in ${LOG_FILE}"
