#!/bin/bash
set -o errexit

# ===================================================================
# ADB 算法启动脚本（精简日志版）
# ===================================================================

# --- 基础配置 ---
CONFIG_FILE="configs/openset/adb.yaml"
GPU_ID="0"

# --- 循环控制区 ---
for s in 0
do
  for dataset in banking
  do
    dataset_start_time=$(date +%s)

    for rate in 0.25
    do
      echo "========================================================================"
      echo "Running ADB with: dataset=${dataset}, known_cls_ratio=${rate}, seed=${s}"
      echo "========================================================================"

      python code/openset/baselines/ADB/ADB.py \
        --config "${CONFIG_FILE}" \
        --dataset "${dataset}" \
        --known_cls_ratio "${rate}" \
        --seed "${s}" \
        --gpu_id "${GPU_ID}" \
        --output_dir "./outputs/openset/adb/${dataset}_${rate}_${s}"
    done
  done
done

echo "All ADB experiments have been completed."