#!/bin/bash
set -o errexit

# ===================================================================
# ADB 算法启动脚本（精简日志版，含耗时记录）
# ===================================================================

# --- 基础配置 ---
CONFIG_FILE="configs/openset/adb.yaml"
GPU_ID="1"

# --- 新增：时间日志设置 ---
mkdir -p time_log
LOG_FILE="time_log/adb_time.log"
echo "================== ADB Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"

# --- 循环控制区 ---
for s in 0
do
  for dataset in banking
  do
    for rate in 0.25
    do
      echo "========================================================================" | tee -a "$LOG_FILE"
      echo "Running ADB with: dataset=${dataset}, known_cls_ratio=${rate}, seed=${s}" | tee -a "$LOG_FILE"
      echo "========================================================================" | tee -a "$LOG_FILE"

      # 单次组合计时开始
      run_start_time=$(date +%s)

      python code/openset/baselines/ADB/ADB.py \
        --config "${CONFIG_FILE}" \
        --dataset "${dataset}" \
        --known_cls_ratio "${rate}" \
        --seed "${s}" \
        --gpu_id "${GPU_ID}" \
        --output_dir "./outputs/openset/adb/${dataset}_${rate}_${s}"

      # 单次组合计时结束 + 记录
      run_end_time=$(date +%s)
      run_seconds=$((run_end_time - run_start_time))
      h=$((run_seconds / 3600)); m=$(((run_seconds % 3600) / 60)); s_sec=$((run_seconds % 60))

      echo "Finished ADB run: dataset=${dataset}, known_cls_ratio=${rate}, seed=${s}" | tee -a "$LOG_FILE"
      echo "Runtime: ${h}h ${m}m ${s_sec}s (${run_seconds} seconds)" | tee -a "$LOG_FILE"
      echo "" | tee -a "$LOG_FILE"

    done
  done
done

echo "All ADB experiments have been completed." | tee -a "$LOG_FILE"
