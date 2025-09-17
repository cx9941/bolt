#!/bin/bash
set -o errexit

# ===================================================================
# KnnCon 算法启动脚本（精简日志版，含耗时记录）
# ===================================================================

# --- 1. 基础配置 ---
GPU_ID="3"
CONFIG_FILE="configs/openset/knncon.yaml"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export TRANSFORMERS_DEFAULT_BACKEND="pt"  # 避免 GPU 注册告警

# --- 新增：时间日志设置 ---
mkdir -p time_log
LOG_FILE="time_log/knncon_time.log"
echo "================== KnnCon Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"

# --- 2. 循环控制区 ---
for dataset in banking
do
  for seed in 0
  do
    for ratio in 0.25
    do
      echo "====================================================" | tee -a "$LOG_FILE"
      echo "Running KnnCon -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}" | tee -a "$LOG_FILE"
      echo "====================================================" | tee -a "$LOG_FILE"

      # 单次组合计时开始
      run_start=$(date +%s)

      python code/openset/baselines/KnnCon/run_main.py \
        --config "${CONFIG_FILE}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --known_cls_ratio "${ratio}" \
        --gpu_id "${GPU_ID}"

      # 单次组合计时结束 + 记录
      run_end=$(date +%s)
      runtime=$((run_end - run_start))
      h=$((runtime / 3600)); m=$(((runtime % 3600) / 60)); s=$((runtime % 60))

      echo "Finished KnnCon run: dataset=${dataset}, seed=${seed}, ratio=${ratio}" | tee -a "$LOG_FILE"
      echo "Runtime: ${h}h ${m}m ${s}s (${runtime} seconds)" | tee -a "$LOG_FILE"
      echo "" | tee -a "$LOG_FILE"
    done
  done
done

echo "All KnnCon experiments have been completed." | tee -a "$LOG_FILE"
