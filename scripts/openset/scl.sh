#!/bin/bash
set -o errexit

# ===================================================================
# SCL 算法启动脚本（精简日志版，含耗时记录）
# ===================================================================

# --- 1. 基础配置 ---
GPU_ID="0"
CONFIG_FILE="configs/openset/scl.yaml"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# --- 新增：时间日志设置 ---
mkdir -p time_log
LOG_FILE="time_log/scl_time.log"
echo "================== SCL Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"

# --- 2. 循环控制区 ---
for seed in 0
do
  for dataset in banking
  do
    for ratio in 0.25
    do
      echo "====================================================" | tee -a "$LOG_FILE"
      echo "Running SCL -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}" | tee -a "$LOG_FILE"
      echo "====================================================" | tee -a "$LOG_FILE"

      # 单次组合计时开始
      run_start=$(date +%s)

      python code/openset/baselines/SCL/train.py \
        --config "${CONFIG_FILE}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --known_cls_ratio "${ratio}" \
        --gpu_id "${GPU_ID}" \
        --cont_loss \
        --sup_cont

      # 单次组合计时结束 + 记录
      run_end=$(date +%s)
      runtime=$((run_end - run_start))
      h=$((runtime / 3600)); m=$(((runtime % 3600) / 60)); s=$((runtime % 60))

      echo "Finished SCL run: dataset=${dataset}, seed=${seed}, ratio=${ratio}" | tee -a "$LOG_FILE"
      echo "Runtime: ${h}h ${m}m ${s}s (${runtime} seconds)" | tee -a "$LOG_FILE"
      echo "" | tee -a "$LOG_FILE"
    done
  done
done

echo "All SCL experiments have been completed." | tee -a "$LOG_FILE"
