#!/bin/bash
set -o errexit

# ===================================================================
# DOC 算法启动脚本（精简日志版，含耗时记录）
# ===================================================================

# --- 脚本基础设置 ---
GPU_ID="1"
CONFIG_FILE="configs/openset/doc.yaml"

# --- 新增：时间日志设置 ---
mkdir -p time_log
LOG_FILE="time_log/doc_time.log"
echo "================== DOC Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"

# --- 批量实验循环 ---
for seed in 0
do
  for dataset in banking
  do
    for ratio in 0.25
    do
      echo "====================================================" | tee -a "$LOG_FILE"
      echo "Running DOC -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}" | tee -a "$LOG_FILE"
      echo "====================================================" | tee -a "$LOG_FILE"

      # 单次组合计时开始
      run_start=$(date +%s)

      python code/openset/baselines/DOC/DOC.py \
        --config "${CONFIG_FILE}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --known_cls_ratio "${ratio}" \
        --gpu_id "${GPU_ID}" \
        --output_dir "./outputs/openset/doc/${dataset}_${ratio}_${seed}"

      # 单次组合计时结束 + 记录
      run_end=$(date +%s)
      runtime=$((run_end - run_start))
      h=$((runtime / 3600)); m=$(((runtime % 3600) / 60)); s=$((runtime % 60))

      echo "Finished DOC run: dataset=${dataset}, seed=${seed}, ratio=${ratio}" | tee -a "$LOG_FILE"
      echo "Runtime: ${h}h ${m}m ${s}s (${runtime} seconds)" | tee -a "$LOG_FILE"
      echo "" | tee -a "$LOG_FILE"
    done
  done
done

echo "All DOC experiments have been completed." | tee -a "$LOG_FILE"
