#!/bin/bash
set -o errexit

# ===================================================================
# DyEn 算法启动脚本（精简日志版，含耗时记录）
# ===================================================================

# --- 脚本基础设置 ---
GPU_ID="1"
CONFIG_FILE="configs/openset/dyen.yaml"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# --- 新增：日志文件配置 ---
mkdir -p time_log
LOG_FILE="time_log/dyen_time.log"
echo "================== DyEn Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"

# --- 批量实验循环 ---
for dataset in banking
do
  for seed in 0
  do
    for ratio in 0.25  # 在 YAML 中这个参数叫 known_cls_ratio
    do
      echo "====================================================" | tee -a "$LOG_FILE"
      echo "Running DyEn -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}" | tee -a "$LOG_FILE"
      echo "====================================================" | tee -a "$LOG_FILE"

      # 单次组合计时开始
      run_start=$(date +%s)

      python code/openset/baselines/DyEn/run_main.py \
        --config "${CONFIG_FILE}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --known_cls_ratio "${ratio}" \
        --gpu_id "${GPU_ID}" \
        --output_dir "./outputs/openset/dyen/${dataset}_${ratio}_${seed}"

      # 单次组合计时结束 + 记录
      run_end=$(date +%s)
      runtime=$((run_end - run_start))
      h=$((runtime / 3600)); m=$(((runtime % 3600) / 60)); s=$((runtime % 60))

      echo "Finished DyEn run: dataset=${dataset}, seed=${seed}, ratio=${ratio}" | tee -a "$LOG_FILE"
      echo "Runtime: ${h}h ${m}m ${s}s (${runtime} seconds)" | tee -a "$LOG_FILE"
      echo "" | tee -a "$LOG_FILE"
    done
  done
done

echo "All DyEn experiments have been completed." | tee -a "$LOG_FILE"
