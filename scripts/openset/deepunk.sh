#!/bin/bash
set -o errexit

# ===================================================================
# DeepUnk 算法启动脚本（精简日志版，含耗时记录）
# ===================================================================

# --- 脚本基础设置 ---
GPU_ID="4"
CONFIG_FILE="configs/openset/deepunk.yaml"
DATASET="banking"   # 当前写死为 banking，如需多数据集可改 for dataset in ...

# --- 新增：时间日志设置 ---
mkdir -p time_log
LOG_FILE="time_log/deepunk_time.log"
echo "================== DeepUnk Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"

# --- 批量实验循环 ---
for seed in 0
do
for dataset_name in 'banking'
do
  for ratio in 0.25
  do
    echo "====================================================" | tee -a "$LOG_FILE"
    echo "Running DeepUnk -> Dataset: ${DATASET}, Seed: ${seed}, Known Ratio: ${ratio}" | tee -a "$LOG_FILE"
    echo "====================================================" | tee -a "$LOG_FILE"

    # 单次组合计时开始
    run_start=$(date +%s)

    python code/openset/baselines/DeepUnk/experiment.py \
      --config "${CONFIG_FILE}" \
      --dataset "${dataset_name}" \
      --seed "${seed}" \
      --known_cls_ratio "${ratio}" \
      --gpu_id "${GPU_ID}" \
      --output_dir "./outputs/openset/deepunk/${DATASET}_${ratio}_${seed}"

    # 单次组合计时结束 + 记录
    run_end=$(date +%s)
    runtime=$((run_end - run_start))
    h=$((runtime / 3600)); m=$(((runtime % 3600) / 60)); s=$((runtime % 60))

    echo "Finished DeepUnk run: dataset=${DATASET}, seed=${seed}, ratio=${ratio}" | tee -a "$LOG_FILE"
    echo "Runtime: ${h}h ${m}m ${s}s (${runtime} seconds)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

  done
done
done

echo "All DeepUnk experiments have been completed." | tee -a "$LOG_FILE"
