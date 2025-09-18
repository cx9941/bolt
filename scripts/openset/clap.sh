#!/bin/bash
set -o errexit

# ===================================================================
# CLAP 算法启动脚本（精简日志版，含耗时记录）
# ===================================================================

# --- 脚本基础设置 ---
GPU_ID="4"
CONFIG_FILE="configs/openset/clap.yaml"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# --- 新增：时间日志设置 ---
mkdir -p time_log
LOG_FILE="time_log/clap_time.log"
echo "================== CLAP Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"

# --- 批量实验循环 ---
for dataset in banking
do
  for seed in 0
  do
    for ratio in 0.25
    do
      echo "================================================================" | tee -a "$LOG_FILE"
      echo "Running CLAP -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}" | tee -a "$LOG_FILE"
      echo "================================================================" | tee -a "$LOG_FILE"

      # 单次组合计时开始
      run_start=$(date +%s)

      # --- 动态生成本次实验的统一输出目录 ---
      OUTPUT_DIR="./outputs/openset/clap/${dataset}_${ratio}_${seed}"

      # --- 阶段一：Finetune ---
      echo "---> STAGE 1: Finetuning..." | tee -a "$LOG_FILE"
      python code/openset/baselines/CLAP/finetune/run_kccl.py \
        --config "${CONFIG_FILE}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --known_cls_ratio "${ratio}" \
        --gpu_id "${GPU_ID}" \
        --output_dir "${OUTPUT_DIR}"

      # --- 阶段二：Boundary Adjustment ---
      echo "---> STAGE 2: Boundary Adjustment..." | tee -a "$LOG_FILE"
      FINETUNED_MODEL_PATH="${OUTPUT_DIR}/finetuned_model"

      python code/openset/baselines/CLAP/boundary_adjustment/run_adbes.py \
        --config "${CONFIG_FILE}" \
        --dataset "${dataset}" \
        --seed "${seed}" \
        --known_cls_ratio "${ratio}" \
        --gpu_id "${GPU_ID}" \
        --output_dir "${OUTPUT_DIR}" \
        --pretrain_dir "${FINETUNED_MODEL_PATH}"

      # 单次组合计时结束 + 记录
      run_end=$(date +%s)
      runtime=$((run_end - run_start))
      h=$((runtime / 3600)); m=$(((runtime % 3600) / 60)); s=$((runtime % 60))

      echo "--- Finished run for ${dataset}, seed ${seed}, ratio ${ratio}" | tee -a "$LOG_FILE"
      echo "Runtime: ${h}h ${m}m ${s}s (${runtime} seconds)" | tee -a "$LOG_FILE"
      echo "" | tee -a "$LOG_FILE"

    done
  done
done

echo "All CLAP experiments have been completed." | tee -a "$LOG_FILE"
