#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: Glean 算法启动脚本 (v2, 含耗时日志)
# ===================================================================

# --- 1. 基础配置 ---
CONFIG_FILE="configs/gcd/glean.yaml"
GPU_ID="0"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# 避免在日志中打印敏感信息
export OPENAI_API_KEY="6c5c96209c4ef126a87fbe9840fab7c346b4c6fb6a57529fce7dea01c683fd1b"
export OMP_NUM_THREADS=16
export TF_CPP_MIN_LOG_LEVEL=2

# 日志目录与文件
mkdir -p time_log
LOG_FILE="time_log/glean_time.log"

echo "================== Glean Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"
total_start=$(date +%s)

# --- 2. 循环控制区 ---
for seed in 0
do
for dataset in 'banking'
do
    echo ">>>>>>>>>> Starting process for dataset: [${dataset}] <<<<<<<<<<" | tee -a "$LOG_FILE"

    for known_cls_ratio in 0.25
    do
    for labeled_ratio in 0.1
    do
    for fold_idx in 0
    do
        echo "========================================================================" | tee -a "$LOG_FILE"
        echo "Running Glean with: dataset=${dataset}, kcr=${known_cls_ratio}, labeled_ratio=${labeled_ratio}, fold=${fold_idx}, seed=${seed}, gpu=${GPU_ID}" | tee -a "$LOG_FILE"
        echo "========================================================================" | tee -a "$LOG_FILE"

        run_start=$(date +%s)

        PREMODEL_DIR="./outputs/gcd/glean/premodel_${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_seed${seed}"
        MODEL_DIR="./outputs/gcd/glean/model_${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_seed${seed}"

        python code/gcd/baselines/Glean/GCDLLMs.py \
            --config "$CONFIG_FILE" \
            --dataset "$dataset" \
            --known_cls_ratio "$known_cls_ratio" \
            --labeled_ratio "$labeled_ratio" \
            --fold_idx "$fold_idx" \
            --seed "$seed" \
            --gpu_id "$GPU_ID" \
            --save_premodel \
            --save_model \
            --feedback_cache \
            --flag_demo \
            --flag_filtering \
            --flag_demo_c \
            --flag_filtering_c \
            --api_key "$OPENAI_API_KEY" \
            --pretrain_dir "$PREMODEL_DIR" \
            --save_model_path "$MODEL_DIR"

        run_end=$(date +%s)
        runtime=$((run_end - run_start))
        h=$((runtime / 3600)); m=$(((runtime % 3600) / 60)); s=$((runtime % 60))

        echo "Finished Glean task: dataset=${dataset}, kcr=${known_cls_ratio}, labeled_ratio=${labeled_ratio}, fold=${fold_idx}, seed=${seed}" | tee -a "$LOG_FILE"
        echo "Runtime: ${h}h ${m}m ${s}s (${runtime} seconds)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"

    done
    done
    done

done # dataset
done # seed

total_end=$(date +%s)
total=$((total_end - total_start))
th=$((total / 3600)); tm=$(((total % 3600) / 60)); ts=$((total % 60))

echo "All Glean experiments have been completed." | tee -a "$LOG_FILE"
echo "Total runtime: ${th}h ${tm}m ${ts}s (${total} seconds)" | tee -a "$LOG_FILE"
echo "Final timing summary is available in ${LOG_FILE}" | tee -a "$LOG_FILE"
