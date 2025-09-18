#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: LOOP 算法启动脚本 (修正版)
# ===================================================================

# --- 1. 基础配置 ---
CONFIG_FILE="configs/gcd/loop.yaml"
GPU_ID="1"
export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡
export TF_CPP_MIN_LOG_LEVEL=2
export OPENAI_API_KEY=6c5c96209c4ef126a87fbe9840fab7c346b4c6fb6a57529fce7dea01c683fd1b

# 日志目录与文件
mkdir -p time_log
LOG_FILE="time_log/loop_time.log"

echo "================== LOOP Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"
total_start=$(date +%s)

# --- 2. 循环控制区 ---
for seed in 0
do
for dataset in 'banking'
do
    echo ">>>>>>>>>> Starting process for dataset: [${dataset}] <<<<<<<<<<" | tee -a "$LOG_FILE"

    for known_cls_ratio in 0.25
    do
    for fold_idx in 0
    do
    for labeled_ratio in 0.5
    do
        echo "========================================================================" | tee -a "$LOG_FILE"
        echo "Running LOOP with: dataset=${dataset}, known_cls_ratio=${known_cls_ratio}, labeled_ratio=${labeled_ratio}, fold=${fold_idx}, seed=${seed}" | tee -a "$LOG_FILE"
        echo "========================================================================" | tee -a "$LOG_FILE"

        run_start=$(date +%s)

        PRETRAIN_DIR_DYN="outputs/gcd/loop/premodels/${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_seed${seed}"
        SAVE_MODEL_DIR="outputs/gcd/loop/models/${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_seed${seed}"
        # --- 3. 执行区 (直接调用) ---
        # 移除了 parse_yaml.py，直接向 loop.py 传递参数
        python code/gcd/baselines/LOOP/run.py \
            --config $CONFIG_FILE \
            --dataset $dataset \
            --known_cls_ratio $known_cls_ratio \
            --labeled_ratio $labeled_ratio \
            --fold_idx $fold_idx \
            --seed $seed \
            --gpu_id $GPU_ID \
            --pretrain_dir $PRETRAIN_DIR_DYN \
            --save_model_path $SAVE_MODEL_DIR \
            --save_premodel \
            --save_model

        run_end=$(date +%s)
        runtime=$((run_end - run_start))
        h=$((runtime / 3600)); m=$(((runtime % 3600) / 60)); s=$((runtime % 60))

        echo "Finished LOOP task: dataset=${dataset}, known_cls_ratio=${known_cls_ratio}, labeled_ratio=${labeled_ratio}, fold=${fold_idx}, seed=${seed}" | tee -a "$LOG_FILE"
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

echo "All LOOP experiments have been completed." | tee -a "$LOG_FILE"
echo "Total runtime: ${th}h ${tm}m ${ts}s (${total} seconds)" | tee -a "$LOG_FILE"
echo "Final timing summary is available in ${LOG_FILE}" | tee -a "$LOG_FILE"
