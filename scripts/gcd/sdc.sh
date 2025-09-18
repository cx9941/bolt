#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: SDC 算法启动脚本 (最终重构版)
# ===================================================================

# --- 1. 基础配置 ---
CONFIG_FILE="configs/gcd/sdc.yaml" 
GPU_ID="1"
export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡
export OPENBLAS_NUM_THREADS=32
export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出

# 创建日志目录
mkdir -p time_log
LOG_FILE="time_log/sdc_time.log"

# --- 2. 循环控制区 ---
for seed in 0 
do
for dataset in 'banking'
do
for known_cls_ratio in 0.25
do
for labeled_ratio in 0.1
do
for fold_idx in 0
do
    echo "========================================================================" | tee -a $LOG_FILE
    echo "Running SDC with: dataset=${dataset}, known_cls_ratio=${known_cls_ratio}, labeled_ratio=${labeled_ratio}, fold=${fold_idx}, seed=${seed}" | tee -a $LOG_FILE
    echo "========================================================================" | tee -a $LOG_FILE

    start_time=$(date +%s)

    # --- 3. 参数生成 (动态路径) ---
    PRETRAIN_DIR_DYN="outputs/gcd/sdc/premodels/${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_seed${seed}"
    TRAIN_DIR_DYN="outputs/gcd/sdc/models/${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_seed${seed}"

    # --- 4. 执行区 (两阶段) ---
    # 第 1 阶段: 执行预训练和基线评估 (baseline.py)
    echo "--- Stage 1: Running Pre-training (baseline.py) ---" | tee -a $LOG_FILE
    python code/gcd/baselines/SDC/pretrain.py \
        --config $CONFIG_FILE \
        --dataset $dataset \
        --known_cls_ratio $known_cls_ratio \
        --labeled_ratio $labeled_ratio \
        --fold_idx $fold_idx \
        --seed $seed \
        --gpu_id $GPU_ID \
        --pretrain_dir $PRETRAIN_DIR_DYN \
        --pretrain \
        --save_model
    
    echo "--- Pre-training finished. ---" | tee -a $LOG_FILE
    
    # 第 2 阶段: 执行正式训练 (train.py)
    echo "--- Stage 2: Running Main Training (train.py) ---" | tee -a $LOG_FILE
    python code/gcd/baselines/SDC/run.py \
        --config $CONFIG_FILE \
        --dataset $dataset \
        --known_cls_ratio $known_cls_ratio \
        --labeled_ratio $labeled_ratio \
        --fold_idx $fold_idx \
        --seed $seed \
        --gpu_id $GPU_ID \
        --pretrain_dir $PRETRAIN_DIR_DYN \
        --train_dir $TRAIN_DIR_DYN \
        --save_model

    echo "--- Main training finished. ---" | tee -a $LOG_FILE

    end_time=$(date +%s)
    runtime=$((end_time - start_time))

    # 转换成时分秒
    hours=$((runtime / 3600))
    minutes=$(((runtime % 3600) / 60))
    seconds=$((runtime % 60))

    echo "Finished SDC task: dataset=${dataset}, known_cls_ratio=${known_cls_ratio}, labeled_ratio=${labeled_ratio}, fold=${fold_idx}, seed=${seed}" | tee -a $LOG_FILE
    echo "Runtime: ${hours}h ${minutes}m ${seconds}s (${runtime} seconds)" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE

done
done
done
done
done

echo "All SDC experiments (two-stage) have been completed." | tee -a $LOG_FILE
