#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: SDC 算法启动脚本 (最终重构版)
# ===================================================================
# --- 1. 基础配置 ---
CONFIG_FILE="configs/gcd/sdc.yaml" 
GPU_ID="3"
export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡
export OPENBLAS_NUM_THREADS=32
export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出

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
    echo "========================================================================"
    echo "Running SDC with: dataset=${dataset}, known_cls_ratio=${known_cls_ratio}, fold=${fold_idx}, seed=${seed}"
    echo "========================================================================"

    # --- 3. 参数生成 (动态路径) ---
    # 定义动态的 pretrain_dir 和 train_dir，这将覆盖 YAML 中的值
    PRETRAIN_DIR_DYN="outputs/gcd/sdc/premodels/${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_seed${seed}"
    TRAIN_DIR_DYN="outputs/gcd/sdc/models/${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_seed${seed}"

    # --- 4. 执行区 (两阶段) ---
    # 第 1 阶段: 执行预训练和基线评估 (baseline.py)
    echo "--- Stage 1: Running Pre-training (baseline.py) ---"
    python code/gcd/baselines/SDC/baseline.py \
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
    
    echo "--- Pre-training finished. ---"
    
    # 第 2 阶段: 执行正式训练 (train.py)
    echo "--- Stage 2: Running Main Training (train.py) ---"
    python code/gcd/baselines/SDC/train.py \
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

    echo "--- Main training finished. ---"

done
done
done
done
done

echo "All SDC experiments (two-stage) have been completed."