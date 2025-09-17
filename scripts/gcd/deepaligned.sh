#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: DeepAligned 算法启动脚本 (v2)
# ===================================================================

# --- 1. 基础配置 ---
CONFIG_FILE="configs/gcd/deepaligned.yaml" 
GPU_ID="3"
# export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡

export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出

# --- 2. 循环控制区 ---
for seed in 0
do
# 使用标准的数据集列表
for dataset in 'banking'
do
    echo ""
    echo ">>>>>>>>>> Starting process for dataset: [${dataset}] <<<<<<<<<<"

    for known_cls_ratio in 0.25
    do
    for fold_idx in 0
    do
    for labeled_ratio in 0.1
    do
        echo "========================================================================"
        echo "Running DeepAligned with: dataset=${dataset}, kcr=${known_cls_ratio}, fold=${fold_idx}, seed=${seed}"
        echo "========================================================================"

        # --- 3. 执行区 (直接调用) ---
        python code/gcd/baselines/DeepAligned-Clustering/DeepAligned.py \
            --config $CONFIG_FILE \
            --dataset $dataset \
            --known_cls_ratio $known_cls_ratio \
            --labeled_ratio $labeled_ratio \
            --fold_idx $fold_idx \
            --seed $seed \
            --gpu_id $GPU_ID \
            --freeze_bert_parameters \
            --save_model \
            --pretrain

    done
    done
    done
done # dataset 循环结束
done # seed 循环结束

echo "All DeepAligned experiments have been completed."