#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: LOOP 算法启动脚本 (修正版)
# ===================================================================
# --- 1. 基础配置 ---
CONFIG_FILE="configs/gcd/loop.yaml" 
GPU_ID="3"
export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡

export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出
export OPENAI_API_KEY=6c5c96209c4ef126a87fbe9840fab7c346b4c6fb6a57529fce7dea01c683fd1b

# --- 2. 循环控制区 ---
for seed in 0 
do
# 使用标准的数据集列表
for dataset in 'banking'
do
    echo ">>>>>>>>>> Starting process for dataset: [${dataset}] <<<<<<<<<<"

    for known_cls_ratio in 0.25
    do
    for fold_idx in 0
    do
    for labeled_ratio in 0.5
    do
        echo "========================================================================"
        echo "Running LOOP with: dataset=${dataset}, known_cls_ratio=${known_cls_ratio}, fold=${fold_idx}, seed=${seed}"
        echo "========================================================================"

        PRETRAIN_DIR_DYN="outputs/gcd/loop/premodels/${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_seed${seed}"
        SAVE_MODEL_DIR="outputs/gcd/loop/models/${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_seed${seed}"
        # --- 3. 执行区 (直接调用) ---
        # 移除了 parse_yaml.py，直接向 loop.py 传递参数
        python code/gcd/baselines/LOOP/loop.py \
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

    done
    done
    done

done # dataset 循环结束
done # seed 循环结束

echo "All LOOP experiments have been completed."
echo "Final timing summary is available in ${LOG_FILE}"