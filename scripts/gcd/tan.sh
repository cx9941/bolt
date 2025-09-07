#!/bin/bash
set -o errexit

# --- 基础配置 ---
CONFIG_FILE="configs/gcd/tan.yaml" 
GPU_ID="3"
# export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡

export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出

# --- 循环控制区 ---
for seed in 0
do
for dataset in 'banking'
do
for known_cls_ratio in 0.25
do
for fold_idx in 0
do
    echo "========================================================================"
    echo "Running TAN with: dataset=$dataset, known_cls_ratio=$known_cls_ratio, seed=$seed"
    echo "========================================================================"

    python code/gcd/TAN/TAN.py \
        --config $CONFIG_FILE \
        --dataset $dataset \
        --known_cls_ratio $known_cls_ratio \
        --seed $seed \
        --gpu_id $GPU_ID \
        --pretrain_dir ./outputs/gcd/tan/premodel_${dataset}_${known_cls_ratio}_${seed} \
        --pretrain \
        --save_model \
        --freeze_bert_parameters

done
done
done
done

echo "All TAN experiments have been completed."