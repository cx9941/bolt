#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: TAN 算法启动脚本
# ===================================================================

# --- 1. 基础配置 ---
CONFIG_FILE="configs/gcd/tan.yaml" 
GPU_ID="0"
# export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡

export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出

# 创建日志目录
mkdir -p time_log
LOG_FILE="time_log/tan_time.log"

# --- 2. 循环控制区 ---
for seed in 0
do
for dataset in 'banking'
do
    echo ">>>>>>>>>> Starting process for dataset: [${dataset}] <<<<<<<<<<" | tee -a $LOG_FILE

    for known_cls_ratio in 0.25
    do
    for fold_idx in 0
    do
    for labeled_ratio in 0.1
    do
        echo "========================================================================" | tee -a $LOG_FILE
        echo "Running TAN with: dataset=$dataset, known_cls_ratio=$known_cls_ratio, seed=$seed" | tee -a $LOG_FILE
        echo "========================================================================" | tee -a $LOG_FILE

        start_time=$(date +%s)

        python code/gcd/baselines/TAN/run.py \
            --config $CONFIG_FILE \
            --dataset $dataset \
            --known_cls_ratio $known_cls_ratio \
            --seed $seed \
            --gpu_id $GPU_ID \
            --pretrain_dir ./outputs/gcd/tan/premodel_${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_seed${seed} \
            --pretrain \
            --save_model \
            --freeze_bert_parameters

        end_time=$(date +%s)
        runtime=$((end_time - start_time))

        # 转换成时分秒
        hours=$((runtime / 3600))
        minutes=$(((runtime % 3600) / 60))
        seconds=$((runtime % 60))

        echo "Finished task: dataset=$dataset, known_cls_ratio=$known_cls_ratio, seed=$seed" | tee -a $LOG_FILE
        echo "Runtime: ${hours}h ${minutes}m ${seconds}s (${runtime} seconds)" | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE

    done
    done
    done

done # dataset 循环结束
done # seed 循环结束

echo "All TAN experiments have been completed." | tee -a $LOG_FILE
