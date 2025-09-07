#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# --- 脚本基础设置 ---
GPU_ID="0"
# 只有一个统一的配置文件
CONFIG_FILE="configs/openset/clap.yaml" 

# --- 批量实验循环 ---
for dataset in banking
do
    for seed in 0
    do
        for ratio in 0.25
        do
            echo "================================================================"
            echo "Running CLAP -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
            echo "================================================================"
            
            # --- 动态生成本次实验的统一输出目录 ---
            OUTPUT_DIR="./outputs/openset/clap/${dataset}_${ratio}_${seed}"

            # --- 阶段一：Finetune ---
            echo "---> STAGE 1: Finetuning..."
            python code/openset/baselines/CLAP/finetune/run_kccl.py \
                --config ${CONFIG_FILE} \
                --dataset ${dataset} \
                --seed ${seed} \
                --known_cls_ratio ${ratio} \
                --gpu_id ${GPU_ID} \
                --output_dir ${OUTPUT_DIR} # 传入统一的输出目录

            # --- 阶段二：Boundary Adjustment ---
            echo "---> STAGE 2: Boundary Adjustment..."
            # 关键：将第一阶段的模型输出路径，作为第二阶段的预训练模型输入路径
            FINETUNED_MODEL_PATH="${OUTPUT_DIR}/finetuned_model"

            python code/openset/baselines/CLAP/boundary_adjustment/run_adbes.py \
                --config ${CONFIG_FILE} \
                --dataset ${dataset} \
                --seed ${seed} \
                --known_cls_ratio ${ratio} \
                --gpu_id ${GPU_ID} \
                --output_dir ${OUTPUT_DIR} \
                --pretrain_dir ${FINETUNED_MODEL_PATH} # 关键：动态传入预训练模型路径

            echo "--- Finished run for ${dataset}, seed ${seed}, ratio ${ratio} ---"
        done
    done
done

echo "All CLAP experiments have been completed."
