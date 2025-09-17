#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# --- 脚本基础设置 ---
GPU_ID="3"
CONFIG_FILE="configs/gcd/plm_gcd.yaml"
# 将 backbone 作为 shell 变量，方便修改
BACKBONE="Meta-Llama-3.1-8B-Instruct"

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# --- 批量实验循环 ---
# 使用标准的数据集列表
for dataset in 'banking'
do
    for seed in 0
    do
        for rate in 0.25
        do
        for fold_idx in 0
        do
        for labeled_ratio in 0.1
        do
            echo "========================================================================"
            echo "Running PLM-GCD -> Dataset: ${dataset}, Rate: ${rate}, Seed: ${seed}"
            echo "========================================================================"
            
            # --- 动态生成本次实验的统一输出目录 ---
            OUTPUT_DIR="./outputs/gcd/plm_gcd/${dataset}_${rate}_${seed}_${BACKBONE}"

            # --- 阶段一：Pretrain ---
            echo "---> STAGE 1: Pretraining..."
            python code/gcd/plm_gcd/pretrain.py \
                --config ${CONFIG_FILE} \
                --dataset_name ${dataset} \
                --rate ${rate} \
                --seed ${seed} \
                --gpu_id ${GPU_ID} \
                --backbone ${BACKBONE} \
                --output_dir ${OUTPUT_DIR}

            # --- 阶段二：Test ---
            echo "---> STAGE 2: Testing..."
            python code/gcd/plm_gcd/test.py \
                --config ${CONFIG_FILE} \
                --dataset_name ${dataset} \
                --rate ${rate} \
                --seed ${seed} \
                --gpu_id ${GPU_ID} \
                --backbone ${BACKBONE} \
                --output_dir ${OUTPUT_DIR}

            echo "--- Finished run for ${dataset}, rate ${rate}, seed ${seed} ---"
        done
    done
    done
    done

done

echo "All PLM-GCD experiments have been completed."