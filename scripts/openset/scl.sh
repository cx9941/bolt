#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# --- 1. 基础配置 ---
GPU_ID="0" 
CONFIG_FILE="configs/openset/scl.yaml"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# --- 2. 循环控制区 ---
for seed in 0
do
    for dataset in banking
    do
        for ratio in 0.25
        do
            echo "===================================================="
            echo "Running SCL -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
            echo "===================================================="

            # --- 3. 执行区 (新版) ---
            # 直接调用主脚本，传入配置文件和覆盖参数。
            # 注意：--cont_loss 和 --sup_cont 是根据您的YAML配置添加的行为开关。
            python code/openset/baselines/SCL/train.py \
                --config ${CONFIG_FILE} \
                --dataset ${dataset} \
                --seed ${seed} \
                --known_cls_ratio ${ratio} \
                --gpu_id ${GPU_ID} \
                --cont_loss \
                --sup_cont

            echo "Finished run for Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
            echo ""
        done
    done
done

echo "All SCL experiments have been completed."