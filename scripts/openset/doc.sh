#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# --- 脚本基础设置 ---
GPU_ID="0"
CONFIG_FILE="configs/openset/doc.yaml"

# --- 批量实验循环 ---
for seed in 0
do
    for dataset in banking
    do
        for ratio in 0.25
        do
            echo "===================================================="
            echo "Running DOC -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
            echo "===================================================="

            # --- 改造核心 ---
            # 废弃 parse_yaml.py，直接调用主程序并传入高优先级参数
            python code/openset/baselines/DOC/DOC.py \
                --config ${CONFIG_FILE} \
                --dataset ${dataset} \
                --seed ${seed} \
                --known_cls_ratio ${ratio} \
                --gpu_id ${GPU_ID} \
                --output_dir ./outputs/openset/doc/${dataset}_${ratio}_${seed} # 动态生成输出目录

            echo "Finished run for Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
        done
    done
done

echo "All specified DOC experiments have been completed."
