#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# --- 脚本基础设置 ---
GPU_ID="0"
CONFIG_FILE="configs/openset/dyen.yaml"

# --- 批量实验循环 ---
for dataset in banking
do
    for seed in 0
    do
        for ratio in 0.25
        do
            echo "===================================================="
            echo "Running DyEn -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
            echo "===================================================="

            # 使用 parse_yaml.py 将YAML配置和循环变量转换为命令行参数
            params=$(python tools/parse_yaml.py \
                --config ${CONFIG_FILE} \
                --dataset ${dataset} \
                --seed ${seed} \
                --known_cls_ratio ${ratio} \
                --gpu_id ${GPU_ID})
            
            # 执行模型的主Python脚本
            python code/openset/baselines/DyEn/run_main.py ${params}

            echo "Finished run for Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
        done
    done
done

echo "All DyEn experiments have been completed."