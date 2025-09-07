#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# --- 脚本基础设置 ---
GPU_ID="0"
CONFIG_FILE="configs/openset/deepunk.yaml"

# --- 批量实验循环 ---
for seed in 0
do
    for ratio in 0.25
    do
        echo "===================================================="
        echo "Running DeepUnk -> Seed: ${seed}, Known Ratio: ${ratio}"
        echo "===================================================="

        # --- 改造核心 ---
        # 废弃 parse_yaml.py，直接调用主程序并传入高优先级参数
        python code/openset/baselines/DeepUnk/experiment.py \
            --config ${CONFIG_FILE} \
            --seed ${seed} \
            --known_cls_ratio ${ratio} \
            --gpu_id ${GPU_ID} \
            --output_dir ./outputs/openset/deepunk/banking_${ratio}_${seed} # 动态生成输出目录
        
        echo "Finished run for Seed: ${seed}, Known Ratio: ${ratio}"
    done
done

echo "All DeepUnk experiments have been completed."
