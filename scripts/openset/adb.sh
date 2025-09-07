#!/bin/bash
set -o errexit

# --- 基础配置 ---
CONFIG_FILE="configs/openset/adb.yaml" 
GPU_ID="0"

# --- 循环控制区 ---
for s in 0
do
for dataset in 'banking'
do
for rate in 0.25
do
    echo "========================================================================"
    echo "Running ADB with: dataset=$dataset, known_cls_ratio=$rate, seed=$s"
    echo "========================================================================"

    # --- 改造核心 ---
    # 废弃 parse_yaml.py，直接调用主程序并传入高优先级参数
    python code/openset/baselines/ADB/ADB.py \
        --config $CONFIG_FILE \
        --dataset $dataset \
        --known_cls_ratio $rate \
        --seed $s \
        --gpu_id $GPU_ID \
        --output_dir ./outputs/openset/adb/${dataset}_${rate}_${s} # 动态生成统一的输出目录
done
done
done

echo "All ADB experiments have been completed."
