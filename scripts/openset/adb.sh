#!/bin/bash
set -o errexit

CONFIG_FILE="configs/openset/adb.yaml" 
GPU_ID="0"

export CUDA_VISIBLE_DEVICES=$GPU_ID

# --- 循环控制区 ---
# 在这里定义您要批量运行的所有变量
for s in 0
do
for dataset in 'banking'
do
for rate in 0.25
do
    echo "========================================================================"
    echo "Running ADB with: dataset=$dataset, known_cls_ratio=$rate, seed=$s"
    echo "========================================================================"

    # --- 执行区 ---
    # 1. 调用Python工具，传入基础配置文件和本次循环的覆盖参数
    ALL_ARGS=$(python tools/parse_yaml.py --config $CONFIG_FILE \
        --dataset $dataset \
        --known_cls_ratio $rate \
        --seed $s)

    # 2. 执行 ADB 的主程序，并传入所有参数
    python code/openset/baselines/ADB/ADB.py $ALL_ARGS --gpu_id $GPU_ID

done
done
done