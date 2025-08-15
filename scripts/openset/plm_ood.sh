#!/bin/bash
set -o errexit

# --- 基础配置 ---
# 基础模板配置文件，循环中的参数将覆盖这里面的值
CONFIG_FILE="configs/openset/plm_ood.yaml" 
GPU_ID="0"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID

# --- 循环控制区 ---
# 在这里定义您要批量运行的所有变量
for reg_loss in 'normal'
do
for seed in 0
do
for dataset_name in 'banking'
do
for rate in 0.25
do
    echo "========================================================================"
    echo "Running with: dataset=$dataset_name, rate=$rate, seed=$seed, reg_loss=$reg_loss"
    echo "========================================================================"

    # --- 执行区 ---
    # 1. 调用Python工具，传入基础配置文件和本次循环的覆盖参数
    ALL_ARGS=$(python tools/parse_yaml.py --config $CONFIG_FILE \
        --dataset_name $dataset_name \
        --rate $rate \
        --seed $seed \
        --reg_loss $reg_loss)

    # 2. 按顺序执行您的Python脚本
    python code/openset/plm_ood/pretrain.py $ALL_ARGS
    python code/openset/plm_ood/train_ood.py $ALL_ARGS

done
done
done
done