#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: PLM OOD 算法启动脚本 (V2 - 带阶段控制)
# ===================================================================

# --- 1. 控制面板 ---
# 通过修改下面的 true/false 来选择要执行的阶段
RUN_STAGE_1=false   # 阶段1: 执行预训练 (pretrain.py)
RUN_STAGE_2=true   # 阶段2: 执行OOD训练 (train_ood.py)

# --- 2. 基础配置 ---
# 基础模板配置文件，命令行中的参数将覆盖这里面的值
CONFIG_FILE="configs/openset/plm_ood.yaml" 
GPU_ID="0"
export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡

# --- 3. 循环控制区 ---
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

    # --- 4. 执行区 (根据控制面板选择执行) ---
    
    # 阶段 1: 预训练
    if [ "$RUN_STAGE_1" = true ]; then
        echo "--- Stage 1: Running Pre-training ---"
        python code/openset/plm_ood/pretrain.py \
            --config $CONFIG_FILE \
            --dataset_name $dataset_name \
            --rate $rate \
            --seed $seed \
            --reg_loss $reg_loss
        echo "--- Stage 1 finished. ---"
    else
        echo "--- Stage 1: Skipped. ---"
    fi

    # 阶段 2: OOD 训练
    if [ "$RUN_STAGE_2" = true ]; then
        echo "--- Stage 2: Running OOD Training ---"
        python code/openset/plm_ood/train_ood.py \
            --config $CONFIG_FILE \
            --dataset_name $dataset_name \
            --rate $rate \
            --seed $seed \
            --reg_loss $reg_loss
        echo "--- Stage 2 finished. ---"
    else
        echo "--- Stage 2: Skipped. ---"
    fi

done
done
done
done

echo "All selected PLM OOD experiments have been completed."