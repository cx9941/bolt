#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: BERT 任务启动脚本 (两阶段)
# ===================================================================

# --- 1. 基础配置 ---
CONFIG_FILE="configs/gcd/bert.yaml"
GPU_ID="0" # 根据您的原始脚本设置

# --- 添加原始脚本中的环境变量 ---
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# --- 2. 循环控制区 ---
for seed in 0
do
for dataset in 'banking'
do
for known_cls_ratio in 0.25
do
for fold_idx in 0
do
    echo "========================================================================"
    echo "Running BERT with: dataset=${dataset}, kcr=${known_cls_ratio}, fold=${fold_idx}, seed=${seed}"
    echo "========================================================================"

    # --- 3. 参数生成 ---
    ALL_ARGS=$(python tools/parse_yaml.py --config $CONFIG_FILE \
        --dataset $dataset \
        --known_cls_ratio $known_cls_ratio \
        --fold_idx $fold_idx \
        --seed $seed \
        --gpu_id $GPU_ID \
    )

    # --- 4. 执行区 (两阶段) ---
    echo "--- Stage 1: Running Pre-training ---"
    python code/gcd/bert/pretrain.py $ALL_ARGS

    echo "--- Stage 2: Running Testing ---"
    python code/gcd/bert/test.py $ALL_ARGS

done
done
done
done

echo "All BERT experiments have been completed."