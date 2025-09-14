#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: GeoID 算法启动脚本 (v2)
# ===================================================================

# --- 1. 基础配置 ---
CONFIG_FILE="configs/gcd/geoid.yaml" 
GPU_ID="3"
export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡

export OMP_NUM_THREADS=16
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export DISPLAY=:10.0
export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出

# --- 2. 循环控制区 ---
for seed in 0
do
# 使用标准的数据集列表
for dataset in 'banking'
do
    # --- A. 为当前 dataset 启动计时器 ---
    dataset_start_time=$(date +%s)
    echo ""
    echo ">>>>>>>>>> Starting process for dataset: [${dataset}] <<<<<<<<<<"

    for known_cls_ratio in 0.25
    do
    for fold_idx in 0
    do
        echo "========================================================================"
        echo "Running GeoID with: dataset=${dataset}, known_cls_ratio=${known_cls_ratio}, fold=${fold_idx}, seed=${seed}"
        echo "========================================================================"

        # --- 3. 执行区 (直接调用) ---
        python code/gcd/baselines/GeoID/geoid.py \
            --config $CONFIG_FILE \
            --dataset $dataset \
            --known_cls_ratio $known_cls_ratio \
            --fold_idx $fold_idx \
            --seed $seed \
            --report_pretrain

    done
    done

done # dataset 循环结束
done # seed 循环结束

echo "All GeoID experiments have been completed."
echo "Final timing summary is available in ${LOG_FILE}"