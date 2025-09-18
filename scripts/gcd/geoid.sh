#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: GeoID 算法启动脚本 (v2)
# ===================================================================

# --- 1. 基础配置 ---
CONFIG_FILE="configs/gcd/geoid.yaml" 
GPU_ID="2"
export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡

export OMP_NUM_THREADS=16
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export DISPLAY=:10.0
export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出

# --- 新增：时间日志设置 ---
mkdir -p time_log
LOG_FILE="time_log/geoid_time.log"
echo "================== GeoID Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"
total_start=$(date +%s)

# --- 2. 循环控制区 ---
for seed in 0
do
# 使用标准的数据集列表
for dataset in 'banking'
do
    echo ">>>>>>>>>> Starting process for dataset: [${dataset}] <<<<<<<<<<" | tee -a "$LOG_FILE"

    for known_cls_ratio in 0.25
    do
    for labeled_ratio in 0.1
    do
    for fold_idx in 0
    do
        echo "========================================================================" | tee -a "$LOG_FILE"
        echo "Running GeoID with: dataset=${dataset}, known_cls_ratio=${known_cls_ratio}, fold=${fold_idx}, seed=${seed}" | tee -a "$LOG_FILE"
        echo "========================================================================" | tee -a "$LOG_FILE"

        # --- 新增：单次组合计时开始 ---
        run_start=$(date +%s)

        # --- 3. 执行区 (直接调用) ---
        python code/gcd/baselines/GeoID/run.py \
            --config $CONFIG_FILE \
            --dataset $dataset \
            --known_cls_ratio $known_cls_ratio \
            --labeled_ratio $labeled_ratio \
            --fold_idx $fold_idx \
            --seed $seed \
            --report_pretrain

        # --- 新增：单次组合计时结束+记录 ---
        run_end=$(date +%s)
        runtime=$((run_end - run_start))
        h=$((runtime / 3600)); m=$(((runtime % 3600) / 60)); s=$((runtime % 60))

        echo "Finished GeoID task: dataset=${dataset}, known_cls_ratio=${known_cls_ratio}, labeled_ratio=${labeled_ratio}, fold=${fold_idx}, seed=${seed}" | tee -a "$LOG_FILE"
        echo "Runtime: ${h}h ${m}m ${s}s (${runtime} seconds)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"

    done
    done
    done

done # dataset 循环结束
done # seed 循环结束

# --- 新增：整体总耗时 ---
total_end=$(date +%s)
total=$((total_end - total_start))
th=$((total / 3600)); tm=$(((total % 3600) / 60)); ts=$((total % 60))

echo "All GeoID experiments have been completed." | tee -a "$LOG_FILE"
echo "Total runtime: ${th}h ${tm}m ${ts}s (${total} seconds)" | tee -a "$LOG_FILE"
