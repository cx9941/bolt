#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: DPN 算法启动脚本 (v2, 含耗时日志)
# ===================================================================

# --- 1. 基础配置 ---
CONFIG_FILE="configs/gcd/dpn.yaml" 
GPU_ID="3"
# export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡

export OPENBLAS_NUM_THREADS=16
export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出

# --- 新增：时间日志设置 ---
mkdir -p time_log
LOG_FILE="time_log/dpn_time.log"
echo "================== DPN Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"
total_start=$(date +%s)

# --- 2. 循环控制区 ---
for seed in 0
do
# 使用标准的数据集列表
for dataset in 'banking'
do
    echo "" | tee -a "$LOG_FILE"
    echo ">>>>>>>>>> Starting process for dataset: [${dataset}] <<<<<<<<<<" | tee -a "$LOG_FILE"

    for known_cls_ratio in 0.25
    do
    for fold_idx in 0
    do
    for labeled_ratio in 0.5
    do
        echo "========================================================================" | tee -a "$LOG_FILE"
        echo "Running DPN with: dataset=${dataset}, known_cls_ratio=${known_cls_ratio}, labeled_ratio=${labeled_ratio}, fold=${fold_idx}, seed=${seed}" | tee -a "$LOG_FILE"
        echo "========================================================================" | tee -a "$LOG_FILE"

        # --- 单次组合计时开始 ---
        run_start=$(date +%s)

        # --- 3. 执行区 (直接调用) ---
        python code/gcd/baselines/DPN/DPN.py \
            --config $CONFIG_FILE \
            --dataset $dataset \
            --known_cls_ratio $known_cls_ratio \
            --labeled_ratio $labeled_ratio \
            --fold_idx $fold_idx \
            --seed $seed \
            --gpu_id $GPU_ID \
            --freeze_bert_parameters \
            --save_model \
            --pretrain

        # --- 单次组合计时结束+记录 ---
        run_end=$(date +%s)
        runtime=$((run_end - run_start))
        h=$((runtime / 3600)); m=$(((runtime % 3600) / 60)); s=$((runtime % 60))

        echo "Finished DPN task: dataset=${dataset}, known_cls_ratio=${known_cls_ratio}, labeled_ratio=${labeled_ratio}, fold=${fold_idx}, seed=${seed}" | tee -a "$LOG_FILE"
        echo "Runtime: ${h}h ${m}m ${s}s (${runtime} seconds)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"

    done
    done
    done

done # dataset 循环结束
done # seed 循环结束

# --- 脚本整体总耗时 ---
total_end=$(date +%s)
total=$((total_end - total_start))
th=$((total / 3600)); tm=$(((total % 3600) / 60)); ts=$((total % 60))

echo "All DPN experiments have been completed." | tee -a "$LOG_FILE"
echo "Total runtime: ${th}h ${tm}m ${ts}s (${total} seconds)" | tee -a "$LOG_FILE"
echo "Final timing summary is available in ${LOG_FILE}" | tee -a "$LOG_FILE"
