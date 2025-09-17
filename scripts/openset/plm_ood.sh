#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: PLM OOD 算法启动脚本 (V2 - 带阶段控制)
# ===================================================================

# --- 1. 控制面板 ---
RUN_STAGE_1=true   # 阶段1: 执行预训练 (pretrain.py)
RUN_STAGE_2=true   # 阶段2: 执行OOD训练 (train_ood.py)

# --- 2. 基础配置 ---
CONFIG_FILE="configs/openset/plm_ood.yaml"
GPU_ID="3"
export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡

# --- 新增：时间日志设置 ---
mkdir -p time_log
LOG_FILE="time_log/plm_ood_time.log"
echo "================== PLM_OOD Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"

# --- 3. 循环控制区 ---
for reg_loss in 'normal'
do
for seed in 0
do
for dataset_name in 'banking'
do
    for rate in 0.25
    do
        echo "========================================================================" | tee -a "$LOG_FILE"
        echo "Running with: dataset=$dataset_name, rate=$rate, seed=$seed, reg_loss=$reg_loss" | tee -a "$LOG_FILE"
        echo "========================================================================" | tee -a "$LOG_FILE"

        # 组合计时开始（覆盖 Stage1 + Stage2）
        run_start=$(date +%s)

        # 阶段 1: 预训练
        if [ "$RUN_STAGE_1" = true ]; then
            echo "--- Stage 1: Running Pre-training ---" | tee -a "$LOG_FILE"
            python code/openset/plm_ood/pretrain.py \
                --config $CONFIG_FILE \
                --dataset_name $dataset_name \
                --rate $rate \
                --seed $seed \
                --reg_loss $reg_loss
            echo "--- Stage 1 finished. ---" | tee -a "$LOG_FILE"
        else
            echo "--- Stage 1: Skipped. ---" | tee -a "$LOG_FILE"
        fi

        # 阶段 2: OOD 训练
        if [ "$RUN_STAGE_2" = true ]; then
            echo "--- Stage 2: Running OOD Training ---" | tee -a "$LOG_FILE"
            python code/openset/plm_ood/train_ood.py \
                --config $CONFIG_FILE \
                --dataset_name $dataset_name \
                --rate $rate \
                --seed $seed \
                --reg_loss $reg_loss
            echo "--- Stage 2 finished. ---" | tee -a "$LOG_FILE"
        else
            echo "--- Stage 2: Skipped. ---" | tee -a "$LOG_FILE"
        fi

        # 组合计时结束 + 记录
        run_end=$(date +%s)
        runtime=$((run_end - run_start))
        h=$((runtime / 3600)); m=$(((runtime % 3600) / 60)); s=$((runtime % 60))

        echo "Finished PLM_OOD run: dataset=${dataset_name}, rate=${rate}, seed=${seed}, reg_loss=${reg_loss}" | tee -a "$LOG_FILE"
        echo "Runtime: ${h}h ${m}m ${s}s (${runtime} seconds)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"

    done
done # dataset_name 循环结束
done
done

echo "All selected PLM OOD experiments have been completed." | tee -a "$LOG_FILE"
