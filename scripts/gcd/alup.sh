#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: ALUP 算法启动脚本 (V2 - 带阶段控制, 含耗时日志)
# ===================================================================

# --- 1. 控制面板 ---
RUN_STAGE_1=true
RUN_STAGE_2=true

# --- 2. 基础配置 ---
CONFIG_FILE="configs/gcd/alup.yaml" 
GPU_ID="5"
# export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡
export TF_CPP_MIN_LOG_LEVEL=2
export OPENAI_API_KEY="6c5c96209c4ef126a87fbe9840fab7c346b4c6fb6a57529fce7dea01c683fd1b"

# --- 日志目录 ---
mkdir -p time_log
LOG_FILE="time_log/alup_time.log"
echo "================== ALUP Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"
total_start=$(date +%s)

# --- 3. 循环控制区 ---
for seed in 0 
do
for dataset in 'banking'
do
    echo ">>>>>>>>>> Starting process for dataset: [${dataset}] <<<<<<<<<<" | tee -a "$LOG_FILE"

    for known_cls_ratio in 0.25
    do
    for fold_idx in 0
    do
    for labeled_ratio in 0.1
    do
        echo "========================================================================" | tee -a "$LOG_FILE"
        echo "Running ALUP with: dataset=${dataset}, kcr=${known_cls_ratio}, labeled_ratio=${labeled_ratio}, fold=${fold_idx}, seed=${seed}" | tee -a "$LOG_FILE"
        echo "========================================================================" | tee -a "$LOG_FILE"

        run_start=$(date +%s)

        # --- 4. 参数生成 ---
        PRETRAIN_SUBDIR="pretrain/${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_${seed}"
        FINETUNE_SUBDIR="finetune/${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_${seed}"

        # --- 5. 执行区 ---
        if [ "$RUN_STAGE_1" = true ]; then
            echo "--- Stage 1: Running Pre-training & Contrastive Learning ---" | tee -a "$LOG_FILE"
            python code/gcd/baselines/ALUP/run.py \
                --config $CONFIG_FILE \
                --dataset $dataset \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --fold_idx $fold_idx \
                --seed $seed \
                --gpu_id $GPU_ID \
                --do_pretrain_and_contrastive \
                --output_subdir $PRETRAIN_SUBDIR
            echo "--- Stage 1 finished. ---" | tee -a "$LOG_FILE"
        else
            echo "--- Stage 1: Skipped. ---" | tee -a "$LOG_FILE"
        fi

        if [ "$RUN_STAGE_2" = true ]; then
            echo "--- Stage 2: Running Active Learning Fine-tuning ---" | tee -a "$LOG_FILE"
            python code/gcd/baselines/ALUP/run.py \
                --config $CONFIG_FILE \
                --dataset $dataset \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --fold_idx $fold_idx \
                --seed $seed \
                --gpu_id $GPU_ID \
                --do_al_finetune \
                --pretrained_stage1_subdir $PRETRAIN_SUBDIR \
                --output_subdir $FINETUNE_SUBDIR \
                --save_results
            echo "--- Stage 2 finished. ---" | tee -a "$LOG_FILE"
        else
            echo "--- Stage 2: Skipped. ---" | tee -a "$LOG_FILE"
        fi

        run_end=$(date +%s)
        runtime=$((run_end - run_start))
        h=$((runtime / 3600)); m=$(((runtime % 3600) / 60)); s=$((runtime % 60))

        echo "Finished ALUP task: dataset=${dataset}, kcr=${known_cls_ratio}, labeled_ratio=${labeled_ratio}, fold=${fold_idx}, seed=${seed}" | tee -a "$LOG_FILE"
        echo "Runtime: ${h}h ${m}m ${s}s (${runtime} seconds)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"

    done
    done
    done
done # dataset
done # seed

# --- 总耗时 ---
total_end=$(date +%s)
total=$((total_end - total_start))
th=$((total / 3600)); tm=$(((total % 3600) / 60)); ts=$((total % 60))

echo "All selected ALUP experiments have been completed." | tee -a "$LOG_FILE"
echo "Total runtime: ${th}h ${tm}m ${ts}s (${total} seconds)" | tee -a "$LOG_FILE"
echo "Final timing summary is available in ${LOG_FILE}" | tee -a "$LOG_FILE"
