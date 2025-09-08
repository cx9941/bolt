#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: ALUP 算法启动脚本 (V2 - 带阶段控制)
# ===================================================================

# --- 1. 控制面板 ---
# 通过修改下面的 true/false 来选择要执行的阶段
RUN_STAGE_1=true
RUN_STAGE_2=true

# --- 2. 基础配置 ---
CONFIG_FILE="configs/gcd/alup.yaml" 
GPU_ID="3"
# export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡
export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出
export OPENAI_API_KEY="6c5c96209c4ef126a87fbe9840fab7c346b4c6fb6a57529fce7dea01c683fd1b"

# --- 3. 循环控制区 ---
for seed in 0 
do
for dataset in 'banking'
do
for known_cls_ratio in 0.25
do
for fold_idx in 0
do
    echo "========================================================================"
    echo "Running ALUP with: dataset=${dataset}, kcr=${known_cls_ratio}, fold=${fold_idx}, seed=${seed}"
    echo "========================================================================"

    # --- 4. 参数生成 (动态路径) ---
    # `output_base_dir` 从YAML读取, 这里只定义子目录
    PRETRAIN_SUBDIR="pretrain/${dataset}_${known_cls_ratio}_${fold_idx}_${seed}"
    FINETUNE_SUBDIR="finetune/${dataset}_${known_cls_ratio}_${fold_idx}_${seed}"
    
    # --- 5. 执行区 (根据控制面板选择执行) ---
    if [ "$RUN_STAGE_1" = true ]; then
        # 第 1 阶段: 执行预训练和对比学习
        echo "--- Stage 1: Running Pre-training & Contrastive Learning ---"
        python code/gcd/baselines/ALUP/run.py \
            --config $CONFIG_FILE \
            --dataset $dataset \
            --known_cls_ratio $known_cls_ratio \
            --fold_idx $fold_idx \
            --seed $seed \
            --gpu_id $GPU_ID \
            --do_pretrain_and_contrastive \
            --output_subdir $PRETRAIN_SUBDIR
        
        echo "--- Stage 1 finished. ---"
    else
        echo "--- Stage 1: Skipped. ---"
    fi
    
    if [ "$RUN_STAGE_2" = true ]; then
        # 第 2 阶段: 执行主动学习微调
        echo "--- Stage 2: Running Active Learning Fine-tuning ---"
        python code/gcd/baselines/ALUP/run.py \
            --config $CONFIG_FILE \
            --dataset $dataset \
            --known_cls_ratio $known_cls_ratio \
            --fold_idx $fold_idx \
            --seed $seed \
            --gpu_id $GPU_ID \
            --do_al_finetune \
            --pretrained_stage1_subdir $PRETRAIN_SUBDIR \
            --output_subdir $FINETUNE_SUBDIR

        echo "--- Stage 2 finished. ---"
    else
        echo "--- Stage 2: Skipped. ---"
    fi

done
done
done
done

echo "All selected ALUP experiments have been completed."