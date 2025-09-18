#!/bin/bash
set -o errexit

# ===================================================================
# BOLT Platform: Glean 算法启动脚本 (v2)
# ===================================================================
# --- 1. 基础配置 ---
CONFIG_FILE="configs/gcd/glean.yaml" 
GPU_ID="0"
export CUDA_VISIBLE_DEVICES=$GPU_ID # 防止自动分配到不同卡

# 从 YAML 文件读取 API Key，或在此处设置
export OPENAI_API_KEY=6c5c96209c4ef126a87fbe9840fab7c346b4c6fb6a57529fce7dea01c683fd1b
export OMP_NUM_THREADS=16
export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出

# --- 2. 循环控制区 ---
for seed in 0
do
# 使用标准的数据集列表
for dataset in 'banking'
do
    echo ">>>>>>>>>> Starting process for dataset: [${dataset}] <<<<<<<<<<"

    for known_cls_ratio in 0.25
    do
    for labeled_ratio in 0.1
    do
    for fold_idx in 0
    do
        echo "========================================================================"
        echo "Running Glean with: dataset=${dataset}, kcr=${known_cls_ratio}, fold=${fold_idx}, seed=${seed}"
        echo "========================================================================"

        PREMODEL_DIR=./outputs/gcd/glean/premodel_${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_seed${seed}
        MODEL_DIR=./outputs/gcd/glean/model_${dataset}_${known_cls_ratio}_${labeled_ratio}_fold${fold_idx}_seed${seed}

        # --- 3. 执行区 (直接调用) ---
        python code/gcd/baselines/Glean/run.py \
            --config $CONFIG_FILE \
            --dataset $dataset \
            --known_cls_ratio $known_cls_ratio \
            --labeled_ratio $labeled_ratio \
            --fold_idx $fold_idx \
            --seed $seed \
            --gpu_id $GPU_ID \
            --save_premodel \
            --save_model \
            --feedback_cache \
            --flag_demo \
            --flag_filtering \
            --flag_demo_c \
            --flag_filtering_c \
            --api_key "$OPENAI_API_KEY" \
            --pretrain_dir $PREMODEL_DIR \
            --save_model_path $MODEL_DIR

    done
    done
    done
    done


done # dataset 循环结束
done # seed 循环结束

echo "All Glean experiments have been completed."
echo "Final timing summary is available in ${LOG_FILE}"