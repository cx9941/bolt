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
    # --- A. 为当前 dataset 启动计时器 ---
    dataset_start_time=$(date +%s)
    echo ""
    echo ">>>>>>>>>> Starting process for dataset: [${dataset}] <<<<<<<<<<"

    for known_cls_ratio in 0.25
    do
    for weight_cluster_instance_cl in 0.0
    do
    for fold_idx in 0
    do
        echo "========================================================================"
        echo "Running Glean with: dataset=${dataset}, kcr=${known_cls_ratio}, fold=${fold_idx}, seed=${seed}, weight_cl=${weight_cluster_instance_cl}"
        echo "========================================================================"

        # --- 3. 执行区 (直接调用) ---
        python code/gcd/baselines/Glean/GCDLLMs.py \
            --config $CONFIG_FILE \
            --dataset $dataset \
            --known_cls_ratio $known_cls_ratio \
            --fold_idx $fold_idx \
            --seed $seed \
            --gpu_id $GPU_ID \
            --weight_cluster_instance_cl $weight_cluster_instance_cl \
            --save_premodel \
            --save_model \
            --report_pretrain \
            --feedback_cache \
            --flag_demo \
            --flag_filtering \
            --flag_demo_c \
            --flag_filtering_c \
            --api_key "$OPENAI_API_KEY" 

    done
    done
    done
    done


done # dataset 循环结束
done # seed 循环结束

echo "All Glean experiments have been completed."
echo "Final timing summary is available in ${LOG_FILE}"