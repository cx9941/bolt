#!/bin/bash
set -o errexit

# --- 1. 基础配置 ---
GPU_ID="0"
CONFIG_FILE="configs/openset/knncon.yaml"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export TRANSFORMERS_DEFAULT_BACKEND="pt" # 避免GPU注册告警

# --- 2. 循环控制区 ---
for dataset in banking
do
    for seed in 0
    do
        for ratio in 0.25
        do
            echo "===================================================="
            echo "Running KnnCon -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
            echo "===================================================="

            # --- 3. 执行区 (新版) ---
            # 直接调用主脚本，传入配置文件和覆盖参数
            python code/openset/baselines/KnnCon/run_main.py \
                --config ${CONFIG_FILE} \
                --dataset ${dataset} \
                --seed ${seed} \
                --known_cls_ratio ${ratio} \
                --gpu_id ${GPU_ID}

            echo "Finished run for Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
        done
    done
done

echo "All KnnCon experiments have been completed."