#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# --- 脚本基础设置 ---
GPU_ID="0"
CONFIG_FILE="configs/openset/ab.yaml"
export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出

# --- 批量实验循环 ---
for emb_name in sbert
do
    for seed in 0
    do
        for dataset in banking
        do
            for ratio in 0.25
            do
                echo "===================================================="
                echo "Running AB -> Dataset: ${dataset}, Embedding: ${emb_name}, Seed: ${seed}, Known Ratio: ${ratio}"
                echo "===================================================="

                # --- 改造核心 ---
                # 废弃 parse_yaml.py，直接调用主程序并传入高优先级参数
                python code/openset/baselines/AB/code/run.py \
                    --config ${CONFIG_FILE} \
                    --dataset ${dataset} \
                    --emb_name ${emb_name} \
                    --seed ${seed} \
                    --known_cls_ratio ${ratio} \
                    --gpu_id ${GPU_ID} \
                    --output_dir ./outputs/openset/ab/${dataset}_${emb_name}_${ratio}_${seed} # 动态生成统一的输出目录
                
                echo "Finished run for Dataset: ${dataset}, Embedding: ${emb_name}, Seed: ${seed}, Known Ratio: ${ratio}"
                echo ""
            done
        done
    done
done

echo "All AB experiments have been completed."
