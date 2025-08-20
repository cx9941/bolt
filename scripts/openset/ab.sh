#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# --- 脚本基础设置 ---
GPU_ID="0"
CONFIG_FILE="configs/openset/ab.yaml"

# --- 批量实验循环 ---
# 注意：最外层循环的变量是 emb_name
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

                # 使用 parse_yaml.py 将YAML配置和循环变量转换为命令行参数
                params=$(python tools/parse_yaml.py \
                    --config ${CONFIG_FILE} \
                    --dataset ${dataset} \
                    --emb_name ${emb_name} \
                    --seed ${seed} \
                    --known_cls_ratio ${ratio} \
                    --gpu_id ${GPU_ID})

                # 执行我们重构后的主Python脚本
                python code/openset/baselines/AB/code/run.py ${params}

                echo "Finished run for Dataset: ${dataset}, Embedding: ${emb_name}, Seed: ${seed}, Known Ratio: ${ratio}"
                echo ""
            done
        done
    done
done

echo "All AB experiments have been completed."