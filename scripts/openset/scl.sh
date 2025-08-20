#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# --- 脚本基础设置 ---
GPU_ID="0" 
CONFIG_FILE="configs/openset/scl.yaml"


for seed in 0
do
    for dataset in banking
    do
        # 直接循环小数，变量名也直接使用 ratio
        for ratio in 0.25
        do
            # 不再需要转换步骤
            echo "===================================================="
            echo "Running SCL -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
            echo "===================================================="

            # 核心步骤：从YAML加载所有配置，直接覆盖小数比例
            params=$(python tools/parse_yaml.py \
                --config ${CONFIG_FILE} \
                --dataset ${dataset} \
                --seed ${seed} \
                --known_cls_ratio ${ratio} \
                --gpu_id ${GPU_ID})

            # 执行模型的主Python脚本
            python code/openset/baselines/SCL/train.py ${params}

            echo "Finished run for Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
            echo ""
        done
    done
done

echo "All SCL experiments have been completed."