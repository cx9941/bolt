#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# --- 脚本基础设置 ---
GPU_ID="2"

# 设置基础配置文件路径
CONFIG_FILE="configs/openset/doc.yaml"


for seed in 0
do
    # 循环遍历所有需要测试的数据集
    for dataset in banking
    do
        # 循环遍历不同的已知类比例
        for ratio in 0.25
        do
            echo "===================================================="
            echo "Running DOC -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
            echo "===================================================="

            # 核心步骤：使用您项目的正确格式调用 parse_yaml.py
            # 直接将需要覆盖的参数以 --key value 的形式传入
            params=$(python tools/parse_yaml.py \
                --config ${CONFIG_FILE} \
                --dataset ${dataset} \
                --seed ${seed} \
                --known_cls_ratio ${ratio} \
                --gpu_id ${GPU_ID})

            # 执行模型的主Python脚本，传入生成的参数
            python code/openset/baselines/DOC/DOC.py ${params}

            echo "Finished run for Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
        done
    done
done

echo "All specified DOC experiments have been completed."