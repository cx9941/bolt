#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# --- 脚本基础设置 ---
GPU_ID="0"
FINETUNE_CONFIG_FILE="configs/openset/clap/1_finetune.yaml"
BOUNDARY_ADJUSTMENT_CONFIG_FILE="configs/openset/clap/2_boundary_adjustment.yaml"

# --- 批量实验循环 ---
for dataset in banking
do
    for seed in 0
    do
        for ratio in 0.25
        do
            echo "================================================================"
            echo "Running CLAP -> Dataset: ${dataset}, Seed: ${seed}, Known Ratio: ${ratio}"
            echo "================================================================"

            # --- 阶段一：Finetune ---
            echo "---> STAGE 1: Finetuning..."
            finetune_params=$(python tools/parse_yaml.py \
                --config ${FINETUNE_CONFIG_FILE} \
                --dataset ${dataset} \
                --seed ${seed} \
                --known_cls_ratio ${ratio} \
                --gpu_id ${GPU_ID})
            python code/openset/baselines/CLAP/finetune/run_kccl.py ${finetune_params}
            
            # --- 阶段二：Boundary Adjustment ---
            echo "---> STAGE 2: Boundary Adjustment..."
            # 动态获取第一阶段的模型输出路径
            output_dir=$(python tools/parse_yaml.py --config ${FINETUNE_CONFIG_FILE} | grep -oP '(?<=--output_dir )[^ ]+')
            finetuned_model_path="${output_dir}/finetuned_model"

            boundary_params=$(python tools/parse_yaml.py \
                --config ${BOUNDARY_ADJUSTMENT_CONFIG_FILE} \
                --dataset ${dataset} \
                --seed ${seed} \
                --known_cls_ratio ${ratio} \
                --gpu_id ${GPU_ID} \
                --pretrain_dir ${finetuned_model_path}) # 关键：将第一阶段的输出作为第二阶段的输入
            python code/openset/baselines/CLAP/boundary_adjustment/run_adbes.py ${boundary_params}
            
            echo "--- Finished run for ${dataset}, seed ${seed}, ratio ${ratio} ---"
        done
    done
done

echo "All CLAP experiments have been completed."