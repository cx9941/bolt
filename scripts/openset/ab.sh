#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# ===================================================================
# AB 算法启动脚本（含耗时日志）
# ===================================================================

# --- 脚本基础设置 ---
GPU_ID="0"
CONFIG_FILE="configs/openset/ab.yaml"
export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出
# 如需锁定到单卡，可取消下一行注释
# export CUDA_VISIBLE_DEVICES=$GPU_ID

# --- 批量实验循环 ---
for emb_name in sbert
do
  for seed in 0
  do
    for dataset in banking
    do
      # 为当前 dataset（在该 emb/seed 下）启动计时器
      dataset_start_time=$(date +%s)
      echo ""
      echo ">>>>>>>>>> Starting process for dataset: [${dataset}] (emb=${emb_name}, seed=${seed}) <<<<<<<<<<"

      for ratio in 0.25
      do
        echo "===================================================="
        echo "Running AB -> Dataset: ${dataset}, Embedding: ${emb_name}, Seed: ${seed}, Known Ratio: ${ratio}"
        echo "===================================================="

        # 单次组合计时开始
        run_start_time=$(date +%s)

        # --- 改造核心 ---
        # 废弃 parse_yaml.py，直接调用主程序并传入高优先级参数
        python code/openset/baselines/AB/code/run.py \
          --config "${CONFIG_FILE}" \
          --dataset "${dataset}" \
          --emb_name "${emb_name}" \
          --seed "${seed}" \
          --known_cls_ratio "${ratio}" \
          --gpu_id "${GPU_ID}" \
          --output_dir "./outputs/openset/ab/${dataset}_${emb_name}_${ratio}_${seed}"

      done

    done
  done
done

echo "All AB experiments have been completed."