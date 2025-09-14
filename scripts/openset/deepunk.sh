#!/bin/bash
set -o errexit

# ===================================================================
# DeepUnk 算法启动脚本（精简日志版）
# ===================================================================

# --- 脚本基础设置 ---
GPU_ID="0"
CONFIG_FILE="configs/openset/deepunk.yaml"
DATASET="banking"   # 当前写死为 banking，如需多数据集可改 for dataset in ...

# --- 批量实验循环 ---
for seed in 0
do
for dataset_name in 'banking'
do

  for ratio in 0.25
  do
    echo "===================================================="
    echo "Running DeepUnk -> Dataset: ${DATASET}, Seed: ${seed}, Known Ratio: ${ratio}"
    echo "===================================================="

    python code/openset/baselines/DeepUnk/experiment.py \
      --config "${CONFIG_FILE}" \
      --dataset "${dataset_name}" \
      --seed "${seed}" \
      --known_cls_ratio "${ratio}" \
      --gpu_id "${GPU_ID}" \
      --output_dir "./outputs/openset/deepunk/${DATASET}_${ratio}_${seed}"
  done

done
done

echo "All DeepUnk experiments have been completed."