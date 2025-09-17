#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -o errexit

# ===================================================================
# AB 算法启动脚本（含耗时日志）
# ===================================================================

# --- 脚本基础设置 ---
GPU_ID="1"
CONFIG_FILE="configs/openset/ab.yaml"
export TF_CPP_MIN_LOG_LEVEL=2   # 忽略tf框架的输出
# 如需锁定到单卡，可取消下一行注释
export CUDA_VISIBLE_DEVICES=$GPU_ID

# --- 新增：时间日志设置 ---
mkdir -p time_log
LOG_FILE="time_log/ab_time.log"
echo "================== AB Run @ $(date '+%F %T') ==================" | tee -a "$LOG_FILE"

# --- 批量实验循环 ---
for emb_name in sbert
do
  for seed in 0
  do
    for dataset in banking
    do
      echo "" | tee -a "$LOG_FILE"
      echo ">>>>>>>>>> Starting process for dataset: [${dataset}] (emb=${emb_name}, seed=${seed}) <<<<<<<<<<" | tee -a "$LOG_FILE"

      for ratio in 0.25
      do
        echo "====================================================" | tee -a "$LOG_FILE"
        echo "Running AB -> Dataset: ${dataset}, Embedding: ${emb_name}, Seed: ${seed}, Known Ratio: ${ratio}" | tee -a "$LOG_FILE"
        echo "====================================================" | tee -a "$LOG_FILE"

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

        # 单次组合计时结束 + 记录
        run_end_time=$(date +%s)
        run_seconds=$((run_end_time - run_start_time))
        run_h=$((run_seconds / 3600)); run_m=$(((run_seconds % 3600) / 60)); run_s=$((run_seconds % 60))
        echo "Finished AB run: dataset=${dataset}, emb=${emb_name}, seed=${seed}, ratio=${ratio}" | tee -a "$LOG_FILE"
        echo "Runtime: ${run_h}h ${run_m}m ${run_s}s (${run_seconds} seconds)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"

      done
    done
  done
done

echo "All AB experiments have been completed." | tee -a "$LOG_FILE"
