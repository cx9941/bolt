#!/bin/bash
method=alup
prefix=outs
suffix=alup_pretrain 

# --- 只跑一个最简单的设定 ---
dataset=banking
known_cls_ratio=0.75
seed=0

echo "===> [Fast Test] Running Pre-training for: ${dataset}, Seed=${seed}"

# 假设预训练的配置文件是 config_banking.yaml
# 假设预训练对应的模式是 'train'
python run.py \
    --dataset $dataset \
    --method ${method} \
    --known_cls_ratio $known_cls_ratio \
    --seed $seed \
    --mode 'train' \
    --config_file_name ./methods/${method}/configs/config_${dataset}.yaml \
    --num_train_epochs 2 \
    --output_dir ./${prefix}/${method} \
    --model_file_name model_${known_cls_ratio}_${dataset}_${seed}_${suffix} \
    --result_dir ./${prefix}/${method}/results_${dataset}_${pretrain_suffix}_kcr${known_cls_ratio} \
    --dataset_dir datasets_${known_cls_ratio}/data_${known_cls_ratio}_${dataset}_${seed}.pkl