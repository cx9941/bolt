#!/bin/bash
method=alup
prefix=outs
pretrain_suffix=alup_pretrain
suffix=alup
export OPENAI_API_KEY=6c5c96209c4ef126a87fbe9840fab7c346b4c6fb6a57529fce7dea01c683fd1b

# --- 只跑一个最简单的设定 ---
dataset=banking 
known_cls_ratio=0.75
seed=0

echo "===> [Fast Test] Running ALUP Fine-tuning for: ${dataset}, Seed=${seed}"

python run.py \
    --dataset $dataset \
    --method ${method} \
    --known_cls_ratio $known_cls_ratio \
    --seed $seed \
    --config_file_name ./methods/${method}/configs/config_${dataset}_al_finetune.yaml \
    --num_train_epochs 2 \
    --save_results \
    --log_dir ./${prefix}/${method}/logs/${dataset}_${suffix} \
    --output_dir ./${prefix}/${method} \
    --dataset_dir datasets_${known_cls_ratio}/data_${known_cls_ratio}_${dataset}_${seed}.pkl \
    --model_file_name model_${known_cls_ratio}_${dataset}_${seed}_${suffix}.pt \
    --pretrained_nidmodel_file_name model_${known_cls_ratio}_${dataset}_${seed}_${pretrain_suffix}_epoch_best.pt \
    --result_dir ./${prefix}/${method}/results_${dataset}_${suffix}_kcr${known_cls_ratio} \
    --results_file_name results_${known_cls_ratio}_${dataset}_${seed}_${suffix}_FAST_TEST.csv  \
    --cl_loss_weight 1.0 \
    --semi_cl_loss_weight 1.0