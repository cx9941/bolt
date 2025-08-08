#!/usr/bin bash
for d in 'banking' 'clinc' 'stackoverflow' 'hwu'
do 
for known_cls_ratio in 0.25 0.5 0.75
do
python TAN.py \
    --dataset $d \
    --gpu_id 4 \
    --known_cls_ratio $known_cls_ratio \
    --cluster_num_factor 1 \
    --seed 0 \
    --freeze_bert_parameters \
    --pretrain_dir ckpts/premodel_${d}_${known_cls_ratio}_${seed} \
    --save_model \
    --pretrain
done
done