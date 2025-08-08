#!/usr/bin bash

for s in 0 1 2 3 4
do 
for dataset in banking clinc stackoverflow hwu
do
for known_cls_ratio in 0.25 0.5 0.75
do
    python main.py \
        --dataset $dataset \
        --known_cls_ratio $known_cls_ratio \
        --cluster_num_factor 1 \
        --seed $s \
        --gpu_id 1
done
done
done