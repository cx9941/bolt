#!/usr/bin bash

for s in 0 1 2 3 4
do 
for dataset in banking clinc stackoverflow hwu ecdt mcid
do
for known_cls_ratio in 0.25 0.5 0.75
do
    python DeepAligned.py \
        --dataset $dataset \
        --known_cls_ratio $known_cls_ratio \
        --cluster_num_factor 1 \
        --labeled_ratio 0.1 \
        --seed $s \
        --gpu_id 3
done
done
done