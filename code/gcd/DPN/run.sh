# #!/usr/bin bash

# OPENBLAS_NUM_THREADS=16 python DPN.py \
#     --dataset stackoverflow \
#     --known_cls_ratio 0.75 \
#     --cluster_num_factor 1 \
#     --seed 0 \
#     --freeze_bert_parameters \
#     --gpu_id 0 \
#     --num_pretrain_epochs 1

#!/usr/bin bash

for s in 0 1 2 3
do 
for dataset in stackoverflow banking clinc hwu ecdt mcid
do
for known_cls_ratio in 0.75 0.25 0.5
do
    OPENBLAS_NUM_THREADS=16 python DPN.py \
        --dataset $dataset \
        --known_cls_ratio $known_cls_ratio \
        --cluster_num_factor 1 \
        --labeled_ratio 0.1 \
        --seed $s \
        --gpu_id 2 \
        --freeze_bert_parameters \
        --pretrain 
done
done
done