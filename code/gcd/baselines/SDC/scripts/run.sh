set -o errexit
export CUDA_VISIBLE_DEVICES='7' 

for SEED in 0 1 2 3 4
do 
for DATASET in banking clinc stackoverflow
do
for KNOWN_CLS_RATIO in 0.25 0.5 0.75 
do
OPENBLAS_NUM_THREADS=32 python train.py \
    --dataset $DATASET \
    --known_cls_ratio $KNOWN_CLS_RATIO \
    --seed $SEED \
    --pretrain_dir 'sdc_pretrained_models' \
    --train_dir 'sdc_models'
done
done
done