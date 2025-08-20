set -o errexit
export CUDA_VISIBLE_DEVICES='4' 

for SEED in 0 1 2 3 4
do 
for DATASET in mcid hwu ecdt banking clinc stackoverflow
do
for KNOWN_CLS_RATIO in 0.25 0.5 0.75 
do
OPENBLAS_NUM_THREADS=32 python baseline.py \
    --dataset $DATASET \
    --known_cls_ratio $KNOWN_CLS_RATIO \
    --seed $SEED \
    --pretrain_dir 'baseline_models'
done
done
done