#!/usr/bin bash
# conda activate scl
set -o errexit
export CUDA_VISIBLE_DEVICES='2'

for seed in $1
do
for dataset in news
do
for proportion in 0.75 0.5 0.25
do
    python experiment.py --dataset_name $dataset --proportion $proportion --seed $seed
done
done
done