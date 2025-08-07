#!/usr/bin bash
# conda activate scl
set -o errexit

for seed in 0 1 2 3 4
do
for dataset in banking clinc stackoverflow ele thucnews snips
do
for proportion in 0.25 0.5 0.75
do
    python experiment.py --dataset_name $dataset --proportion $proportion --seed $seed --gpu_id "1"
done
done
done