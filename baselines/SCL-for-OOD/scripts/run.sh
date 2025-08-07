#!/usr/bin bash
set -o errexit
export CUDA_VISIBLE_DEVICES='2'

for seed in 0 1 2 3 4
do
for dataset in atis
do
for proportion in 25 50 75
do
    python train.py --dataset $dataset --proportion $proportion --mode both --setting gda lof msp --experiment_No vallian --ind_pre_epoches 30 --supcont_pre_epoches 30 --norm_coef 0.1 --cuda --seed $seed
    # python train.py --dataset $dataset --proportion $proportion --mode both --setting gda lof msp --experiment_No vallian --ind_pre_epoches 30 --supcont_pre_epoches 30 --norm_coef 0.1 --cuda --sup_cont --seed $seed
done
done
done