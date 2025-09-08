set -o errexit
backbone='Meta-Llama-3.1-8B-Instruct'

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
fold_num=5
seed=0
for fold_idx in 0 1 2 3 4
do
for dataset_name in 'banking' 'clinc' 'stackoverflow' 'hwu'
do
for rate in 0.25 0.5 0.75
do
python pretrain.py --dataset_name $dataset_name --backbone $backbone --rate $rate --seed $seed --gpu_id 0 --fold_idx $fold_idx --fold_num $fold_num
python test.py --dataset_name $dataset_name --backbone $backbone --rate $rate --seed $seed --gpu_id 0 --fold_idx $fold_idx --fold_num $fold_num
done
done
done