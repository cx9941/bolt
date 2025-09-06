set -o errexit
backbone='Meta-Llama-3.1-8B-Instruct'

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

for seed in 0 1 2 3
do
for dataset_name in 'banking' 'clinc' 'stackoverflow' 'hwu'
do
for rate in 0.25 0.5 0.75
do
python pretrain.py --dataset_name $dataset_name --backbone $backbone --rate $rate --seed $seed --gpu_id 0
python test.py --dataset_name $dataset_name --backbone $backbone --rate $rate --seed $seed --gpu_id 0
done
done
done