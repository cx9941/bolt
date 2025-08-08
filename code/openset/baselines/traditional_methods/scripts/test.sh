set -o errexit
dataset_name=$1
rate=$2
backbone=$3
seed=$4

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

python train_ood.py --dataset_name $dataset_name --backbone $backbone  --rate $rate --seed $seed