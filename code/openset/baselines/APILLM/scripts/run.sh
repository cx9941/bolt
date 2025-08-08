set -o errexit

max_workers=$1
sample_num=50

for seed in 0
do
for dataset_name in banking clinc stackoverflow
do
for ratio in 0.25 0.5 0.75
do
for model_name in 'deepseek-chat'
do
python main.py  --dataset_name $dataset_name --seed $seed --model_name $model_name --sample_num $sample_num --ratio $ratio --max_workers $max_workers
done
done
done
done