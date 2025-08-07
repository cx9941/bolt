set -o errexit
sample_num=500

for seed in 0
do
for dataset_name in banking clinc stackoverflow
do
for ratio in 0.25 0.5 0.75
do
for model_name in 'gpt-4o-mini'
do
python main.py  --dataset_name $dataset_name --seed $seed --model_name $model_name --ratio $ratio  --sample_num $sample_num
done
done
done
done