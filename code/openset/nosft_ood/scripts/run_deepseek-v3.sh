set -o errexit
for dataset in clinc
do
for ratio in 0.25 0.5 0.75
do
for prompt_method in 'simple' 'analogy'
do
python main.py \
    --dataset $dataset \
    --mask_ratio $ratio \
    --llm_url https://openrouter.ai/api/v1 \
    --llm_name deepseek-v3:671b \
    --data_dir ../../data \
    --prompt_method $prompt_method \
    --num_threads 4
done
done
done