set -o errexit
for dataset in clinc banking stackoverflow
do
for ratio in 0.25 0.5 0.75
do
for prompt_method in 'simple' 'analogy'
do
python main.py \
    --dataset $dataset \
    --mask_ratio $ratio \
    --llm_url http://localhost:8866 \
    --llm_name llama8b \
    --data_dir ../../data \
    --prompt_method $prompt_method \
    --num_threads 5
done
done
done