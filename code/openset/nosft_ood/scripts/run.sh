set -o errexit
for dataset in news thucnews
do
for ratio in 0.25 0.5 0.75
do
python main.py \
    --dataset $dataset \
    --mask_ratio $ratio \
    --llm_url http://localhost:8864 \
    --llm_name qwen27B \
    --data_dir ../../data \

done
done