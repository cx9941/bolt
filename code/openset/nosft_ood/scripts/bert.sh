for dataset in banking clinc stackoverflow ele news thucnews
do
for ratio in 0.25 0.5 0.75
do
python bert.py --dataset $dataset --mask_ratio $ratio --gpu_id 0
done
done