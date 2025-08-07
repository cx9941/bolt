set -o errexit
export CUDA_VISIBLE_DEVICES='3'
for seed in 0 1 2 3 4
do
for dataset_name in 'ele' 'news'
do
for rate in 0.25 0.5 0.75
do
python code/run.py ${dataset_name}_${rate} $seed
done
done
done