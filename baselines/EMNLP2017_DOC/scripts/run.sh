# conda activate tf2
set -o errexit

for seed in 4
do
    for dataset_name in atis reuters banking clinc ele mcid news reuters snips stackoverflow thucnews
    do
        for ratio in 0.25 0.5 0.75
        do
            python DOC.py --dataset_name $dataset_name --seed $seed --ratio $ratio --gpu_id "2" --epochs 20
        done
    done
done
