set -o errexit
# conda activate tf2
export CUDA_VISIBLE_DEVICES='2'

for v_seed in 0 1 2 3 4
do
    for v_dataset in atis
    do
        for ratio in 0.25 0.5 0.75
        do
            python run_main.py json/${v_dataset}/${ratio}/${v_seed}.json
        done
    done
done
