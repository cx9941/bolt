set -o errexit

export CUDA_VISIBLE_DEVICES='1'

s=0
adbes=train
k=5
te=1
cd boundary_adjustment

for seed in 0 1 2 3 4
do
for dataset_name in atis stackoverflow banking ele news clinc snips thucnews
do
for ratio in 0.25 0.5 0.75
do

pretrain_dir=../outputs/save_${dataset_name}_${ratio}_model_${seed}
save_path=../outputs/save_${dataset_name}_${ratio}_boundary_${seed}

python run_adbes.py --dataset ${dataset_name} --known_cls_ratio $ratio --labeled_ratio 1.0 --seed $seed --freeze_bert_parameters --gpu_id 0 --save_model --pretrain_dir $pretrain_dir --save_results_path $save_path --n 0 --do_bert_output_norm --write_results --train_from_scratch

done
done
done