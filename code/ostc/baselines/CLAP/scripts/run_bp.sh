set -o errexit

export CUDA_VISIBLE_DEVICES='3'

dataset_name='banking'
ratio=0.25


s=0
adbes=train
k=5
te=1

pretrain_model_path=../pretrained_model
pretrain_dir=../save_${dataset_name}_${ratio}_model

# cd finetune
# python run_kccl.py --dataset ${dataset_name} --dataset_mode random --known_cls_ratio $ratio --pretrain_loss_type 1 --model_type 1.1 --le_random 1 --kccl_k $k --temperature $te --KCCL_LOSS_LAMBDA 0.25 --CE_LOSS_LAMBDA 0.75 --LMCL_LOSS_LAMBDA 1.0 --seed $s --seed_data $s --adbes_type $adbes  --save_path_suffix cos --ks 1 --km 0 --s_v 1 --m 0 --neg_margin 0 --neg_m 0.35 --loss_metric 0 --neg_method 3 --neg_num 1 --centroids 1 --poolout_norm 0 --centroids_norm 0 --softplus 0 --metric_type 1 --kccl_euc 0 --c_m 2 --t_a 0.35 --eval_metric f1 --optimizer_lr 0 --freeze_bert_parameters True --num_pretrain_epochs 1 --num_train_epochs 1 --pretrain_lr 2e-5 --lr_boundary 0.05 --train_batch_size 32 --eval_batch_size 64

cd boundary_adjustment

pretrain_dir=../save_${dataset_name}_${ratio}_model
# pretrain_model_path=../pretrained_model/bert-base-uncased
save_path=../save_${dataset_name}_${ratio}_boundary

python run_adbes.py --dataset ${dataset_name} --known_cls_ratio $ratio --labeled_ratio 1.0 --seed 0 --freeze_bert_parameters --gpu_id 0 --save_model --pretrain_dir $pretrain_dir --save_results_path $save_path --n 0 --do_bert_output_norm --write_results --train_from_scratch