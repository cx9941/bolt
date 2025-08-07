cd finetune
export s=0
export adbes=train
export k=5
export te=1
export pretrain_model_path="../pretrained_model"
export pretrain_dir='../save_banking_75_model'
python run_kccl.py --dataset banking --dataset_mode random --known_cls_ratio 0.75 --pretrain_loss_type 1 --model_type 1.1 --le_random 1 --kccl_k $k --temperature $te --KCCL_LOSS_LAMBDA 0.25 --CE_LOSS_LAMBDA 0.75 --LMCL_LOSS_LAMBDA 1.0 --seed $s --seed_data $s --adbes_type $adbes --save_path_suffix cos --ks 1 --km 0 --s_v 1 --m 0 --neg_margin 0 --neg_m 0.35 --loss_metric 0 --neg_method 3 --neg_num 1 --centroids 0 --poolout_norm 0 --centroids_norm 0 --softplus 0 --metric_type 1 --kccl_euc 0 --c_m 2 --t_a 0.35 --eval_metric f1 --optimizer_lr 0 --num_pretrain_epochs 100 --num_train_epochs 100 --pretrain_lr 2e-5 --lr_boundary 0.05 --train_batch_size 128 --eval_batch_size 256