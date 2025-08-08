import argparse
import os
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="stackoverflow", type=str)
parser.add_argument("--data_dir", default="../../../data", type=str)
parser.add_argument("--reg_loss", default="npo", type=str, choices=['normal', 'vos', 'npo'])
parser.add_argument("--rate", default=0.25, type=float)
parser.add_argument("--labeled_ratio", default=0.1, type=float)
parser.add_argument("--n_epochs", default=3, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--fold_idx", default=0, type=int)
parser.add_argument("--fold_num", default=5, type=int)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--root", default="data", type=str)
parser.add_argument("--output_dir", default="outputs", type=str)
parser.add_argument("--backbone", default="Meta-Llama-3.1-8B-Instruct", type=str)
parser.add_argument("--lr", default=5e-5, type=float)
parser.add_argument("--train_batch_size", default=16, type=int)
parser.add_argument("--eval_batch_size", default=32, type=int)
args = parser.parse_args()

args.data_identity = f"{args.dataset_name}_{args.labeled_ratio}_{args.rate}_{args.fold_num}_{args.fold_idx}"

args.metric_dir = f"{args.output_dir}/{args.reg_loss}/{args.dataset_name}/{args.labeled_ratio}/{args.data_identity}{args.backbone}_{args.seed}/metrics"
args.log_dir = f"{args.output_dir}/{args.reg_loss}/{args.dataset_name}/{args.labeled_ratio}/{args.data_identity}{args.backbone}_{args.seed}/logs"
args.checkpoint_path = f"{args.output_dir}/{args.reg_loss}/{args.dataset_name}/{args.labeled_ratio}/{args.data_identity}{args.backbone}_{args.seed}"
args.case_path = f"{args.output_dir}/{args.reg_loss}/{args.dataset_name}/{args.labeled_ratio}/{args.data_identity}{args.backbone}_{args.seed}/case_study"
args.vector_path = f"{args.output_dir}/{args.reg_loss}/{args.dataset_name}/{args.labeled_ratio}/{args.data_identity}{args.backbone}_{args.seed}/case_study"

if not os.path.exists(args.metric_dir):
    os.makedirs(args.metric_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
if not os.path.exists(args.case_path):
    os.makedirs(args.case_path)
if not os.path.exists(args.vector_path):
    os.makedirs(args.vector_path)

args.metric_file = f"{args.metric_dir}/epoch_{args.n_epochs}_seed_{args.seed}.csv"

args.model_path = f"../../../pretrained_models/{args.backbone}"

# 配置日志记录器
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
    handlers=[
        logging.FileHandler(f"{args.log_dir}/epoch_{args.n_epochs}_seed_{args.seed}.log", mode="w"),  # 将日志保存到文件
        logging.StreamHandler()  # 将日志输出到控制台
    ]
)

logging.info(args)