import argparse
import os
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="stackoverflow", type=str)
parser.add_argument("--rate", default=0.25, type=float)
parser.add_argument("--n_epochs", default=10, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--root", default="data", type=str)
parser.add_argument("--output_dir", default="outputs", type=str)
parser.add_argument("--backbone", default="llama31_8b", type=str)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--train_batch_size", default=16, type=int)
parser.add_argument("--eval_batch_size", default=128, type=int)
args = parser.parse_args()

args.metric_dir = f"{args.output_dir}/{args.dataset_name}/{args.dataset_name}_{args.rate}/{args.backbone}_{args.seed}/metrics"
args.log_dir = f"{args.output_dir}/{args.dataset_name}/{args.dataset_name}_{args.rate}/{args.backbone}_{args.seed}/logs"
args.checkpoint_path = f"{args.output_dir}/{args.dataset_name}/{args.dataset_name}_{args.rate}/{args.backbone}_{args.seed}"
args.case_path = f"{args.output_dir}/{args.dataset_name}/{args.dataset_name}_{args.rate}/{args.backbone}_{args.seed}/case_study"

if not os.path.exists(args.metric_dir):
    os.makedirs(args.metric_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
if not os.path.exists(args.case_path):
    os.makedirs(args.case_path)

args.metric_file = f"{args.metric_dir}/epoch_{args.n_epochs}_seed_{args.seed}.csv"

model_path_map = {
    "gru": None,
    "bert": "../../llms/bert-base-uncased" if args.dataset_name != 'thucnews' else '../../llms/bert-base-chinese',
    "roberta": "../../llms/roberta-base",
    "t5": "../../llms/t5",
    "llama31_8b": "../../llms/Meta-Llama-3.1-8B-Instruct"
}
args.model_path = model_path_map[args.backbone]

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