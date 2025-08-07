import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='banking', type=str)
parser.add_argument('--model_name', default='deepseek-chat', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--ratio', default=0.25, type=float)
parser.add_argument('--sample_num', default=500, type=float)
parser.add_argument('--max_workers', default=10, type=float)
args = parser.parse_args()


args.input_dir_name = f"seed{args.seed}"

args.output_dir = f"outputs/{args.dataset_name}/{args.ratio}/{args.model_name}/{args.input_dir_name}"
args.input_dir = f"inputs/{args.dataset_name}/{args.ratio}/{args.model_name}/{args.input_dir_name}"

args.train_taxnomy_path = f"../../data/{args.dataset_name}/{args.dataset_name}_{args.ratio}/train_taxnomy.json"

args.input_path = f"../../data/{args.dataset_name}/origin_data/test.tsv"
args.known_label = f"../../data/{args.dataset_name}/{args.dataset_name}_{args.ratio}/idx.txt"

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.input_dir, exist_ok=True)