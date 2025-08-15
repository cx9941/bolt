import argparse
import os
import logging

def get_plm_ood_config():
    parser = argparse.ArgumentParser()
    
    # --- 基础参数定义 (YAML中的同名参数会覆盖这里的default值) ---
    parser.add_argument("--dataset_name", default="stackoverflow", type=str)
    # 改动1：将默认路径修正为相对于项目根目录
    parser.add_argument("--data_dir", default="./data", type=str) 
    parser.add_argument("--reg_loss", default="npo", type=str, choices=['normal', 'vos', 'npo'])
    parser.add_argument("--rate", default=0.25, type=float)
    parser.add_argument("--labeled_ratio", default=1.0, type=float)
    parser.add_argument("--n_epochs", default=20, type=int)
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
    
    # 改动2：添加 model_path 参数，使其可以从YAML接收
    parser.add_argument("--model_path", default=None, type=str)
    
    # 改动3：使用 parse_args()，因为我们的工作流能确保参数匹配
    args = parser.parse_args()

    # --- 派生参数计算 (动态生成路径) ---
    args.data_identity = f"{args.dataset_name}_{args.labeled_ratio}_{args.rate}_{args.fold_num}_{args.fold_idx}"
    run_identity = f"{args.data_identity}{args.backbone}_{args.seed}"
    
    base_path = os.path.join(args.output_dir, args.reg_loss, args.dataset_name, str(args.labeled_ratio), run_identity)

    args.metric_dir = os.path.join(base_path, "metrics")
    args.log_dir = os.path.join(base_path, "logs")
    args.checkpoint_path = base_path
    args.case_path = os.path.join(base_path, "case_study")
    args.vector_path = os.path.join(base_path, "case_study")

    # 创建目录
    os.makedirs(args.metric_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.case_path, exist_ok=True)

    args.metric_file = f"{args.metric_dir}/epoch_{args.n_epochs}_seed_{args.seed}.csv"

    # 改动4：最关键的改动，优先使用YAML中的model_path，如果没有才使用备用路径
    if args.model_path is None:
        logging.warning("model_path not specified in YAML, generating a fallback path. It is recommended to specify this in the YAML file.")
        # 备用路径也必须是相对于项目根目录
        args.model_path = f"./pretrained_models/{args.backbone}"
    else:
        logging.info(f"Using model_path specified in config: {args.model_path}")

    # --- 日志设置 ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.log_dir}/epoch_{args.n_epochs}_seed_{args.seed}.log", mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )

    logging.info(f"Final configuration: {args}")
    
    return args