import os
import sys

import logging
import argparse
import sys
import os
import datetime
import pickle as pkl
import yaml

from utils import set_seed, load_yaml_config
from easydict import EasyDict
from dataloaders.base_data import BaseDataNew




def set_logger(args):

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # file_name = f"{args.method}_{args.dataset}_{args.known_cls_ratio}_{time}_{args.labeled_ratio}.log"
    file_name = f"alup_{args.dataset}_{args.known_cls_ratio}_{time}_{args.labeled_ratio}.log"
    
    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(args.log_dir, file_name))
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

# run.py

def parse_arguments():
    """
    定义所有可配置的参数，使其既能从YAML加载，也能被命令行覆盖。
    """
    parser = argparse.ArgumentParser()

    # --- 1. 核心任务与工作台对接参数 ---
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--dataset", type=str, default=None, help="The name of the dataset to train selected")
    parser.add_argument("--known_cls_ratio", type=float, default=None, help="The ratio of known classes")
    parser.add_argument("--labeled_ratio", type=float, default=None, help="The ratio of labeled samples in the training set")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for initialization")
    parser.add_argument("--fold_num", type=int, default=None, help="The total number of folds for cross validation") # <--- 新增这一行
    parser.add_argument("--fold_idx", type=int, default=None, help="The index of cross validation fold")
    parser.add_argument("--gpu_id", type=int, default=None, help="Which GPU to use")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length for tokenizer.")
    parser.add_argument("--method", type=str, default="ALUP", help="Method name to log in results.")

    # --- 2. 路径与目录参数 ---
    parser.add_argument("--data_dir", type=str, default=None, help="The input data dir")
    parser.add_argument("--bert_model", type=str, default=None, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_base_dir", type=str, default=None, help="Base directory for all outputs")
    parser.add_argument("--output_subdir", type=str, default=None, help="Subdirectory for this specific run's outputs")
    parser.add_argument("--pretrained_stage1_subdir", type=str, default=None, help="Subdirectory of the pretrained model from stage 1")
    
    # --- 3. 流程控制标志 ---
    parser.add_argument("--do_pretrain_and_contrastive", action="store_true", help="Run stage 1: pre-training and contrastive learning")
    parser.add_argument("--do_al_finetune", action="store_true", help="Run stage 2: active learning fine-tuning")
    parser.add_argument("--finish_pretrain", action="store_true", help="Finish running after stage 1 is complete")
    parser.add_argument("--save_results", action="store_true", help="Save final results for open intent detection")
    

    # --- 4. 训练超参数 ---
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Total number of training epochs to perform.")
    parser.add_argument("--num_pretrain_epochs", type=int, default=None, help="Total number of pre-training epochs to perform.")
    parser.add_argument("--pretrain_batch_size", type=int, default=None, help="Batch size for pre-training.")
    parser.add_argument("--train_batch_size", type=int, default=None, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Batch size for eval.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for contrastive/AL finetuning.")
    parser.add_argument("--lr_pre", type=float, default=None, help="Learning rate for pre-training.")
    parser.add_argument("--warmup_proportion", type=float, default=None, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--wait_patient", type=int, default=None, help="Patient epochs for pre-training early stopping.")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                    help="How many epochs with no sufficient improvement before stopping (contrastive & AL finetune).")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                    help="Minimum improvement (on monitored score) to reset patience.")
    parser.add_argument("--monitor_sum_metrics", action="store_true", default=True,
        help="If set, monitor ACC+ARI+NMI (default). Otherwise monitor ACC only.")

    
    # --- 5. ALUP 算法超参数 ---
    # 对比学习与数据增强
    parser.add_argument("--cluster_num_factor", type=float, default=None, help="The factor (magnification) of the number of clusters K.")
    parser.add_argument("--embed_feat_dim", type=int, default=None, help="Feature dimension of the BERT embeddings.")
    parser.add_argument("--head_feat_dim", type=int, default=None, help="Feature dimension of the projection head.")
    parser.add_argument("--topk", type=int, default=None, help="Number of nearest neighbors to search for.")
    parser.add_argument("--rtr_prob", type=float, default=None, help="Probability for random token replacement augmentation.")
    parser.add_argument("--update_per_epoch", type=int, default=None, help="Frequency (in epochs) of running the AL-LLM loop.")
    
    # 主动学习与LLM
    parser.add_argument("--llm_model_name", type=str, default=None, help="Name of the LLM for active labeling.")
    parser.add_argument("--api_base", type=str, default=None, help="API base URL for the LLM service.")
    parser.add_argument("--comparison_cluster_ratio", type=float, default=None, help="Ratio of nearest clusters to use in the LLM prompt.")
    parser.add_argument("--student_t_freedom", type=int, default=None, help="Degree of freedom for student-t distribution in uncertainty calculation.")
    parser.add_argument("--uncertainty_neighbour_num", type=int, default=None, help="Number of neighbors for uncertainty refinement.")
    parser.add_argument("--rho", type=float, default=None, help="Parameter for similarity score calculation in uncertainty refinement.")

    # 日志相关 (可以保留)
    parser.add_argument('--log_dir', type=str, default='logs', help="Logger directory.")
    parser.add_argument("--logger_name", type=str, default='Discovery', help="The name for the logger.")


    args = parser.parse_args()
    return args


def run():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser()
    command_args = parse_arguments()

    # 2. 加载并合并YAML配置
    if command_args.config:
        with open(command_args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # 合并通用配置
        apply_config_updates(command_args, yaml_config, parser)
        
        # 合并数据集专属配置
        if 'dataset_specific_configs' in yaml_config:
            dataset_configs = yaml_config['dataset_specific_configs'].get(command_args.dataset, {})
            apply_config_updates(command_args, dataset_configs, parser)

    args = EasyDict(vars(command_args))
    
    # 3. 动态构建输出和模型路径
    # 基础输出路径
    output_dir = os.path.join(args.output_base_dir, args.output_subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 为了兼容旧的save_results和save_model函数，我们动态设置这些属性
    args.output_dir = output_dir # 用于 save_model
    args.result_dir = output_dir # 用于 save_results
    args.results_file_name = 'results.csv' # 硬编码
    args.model_file_name = 'model' # 硬编码基础名称
    args.log_dir = os.path.join(output_dir, "logs")
    
    # 检查命令行是否传入了 pretrain 的 epoch 数
    if command_args.num_pretrain_epochs is not None:
        print(f"Overwriting num_pretrain_epochs from config file. Using {command_args.num_pretrain_epochs} epochs.")
        args.num_pretrain_epochs = command_args.num_pretrain_epochs
    
    # 检查命令行是否传入了 train 的 epoch 数
    if command_args.num_train_epochs is not None:
        print(f"Overwriting num_train_epochs from config file. Using {command_args.num_train_epochs} epochs.")
        args.num_train_epochs = command_args.num_train_epochs

    args.bert_model = './pretrained_models/bert-base-chinese' if args.dataset == 'ecdt' else args.bert_model
    # args.tokenizer = '../../pretrained_models/bert-base-chinese' if args.dataset == 'ecdt' else args.tokenizer

    logger = set_logger(args)

    logger.debug("="*30+" Params "+"="*30)
    for k in args.keys():
        logger.debug(f"{k}:\t{args[k]}")
    logger.debug("="*30+" End Params "+"="*30)

    set_seed(args.seed)

    logger.info('Data and Model Preparation...')

    data = BaseDataNew(args)
    args.num_labels = data.num_labels

    if args.do_pretrain_and_contrastive:
        from methods.alup.pretrain_manager import PretrainManager
        from methods.alup.manager import Manager

        logger.info('Pretrain Begin...')
        pretrain_manager = PretrainManager(args, data)
        pretrain_manager.train(args)
        
        # 这里的 finish_pretrain 逻辑可以保留或根据需要移除
        if args.finish_pretrain is not None and args.finish_pretrain:
            return
            
        manager = Manager(args, data, pretrained_model=pretrain_manager.model)
        manager.train(args, data)
        logger.info('Pretrain Finished...')

    elif args.do_al_finetune:
        from methods.alup.al_manager import ALManager
        
        # 动态构建第一阶段产出模型的完整路径
        model_filename = f"{args.model_file_name}_epoch_best.pt"
        stage1_model_path = os.path.join(args.output_base_dir, args.pretrained_stage1_subdir, model_filename)
        
        
        logger.info(f'AL Fine-tuning Begin, loading model from {stage1_model_path}...')
        
        # 使用简化的路径来实例化ALManager
        finetune_manager = ALManager(args, data, stage1_model_path)
        finetune_manager.al_finetune(args)

    

def apply_config_updates(args, config_dict, parser):
        """
        使用配置字典中的值更新 args 对象，同时进行类型转换。
        命令行中显式给出的参数不会被覆盖。
        """
        type_map = {action.dest: action.type for action in parser._actions}
        for key, value in config_dict.items():
            if f'--{key}' not in sys.argv and hasattr(args, key):
                expected_type = type_map.get(key)
                if expected_type and value is not None:
                    try:
                        value = expected_type(value)
                    except (TypeError, ValueError):
                        pass
                setattr(args, key, value)
    



if __name__ == '__main__':

    run()