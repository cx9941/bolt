# train_ood.py (最终修正版)

import os
import copy
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve
from src.pytorch_ood.detector import (
    EnergyBased,
    Entropy,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
    OpenMax,
    TemperatureScaling,
    ASH,
    SHE,
    LogitNorm
)
from src.pytorch_ood.utils import OODMetrics, fix_random_seed, custom_metrics
import numpy as np
import logging
import sys
import yaml
import json

from model import Model
from load_dataset import load_and_prepare_datasets
# from configs import get_plm_ood_config
from configs import create_parser, finalize_config

def run_ood_evaluation(args):
    """
    OOD 评测的主函数，包含了所有的核心逻辑。
    """
    # --- 改动 2: 在函数开头，调用数据加载函数 ---
    logging.info("Loading and preparing datasets...")
    # 调用函数，获取一个包含所有数据对象的字典
    data = load_and_prepare_datasets(args)
    tokenizer = data['tokenizer']
    # 从字典中解包出我们需要的变量
    test_loader = data['test_loader']
    loader_in_train = data['loader_in_train']
    logging.info("Datasets loaded successfully.")
    # --- 改动结束 ---

    fix_random_seed(args.seed)
    model = Model(args, tokenizer=tokenizer).to(args.device)
    
    with torch.no_grad():
        model.eval()
        if not os.path.exists(f'{args.vector_path}/logits.npy'):
            logging.info("Calculating logits, predictions, and features...")
            preds, golds, logits, features = [], [], [], []
            # --- 改动 3: 使用上面解包出来的局部变量 test_loader ---
            for batch in tqdm(test_loader, desc="Inference"):
                y = batch['labels'].to(args.device)
                batch = {i: v.to(args.device) for i, v in batch.items() if i != 'labels'}
                logit = model(batch)
                feature = model.features(batch)
                pred = logit.max(dim=1).indices
                
                preds.append(pred)
                logits.append(logit)
                golds.append(y)
                features.append(feature)
            
            logits = torch.concat(logits).detach().to(torch.float32).cpu().numpy()
            preds = torch.concat(preds).detach().cpu().numpy()
            golds = torch.concat(golds).detach().cpu().numpy()
            features = torch.concat(features).detach().to(torch.float32).cpu().numpy()

            np.save(f'{args.vector_path}/logits.npy', logits)
            np.save(f'{args.vector_path}/preds.npy', preds)
            np.save(f'{args.vector_path}/golds.npy', golds)
            np.save(f'{args.vector_path}/features.npy', features)
        else:
            logging.info("Loading pre-calculated logits, predictions, and features...")
            logits = np.load(f'{args.vector_path}/logits.npy')
            preds = np.load(f'{args.vector_path}/preds.npy')
            golds = np.load(f'{args.vector_path}/golds.npy')
            features = np.load(f'{args.vector_path}/features.npy')

        ID_metrics = custom_metrics(preds, golds)
        logging.info(f"Test Accuracy: {ID_metrics['macro avg']}")

    logging.info("STAGE 2: Creating OOD Detectors")
    detectors = {
        "TemperatureScaling": TemperatureScaling(model),
        "LogitNorm": LogitNorm(model),
        "OpenMax": OpenMax(model),
        "Entropy": Entropy(model),
        "Mahalanobis": Mahalanobis(model.features, eps=0.0),
        "KLMatching": KLMatching(model),
        "MaxSoftmax": MaxSoftmax(model),
        "EnergyBased": EnergyBased(model),
        "MaxLogit": MaxLogit(model)
    }

    logging.info(f"> Fitting {len(detectors)} detectors")
    for name, detector in detectors.items():
        logging.info(f"--> Fitting {name}")
        # --- 改动 4: 使用上面解包出来的局部变量 loader_in_train ---
        detector.fit(loader_in_train, device=args.device)

    logging.info(f"STAGE 3: Evaluating {len(detectors)} detectors.")
    results = []
    with torch.no_grad():
        for detector_name, detector in detectors.items():
            logging.info(f"> Evaluating {detector_name}")
            metrics = OODMetrics()
            scores = []
            # --- 改动 5: 再次使用 test_loader ---
            for batch in tqdm(test_loader, desc=f"Evaluating {detector_name}"):
                y = batch['labels'].to(args.device)
                batch = {i: v.to(args.device) for i, v in batch.items() if i != 'labels'}
                score = detector(batch)
                metrics.update(score, y)
                scores.append(score)
            
            r = {"Detector": detector_name}
            r.update(metrics.compute())
            scores = torch.concat(scores).detach().cpu().numpy()
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

            if detector_name == 'Vim':
                new_logits = np.concatenate([logits, scores.reshape(scores.shape[0], 1)], axis=1)
                new_preds = new_logits.argmax(axis=1)
                new_preds[new_preds==logits.shape[-1]] = -1
                preds = new_preds
                r.update(custom_metrics(preds, golds, norm_scores))
                final_preds = copy.deepcopy(preds)
            else:
                r.update(custom_metrics(preds, golds, norm_scores))
                final_preds = copy.deepcopy(preds)
                final_preds[norm_scores > 0.5] = -1

            results.append(r)
            np.save(args.case_path + f'/{detector_name}_preds.npy', final_preds)
            np.save(args.case_path + f'/{detector_name}_golds.npy', golds)
            np.save(args.case_path + f'/{detector_name}_features.npy', features)

    df = pd.DataFrame(results)
    mean_scores = df.groupby(["Detector"]).mean() * 100
    mean_scores['args'] = json.dumps(vars(args), ensure_ascii=False)
    logging.info(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))
    mean_scores.to_csv(args.metric_file, sep='\t')

def apply_config_updates(args, config_dict, parser):
    """
    使用配置字典中的值更新 args 对象，同时进行类型转换。
    命令行中显式给出的参数不会被覆盖。
    """
    # 创建一个从 dest 到 action.type 的映射
    type_map = {action.dest: action.type for action in parser._actions}

    for key, value in config_dict.items():
        # 检查参数是否在命令行中被用户显式提供
        if f'--{key}' not in sys.argv and hasattr(args, key):
            # 获取该参数预期的类型
            expected_type = type_map.get(key)
            # 如果有预期类型且值不为None，则进行类型转换
            if expected_type and value is not None:
                value = expected_type(value)
            setattr(args, key, value)

if __name__ == '__main__':
    parser = create_parser()
    parser.add_argument("--config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        apply_config_updates(args, yaml_config, parser)
        
    config_args = finalize_config(args)
    
    run_ood_evaluation(config_args)