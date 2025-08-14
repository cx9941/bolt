from configs import args
import os
import copy
# if os.path.exists(args.metric_file):
#     exit()
# if os.path.exists(args.case_path + f'/OpenMax_preds.npy'):
#     exit()

import os
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

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import numpy as np
import logging

from model import Model
from load_dataset import train_dataset, dataset_in_test, dataset_out_test, collate_batch, loader_in_train, test_loader

def calculate_gcd_metrics(preds, golds, features, ood_scores, num_known_classes, num_novel_classes, ood_threshold=0.5):
    """
    计算一套完整的广义类别发现（GCD）指标。

    :param preds: 模型对已知类的原始预测
    :param golds: 全部样本的真实标签
    :param features: 全部样本的特征向量
    :param ood_scores: 全部样本的OOD异常分数 (归一化到0-1)
    :param num_known_classes: 已知类的数量
    :param num_novel_classes: 未知类的数量
    :param ood_threshold: 用于区分已知/未知的阈值
    :return: 包含所有GCD指标的字典
    """
    # 1. 创建掩码，区分真实/预测的已知与未知样本
    true_known_mask = golds < num_known_classes
    true_novel_mask = ~true_known_mask  # `~` 是布尔取反

    pred_novel_mask = ood_scores > ood_threshold
    pred_known_mask = ~pred_novel_mask

    # 2. 计算 K-ACC (已知类准确率)
    # 只评估那些被正确识别为“已知”的已知类样本
    k_acc_golds = golds[pred_known_mask & true_known_mask]
    k_acc_preds = preds[pred_known_mask & true_known_mask]
    k_acc = (k_acc_preds == k_acc_golds).sum() / len(k_acc_golds) if len(k_acc_golds) > 0 else 0.0

    # 3. 对预测出的未知样本进行聚类，并计算 N-ACC (未知类准确率)
    pred_novel_features = features[pred_novel_mask]
    pred_novel_golds = golds[pred_novel_mask]

    n_acc = 0.0 # 初始化
    if pred_novel_features.shape[0] > 0 and num_novel_classes > 0:
        # 使用K-Means聚类
        kmeans = KMeans(n_clusters=num_novel_classes, random_state=42, n_init='auto')
        pred_novel_clusters = kmeans.fit_predict(pred_novel_features)

        # 使用匈牙利算法进行最优匹配
        cost_matrix = np.zeros((num_novel_classes, num_novel_classes))
        # 真实的新类别标签从 num_known_classes 开始
        for i in range(num_novel_classes):
            for j in range(num_novel_classes):
                true_label_j = num_known_classes + j
                mask = (pred_novel_clusters == i) & (pred_novel_golds == true_label_j)
                cost_matrix[i, j] = -np.sum(mask)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 计算匹配后的准确率
        n_acc = -cost_matrix[row_ind, col_ind].sum() / len(pred_novel_golds)

        # 为了计算总体指标，需要构建最终的预测标签
        # 创建一个从预测簇到重映射后标签的映射表
        remap = {pred_cluster: true_cluster_offset + num_known_classes for pred_cluster, true_cluster_offset in zip(row_ind, col_ind)}
        final_preds = np.copy(preds)
        final_preds[pred_known_mask] = preds[pred_known_mask] # 已知类的预测不变
        # 对预测为未知的样本，填入重映射后的聚类标签
        # 必须先用一个临时数组来存储聚类结果
        temp_novel_preds = np.full_like(pred_novel_clusters, -1)
        for k, v in remap.items():
            temp_novel_preds[pred_novel_clusters == k] = v
        final_preds[pred_novel_mask] = temp_novel_preds
    else:
        # 如果没有样本被预测为novel，则无法计算聚类，直接使用已知类预测
        final_preds = np.copy(preds)
        final_preds[pred_novel_mask] = -1 # 标记为未知，但没有具体类别


    # 4. 计算剩余的总体指标
    overall_acc = (final_preds == golds).mean()
    h_score = 2 * (k_acc * n_acc) / (k_acc + n_acc) if (k_acc + n_acc) > 0 else 0.0
    nmi = normalized_mutual_info_score(golds, final_preds)
    ari = adjusted_rand_score(golds, final_preds)

    return {
        "Overall ACC": overall_acc,
        "K-ACC": k_acc,
        "N-ACC": n_acc,
        "H-Score": h_score,
        "NMI": nmi,
        "ARI": ari,
    }

if __name__ == '__main__':
    fix_random_seed(args.seed)
    model = Model(args).to(args.device)
    with torch.no_grad():
        model.eval()
        if not os.path.exists(f'{args.vector_path}/logits.npy'):
            preds = []
            golds = []
            logits = []
            features = []
            for batch in tqdm(test_loader):
                y = batch['labels'].to(args.device)
                batch = {i:v.to(args.device) for i,v in batch.items() if i!= 'labels'}
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
            logits = np.load(f'{args.vector_path}/logits.npy')
            preds = np.load(f'{args.vector_path}/preds.npy')
            golds = np.load(f'{args.vector_path}/golds.npy')
            features = np.load(f'{args.vector_path}/features.npy')

        ID_metrics = custom_metrics(preds, golds)
        
        logging.info(f"Test Accuracy: {ID_metrics['macro avg']}")

    logging.info("STAGE 2: Creating OOD Detectors")

    detectors = {}
    detectors["TemperatureScaling"] = TemperatureScaling(model)
    detectors["LogitNorm"] = LogitNorm(model)
    detectors["OpenMax"] = OpenMax(model)
    detectors["Entropy"] = Entropy(model)
    detectors["Mahalanobis"] = Mahalanobis(model.features, eps=0.0)
    detectors["KLMatching"] = KLMatching(model)
    detectors["MaxSoftmax"] = MaxSoftmax(model)
    detectors["EnergyBased"] = EnergyBased(model)
    detectors["MaxLogit"] = MaxLogit(model)

    # detectors["SHE"] = SHE(model=model.features, head=model.fc)
    # detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)

    logging.info(f"> Fitting {len(detectors)} detectors")

    for name, detector in detectors.items():
        logging.info(f"--> Fitting {name}")
        detector.fit(loader_in_train, device=args.device)

    logging.info(f"STAGE 3: Evaluating {len(detectors)} detectors.")
    results = []

    with torch.no_grad():
        for detector_name, detector in detectors.items():
            logging.info(f"> Evaluating {detector_name}")
            metrics = OODMetrics()
            scores = []
            for batch in tqdm(test_loader):
                y = batch['labels'].to(args.device)
                batch = {i:v.to(args.device) for i,v in batch.items() if i!= 'labels'}
                score = detector(batch)
                metrics.update(score, y)
                scores.append(score)

            r = {"Detector": detector_name}
            r.update(metrics.compute())
            scores = torch.concat(scores).detach().cpu().numpy()
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

            if detector_name == 'Vim':
                # 对于Vim，我们同样调用指标计算函数
                # 注意：
                # 1. 'preds' 参数必须是模型原始的预测，在它被Vim的逻辑覆盖之前。
                # 2. 'ood_scores' 参数使用Vim detector计算出的norm_scores。
                gcd_metrics = calculate_gcd_metrics(
                    preds=preds,
                    golds=golds,
                    features=features,
                    ood_scores=norm_scores,
                    num_known_classes=args.num_known_classes,
                    num_novel_classes=args.num_novel_classes,
                    ood_threshold=0.5
                )
                r.update(gcd_metrics)
                # 保留Vim独特的预测生成逻辑，用于后续保存
                new_logits = np.concatenate([logits, scores.reshape(scores.shape[0], 1)], axis=1)
                new_preds = new_logits.argmax(axis=1)
                # 将预测为虚拟logit（最后一个）的样本标签设为-1
                new_preds[new_preds==logits.shape[-1]] = -1
                final_preds = copy.deepcopy(new_preds)
                # preds = new_preds
                # r.update(custom_metrics(preds, golds, norm_scores))
                # final_preds = copy.deepcopy(preds)

            else:
                # r.update(custom_metrics(preds, golds, norm_scores))
                # 确保 args 中有 num_known_classes 和 num_novel_classes
                gcd_metrics = calculate_gcd_metrics(
                    preds=preds,
                    golds=golds,
                    features=features,
                    ood_scores=norm_scores,
                    num_known_classes=args.num_known_classes,
                    num_novel_classes=args.num_novel_classes,
                    ood_threshold=0.5
                )
                r.update(gcd_metrics)
                # 生成用于保存的最终预测
                final_preds = copy.deepcopy(preds)
                final_preds[norm_scores > 0.5] = -1

            results.append(r)
            np.save(args.case_path + f'/{detector_name}_preds.npy', final_preds)
            np.save(args.case_path + f'/{detector_name}_golds.npy', golds)
            np.save(args.case_path + f'/{detector_name}_features.npy', features)

    df = pd.DataFrame(results)
    mean_scores = df.groupby(["Detector"]).mean() * 100
    logging.info(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))
    # mean_scores.to_csv(args.metric_file,  sep='\t')
    mean_scores.to_csv(args.metric_file, sep=',', float_format="%.2f")