import numpy as np
import os
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

def save_results(method_name, args, results, num_labels):
    """
    将实验结果和超参数保存到全局的 CSV 文件中。
    采用 Pandas 实现，功能更强大。
    
    :param method_name: 本次运行的方法名，例如 'TAN'
    :param args: 包含所有超参数的 Namespace 对象
    :param results: 包含评估指标的字典
    :param num_labels: 最终聚类的簇数 K
    """
    # 1. 定义要保存的实验配置
    # 注意：这里我们假设 args 对象中包含了所有需要的参数
    # 你可以根据需要从 args 中添加更多你想记录的参数
    config_to_save = {
        'method': method_name,
        'dataset': args.dataset,
        'known_cls_ratio': args.known_cls_ratio,
        'labeled_ratio': args.labeled_ratio,
        'cluster_num_factor': args.cluster_num_factor,
        'seed': args.seed,
        'K': num_labels
    }
    
    # 2. 合并配置和结果
    full_results = {**config_to_save, **results}
    
    # 3. 定义结果文件路径
    # 我们将结果保存在一个固定的 outputs 文件夹下
    # save_path = "outputs"
    save_path = args.output_dir 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    results_file = os.path.join(save_path, "results.csv")
    
    # 4. 使用 Pandas 写入 CSV
    # 将字典转换为 DataFrame
    new_df = pd.DataFrame([full_results])
    
    if not os.path.exists(results_file):
        # 如果文件不存在，直接写入（包含表头）
        new_df.to_csv(results_file, index=False)
    else:
        # 如果文件存在，追加写入（不含表头）
        new_df.to_csv(results_file, mode='a', header=False, index=False)
        
    print(f"Results successfully saved to {results_file}")
    
    # (可选) 打印整个结果表格
    # print("\n--- Cumulative Results ---")
    # all_data_df = pd.read_csv(results_file)
    # print(all_data_df)
    # print("--------------------------\n")

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred, known_lab):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    ind_map = {j: i for i, j in ind}
    
    old_acc = 0
    total_old_instances = 0
    for i in known_lab:
        # 增加一个检查，防止 ind_map 中没有某个已知类别的映射（当该类在测试集中没有样本时）
        if i in ind_map:
            old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    
    # 防止 total_old_instances 为 0 导致除零错误
    if total_old_instances == 0:
        old_acc = 0.0
    else:
        old_acc /= total_old_instances
    
    new_acc = 0
    total_new_instances = 0
    for i in range(len(np.unique(y_true))):
        if i not in known_lab:
            # 同样增加检查
            if i in ind_map:
                new_acc += w[ind_map[i], i]
            total_new_instances += sum(w[:, i])

    # 防止 total_new_instances 为 0 导致除零错误
    if total_new_instances == 0:
        new_acc = 0.0
    else:
        new_acc /= total_new_instances

    # 防止 old_acc 或 new_acc 为 0 导致除零错误
    if old_acc == 0 or new_acc == 0:
        h_score = 0.0
    else:
        h_score = 2*old_acc*new_acc / (old_acc + new_acc)

    metrics = {
        'ACC': round(acc*100, 2),
        'H-Score': round(h_score*100, 2),
        'K-ACC': round(old_acc*100, 2),
        'N-ACC': round(new_acc*100, 2),
    }

    return metrics

def clustering_score(y_true, y_pred, known_lab):
    metrics = clustering_accuracy_score(y_true, y_pred, known_lab)
    metrics['ARI'] = round(adjusted_rand_score(y_true, y_pred)*100, 2)
    metrics['NMI'] = round(normalized_mutual_info_score(y_true, y_pred)*100, 2)
    return metrics