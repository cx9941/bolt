from sklearn.metrics import classification_report
import numpy as np
import os
import re

def compute_metrics(eval_predictions):
    preds, golds = eval_predictions
    preds = np.argmax(preds, axis=1)
    metrics = classification_report(preds, golds, output_dict=True)
    metrics['macro avg'].update({'accuracy': metrics['accuracy']})
    return metrics['macro avg']

def get_best_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if re.match(r'checkpoint-\d+', d)]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]), reverse=False)
    latest_checkpoint = os.path.join(output_dir, checkpoints[0])
    return latest_checkpoint

import copy
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import sklearn.metrics
from transformers import AutoTokenizer
import torch
import sklearn
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

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
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances
    
    new_acc = 0
    total_new_instances = 0
    for i in range(len(np.unique(y_true))):
        if i not in known_lab:
            new_acc += w[ind_map[i], i]
            total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

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
