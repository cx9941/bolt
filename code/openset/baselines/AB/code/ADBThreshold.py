# -*- coding: utf-8 -*-
"""
ADBThreshold (NumPy-only)
完全去掉 TensorFlow 依赖，避免在 RTX 5090/Blackwell 上触发 TF 的 CUDA 报错。
"""

import numpy as np
import numpy.linalg as LA


def _to_numpy(x):
    """把 TF EagerTensor / torch.Tensor / list 等安全地转成 np.ndarray。"""
    if isinstance(x, np.ndarray):
        return x
    # 优先尝试 .numpy()（TF/PyTorch 张量通常都有）
    try:
        return x.numpy()
    except Exception:
        pass
    # 再尝试 np.asarray
    try:
        return np.asarray(x)
    except Exception:
        pass
    # 兜底
    return np.array(x)


def l2_normalize_rows(X, eps=1e-12):
    """对每一行做 L2 归一化；确保二维。"""
    X = _to_numpy(X).astype(np.float32, copy=False)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def compute_centroids_numpy(X, y):
    """按标签 y 计算每个类的质心（简单均值）。"""
    X = _to_numpy(X)
    y = _to_numpy(y)
    classes = np.unique(y)
    centroids = []
    for c in classes:
        mask = (y == c)
        if not np.any(mask):
            centroids.append(np.zeros((X.shape[1],), dtype=np.float32))
        else:
            centroids.append(np.mean(X[mask], axis=0))
    return np.vstack(centroids).astype(np.float32), classes


def pairwise_euclidean(A, B):
    """
    计算 A (n,d) 与 B (m,d) 的欧氏距离矩阵，返回 (n,m)。
    """
    A = _to_numpy(A).astype(np.float32, copy=False)
    B = _to_numpy(B).astype(np.float32, copy=False)
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
    aa = np.sum(A * A, axis=1, keepdims=True)          # (n,1)
    bb = np.sum(B * B, axis=1, keepdims=True).T        # (1,m)
    ab = A @ B.T                                       # (n,m)
    d2 = np.maximum(aa + bb - 2.0 * ab, 0.0)
    return np.sqrt(d2, dtype=np.float32)


def find_best_radius_numpy(X_train, y_train, centroids, alpha=1.0, step_size=0.01):
    """
    纯 NumPy 版本的半径搜索。
    - X_train, centroids 已做过 L2 归一化，点位于单位球面上；
    - 最大距离上界设为 2（单位 n-sphere 上两点最远距离）。
    """
    X = _to_numpy(X_train).astype(np.float32, copy=False)
    y = _to_numpy(y_train)
    C = _to_numpy(centroids).astype(np.float32, copy=False)

    classes = np.unique(y)
    num_classes = len(classes)
    # 将 label 映射到 [0..num_classes-1] 的索引，方便按类访问
    class_to_idx = {c: i for i, c in enumerate(classes)}
    radius = np.zeros(shape=(num_classes,), dtype=np.float32)

    for i, c in enumerate(classes):
        # 到类 c 质心的距离
        dists_sel = LA.norm(X - C[i], axis=1)

        # 计算 in-domain / out-of-domain 掩码
        id_mask = (y == c).astype(np.float32)
        ood_mask = (y != c).astype(np.float32)

        id_cnt = np.sum(id_mask)
        ood_cnt = np.sum(ood_mask)
        # 防止除零
        if id_cnt == 0:
            # 该类没有样本，半径就保持 0
            radius[i] = 0.0
            continue

        per = ood_cnt / id_cnt

        # 逐步增大半径直到判别条件反向
        # 最大距离上界 2：单位球面上两点的最大欧氏距离
        while radius[i] < 2.0:
            ood_criterion = (dists_sel - radius[i]) * ood_mask
            id_criterion = (radius[i] - dists_sel) * id_mask
            # 用均值衡量
            crit = np.mean(ood_criterion) - (np.mean(id_criterion) * per / float(alpha))

            if crit < 0:
                # ID 权重大于 OOD，回退一步并停止
                radius[i] -= step_size
                break
            radius[i] += step_size

        # 防止出现负半径
        if radius[i] < 0:
            radius[i] = 0.0

    return radius, classes


class ADBThreshold:
    """
    Adaptive Decision Boundary Threshold (NumPy-only)
    """

    def __init__(self, alpha=1.0, step_size=0.01, oos_label=-1):
        self.radius = None            # (num_classes,)
        self.centroids = None         # (num_classes, feat_dim)
        self.classes = None           # 原始类标签集合（与质心、半径顺序一致）
        self.oos_label = oos_label
        self.alpha = alpha
        self.step_size = step_size

    def fit(self, X_train, y_train):
        # 归一化到单位球面
        X_train = l2_normalize_rows(X_train)
        y_train = _to_numpy(y_train)

        # 计算各类质心（再做 L2 归一化，保证在单位球面上）
        centroids, classes = compute_centroids_numpy(X_train, y_train)
        centroids = l2_normalize_rows(centroids)

        # 搜索每个类的阈值半径
        radius, classes_checked = find_best_radius_numpy(
            X_train, y_train, centroids, alpha=self.alpha, step_size=self.step_size
        )
        assert np.array_equal(classes, classes_checked), "Class ordering mismatch."

        self.centroids = centroids
        self.radius = radius
        self.classes = classes

    def predict(self, X_test):
        # 归一化
        X_test = l2_normalize_rows(X_test)

        # 距离矩阵 (n_samples, num_classes)
        dmat = pairwise_euclidean(X_test, self.centroids)

        # 选最近的类
        predictions_idx = np.argmin(dmat, axis=1)  # 索引：0..num_classes-1
        # 转回原始标签
        predictions = self.classes[predictions_idx]

        # 计算与被分配质心的距离，并与半径比较，超出则判为 OOS
        chosen_centroids = self.centroids[predictions_idx]         # (n, d)
        chosen_radius = self.radius[predictions_idx]               # (n,)
        d = np.linalg.norm(X_test - chosen_centroids, axis=1)      # (n,)
        predictions = np.where(d < chosen_radius, predictions, self.oos_label)

        return predictions

    def predict_proba(self, X_test):
        raise NotImplementedError("Adaptive Decision Boundary Threshold 仅在 ood_train 中使用。")
