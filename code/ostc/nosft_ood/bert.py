# baselines/analogy/bert.py
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from init_args import parse_args
args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
from utils import load_data

def get_bert_embeddings(texts, model, tokenizer, device, batch_size=32):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding BERT"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embeddings.cpu())
    return torch.cat(embeddings, dim=0)

# def compute_topk_indices_in_batches(test_embeddings, train_embeddings, train_labels, k=100, batch_size=512):
#     """
#     分批计算余弦相似度并获取每个测试样本的 Top-k 相似训练样本的索引。
#     返回 shape: (num_test, k)
#     """
#     topk_indices_all = []
#     test_embeddings = test_embeddings.to('cpu')
#     train_embeddings = train_embeddings.to('cpu')
#     train_norm = torch.nn.functional.normalize(train_embeddings, dim=1)

#     for i in tqdm(range(0, len(test_embeddings), batch_size), desc="Computing Top-k Similarity"):
#         batch = test_embeddings[i:i+batch_size]
#         batch_norm = torch.nn.functional.normalize(batch, dim=1)
#         sim = torch.mm(batch_norm, train_norm.T)  # (batch_size, train_size)
#         topk_sim, topk_indices = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)  # 返回 indices
#         topk_indices_all.append(topk_indices)

#     return torch.cat(topk_indices_all, dim=0)  # (num_test, k)

def compute_topk_indices_in_batches(test_embeddings, train_embeddings, train_labels, k_per_class=20, batch_size=512):
    """
    分批计算余弦相似度并按类别获取每个测试样本的 Top-k 相似训练样本的索引。
    返回 shape: (num_test, num_classes, k_per_class)
    """
    test_embeddings = test_embeddings.to('cpu')
    train_embeddings = train_embeddings.to('cpu')
    train_labels = np.array(train_labels)
    unique_labels = sorted(set(train_labels))
    label2indices = {label: np.where(train_labels == label)[0] for label in unique_labels}

    # 预归一化训练嵌入
    train_norm = torch.nn.functional.normalize(train_embeddings, dim=1)

    # 存储所有 test_num * class_num * k_per_class 索引
    all_class_topk = []

    for i in tqdm(range(0, len(test_embeddings), batch_size), desc="Computing Class-wise Top-k Similarity"):
        batch = test_embeddings[i:i+batch_size]
        batch_norm = torch.nn.functional.normalize(batch, dim=1)

        class_topk_batch = []
        for label in unique_labels:
            label_indices = label2indices[label]
            label_train_emb = train_norm[label_indices]  # (num_label_train, dim)

            sim = torch.mm(batch_norm, label_train_emb.T)  # (batch_size, num_label_train)
            k_per_class = min(k_per_class, sim.shape[1])
            _, topk_local = torch.topk(sim, k=k_per_class, dim=1, largest=True, sorted=True)  # indices w.r.t label_train_emb

            # 映射回原始索引
            topk_global = torch.tensor(label_indices[topk_local], dtype=torch.long)  # (batch_size, k)
            class_topk_batch.append(topk_global)

        # 拼接为 (batch_size, num_classes, k)
        class_topk_batch = torch.stack([_[:,:k_per_class] for _ in class_topk_batch], dim=1)  # [batch_size, num_classes, k_per_class]
        all_class_topk.append(class_topk_batch)

    return torch.cat(all_class_topk, dim=0)  # [num_test, num_classes, k_per_class]

def main():
    output_dir = os.path.join("outputs", "sim_matrix")
    save_path = os.path.join(output_dir, f"{args.dataset}_{args.mask_ratio}.npy")
    if os.path.exists(save_path):
        print(f"{save_path} has been saved")
        return
    

    # 设置设备
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(device)

    # 路径设定
    data_root = os.path.join(args.data_dir, args.dataset, "origin_data")
    train_file = os.path.join(data_root, "train.tsv")
    test_file = os.path.join(data_root, "test.tsv")
    label_file = os.path.join(args.data_dir, args.dataset, f"{args.dataset}_{args.mask_ratio}.txt")

    # 加载数据
    train_df = load_data(train_file)
    test_df = load_data(test_file)
    with open(label_file, 'r') as f:
        valid_labels = [line.strip() for line in f if line.strip()]
    train_df = train_df[train_df['label'].isin(valid_labels)]
    train_df['label_idx']= train_df['label'].apply(lambda x: valid_labels.index(x))

    # 加载 BERT 模型
    tokenizer = AutoTokenizer.from_pretrained('../../llms/bert-base-uncased')
    model = AutoModel.from_pretrained('../../llms/bert-base-uncased').to(device)

    # 获取 BERT 表征
    train_embeddings = get_bert_embeddings(train_df['text'].tolist(), model, tokenizer, device).cpu()
    test_embeddings = get_bert_embeddings(test_df['text'].tolist(), model, tokenizer, device).cpu()

    topk_indices = compute_topk_indices_in_batches(test_embeddings, train_embeddings, train_labels = train_df['label_idx'], k_per_class=20, batch_size=32)

    # 保存结果
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(save_path, topk_indices.cpu().numpy())
    print(f"Saved top-100 similarity indices to {save_path}, shape: {topk_indices.shape}")

if __name__ == "__main__":
    main()