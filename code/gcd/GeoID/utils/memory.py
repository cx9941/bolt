"""
Code modified from
https://github.com/wvangansbeke/Unsupervised-Classification
"""
import numpy as np
import torch
import torch.nn.functional as F

class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n,768)
        self.f=torch.FloatTensor(self.n,self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred
    


    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        """
        使用 PyTorch 实现最近邻检索，替代 FAISS。
        计算每个样本的 top-k 近邻索引，若开启 calculate_accuracy 则评估近邻标签一致性。
        """

        # 特征向量归一化（用于余弦相似度）
        features = F.normalize(self.f, dim=1)  # [n, dim]
        features = features.to('cuda')  # 若模型在GPU上

        # 相似度矩阵（余弦相似度 = 内积）
        sim_matrix = features @ features.T  # [n, n]

        # 找 topk+1 个（包含自身）
        distances, indices = torch.topk(sim_matrix, k=topk+1, dim=1)

        # 去除每个样本本身（即最大相似度点）
        indices = indices[:, 1:]

        if calculate_accuracy:
            # 获取标签并展开对比
            targets = self.targets.to('cpu').numpy()  # [n]
            neighbor_targets = np.take(targets, indices.cpu().numpy(), axis=0)  # [n, topk]
            anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)    # [n, topk]
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        else:
            return indices
        
        # def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        #     # mine the topk nearest neighbors for every sample
        #     import faiss
        #     features = self.f.cpu().numpy()
        #     n, dim = features.shape[0], features.shape[1]
        #     index = faiss.IndexFlatIP(dim)
        #     index = faiss.index_cpu_to_all_gpus(index)
        #     index.add(features)
        #     distances, indices = index.search(features, topk+1) # Sample itself is included
        #     # evaluate 
        #     if calculate_accuracy:
        #         targets = self.targets.cpu().numpy()
        #         neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
        #         anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
        #         accuracy = np.mean(neighbor_targets == anchor_targets)
        #         return indices, accuracy
            
        #     else:
        #         return indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, f,targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.f[self.ptr:self.ptr+b].copy_(f.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.f=self.f.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):

        batch = tuple(t.cuda(non_blocking=True) for t in batch)
        input_ids, input_mask, segment_ids, label_ids,_ = batch
        X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
        dict = model(X, output_hidden_states=True)
        feature=dict["hidden_states"]
        f=dict["features"]
        memory_bank.update(feature,f, label_ids)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))