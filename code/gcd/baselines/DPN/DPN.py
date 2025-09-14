import os
from init_parameter import init_model
from data import Data
from model import BertForModel, PretrainModelManager
import numpy as np
import torch.nn.functional as F
from util import clustering_score
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import torch
from sklearn.cluster import KMeans
from transformers import logging, AutoTokenizer, WEIGHTS_NAME
import random
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from pytorch_pretrained_bert.optimization import BertAdam
import math
import warnings
# 添加 tqdm 导入
from tqdm import tqdm
# 在文件顶部的导入部分添加
import pandas as pd
import copy
import yaml
import sys



class ModelManager:
    def __init__(self, args, data, pretrained_model=None):
        self.set_seed(args.seed)
        if pretrained_model is None:
            pretrained_model = BertForModel(args, data.n_known_cls)
            if os.path.exists(args.pretrain_dir):
                pretrained_model = self.restore_model(args, pretrained_model)
        self.pretrained_model = pretrained_model
        self.labelMap = None
        
        self.seed = args.seed         
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.cluster_num_factor > 1:
            self.num_labels = self.predict_k(args, data) 
        else:
            self.num_labels = data.num_labels       

        self.model = BertForModel(args, self.num_labels)
        
        self.load_pretrained_model()

        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)
            
        self.model.to(self.device)
        
        self.data = data
        num_train_examples = 2 * len(data.train_labeled_examples) + 2 * len(data.train_unlabeled_examples)
        self.num_training_steps = math.ceil(num_train_examples / args.train_batch_size) * 100
        self.num_warmup_steps= int(args.warmup_proportion * self.num_training_steps) 
        self.optimizer = self.get_optimizer(args)
        self.best_eval_score = 0
        self.centroids = None
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        
        # 添加这行来存储测试结果
        self.test_results = None

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def get_features_labels(self, dataloader, model, args):      
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        for _, batch in enumerate(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(input_ids, segment_ids, input_mask, mode='feature_extract')
            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))
        return total_features, total_labels
    
    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr,
                         warmup = args.warmup_proportion,
                         t_total = self.num_training_steps)   
        return optimizer

    def evaluation(self, args, data):
        self.model.eval()
        feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = self.num_labels, n_init=20).fit(feats)
        y_true = labels.cpu().numpy()
        results = clustering_score(y_true, km.labels_, data.known_lab)
        print(results)
        
        # 保存结果到实例变量
        self.test_results = results
        
        # 调用保存结果方法
        self.save_results(args)
        
    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed, self.num_labels]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor','seed', 'K']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        file_name = 'results.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori,columns = keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results,index=[1])
            # 使用 concat 替代已弃用的 _append 方法
            df1 = pd.concat([df1, new], ignore_index=True)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
        
        print('test_results', data_diagram)
    
    def alignment(self, km, args):
        if self.centroids is not None:
            old_centroids = self.centroids.cpu().numpy()
            new_centroids = km.cluster_centers_
            
            DistanceMatrix = np.linalg.norm(old_centroids[:,np.newaxis,:]-new_centroids[np.newaxis,:,:],axis=2) 
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)
            
            new_centroids = torch.tensor(new_centroids).to(self.device)
            self.centroids = torch.empty(self.num_labels ,args.feat_dim).to(self.device)
            
            alignment_labels = list(col_ind)
            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]
                
            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])
            self.labelMap = pseudo2label
        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)        
            pseudo_labels = km.labels_ 

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)
        
        return pseudo_labels

    def update_cluster_ids(self, pseudo_labels, args, data):
        train_data = TensorDataset(data.semi_input_ids, data.semi_input_mask, data.semi_segment_ids, pseudo_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size)
        return train_dataloader

    def quick_eval(self, args, dataloader, known_lab):
        self.model.eval()
        feats, labels = self.get_features_labels(dataloader, self.model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters=self.num_labels, n_init=20).fit(feats)
        y_true = labels.cpu().numpy()
        return clustering_score(y_true, km.labels_, known_lab)

    def train(self, args, data):    
        feats_label, labels = self.get_features_labels(data.train_labeled_dataloader, self.model, args)
        feats_label = feats_label.cpu().numpy()
        labels = labels.cpu().numpy()
        [rows, cols] = feats_label.shape
        num = np.zeros(data.n_known_cls)
        # labeled prototypes
        proto_l = np.zeros((data.n_known_cls, args.feat_dim))
        for i in range(rows):
            proto_l[labels[i]] += feats_label[i]
            num[labels[i]] += 1
        for i in range(data.n_known_cls):
            proto_l[i] = proto_l[i] / num[i]

        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        
        print("Performing k-means...")
        km = KMeans(n_clusters = self.num_labels, n_init=20).fit(feats)
        print("K-means finished")
        # unlabeled prototypes
        proto_u = km.cluster_centers_
        distance = dist.cdist(proto_l, proto_u, 'euclidean')
        row_ind, col_ind = linear_sum_assignment(distance)
        pro_l = []
        for i in range(len(col_ind)):
            pro_l.append(proto_u[col_ind[i]][:])
        pro_u = []
        for j in range(self.num_labels):
            if j not in col_ind:
                pro_u.append(proto_u[j][:])
        
        # aligned unlabeled prototypes
        pro_u = pro_l + pro_u   
        proto_u = torch.tensor(np.array(pro_u)).to(self.device)
        
        # update cluster_ids for unlabeled data
        self.centroids = proto_u
        pseudo_labels = self.alignment(km, args)
        train_dataloader = self.update_cluster_ids(pseudo_labels, args, data)
        
        labelediter = iter(data.train_labeled_dataloader)
        
        self.proto_l = proto_l
        self.proto_u = proto_u

        # 只保留epoch进度条，移除batch进度条
        epoch_pbar = tqdm(range(1, int(args.num_train_epochs)+1), desc="Fine-tuning Epochs")
        
        for epoch in epoch_pbar:
            
            feats_label, labels = self.get_features_labels(data.train_labeled_dataloader, self.model, args)
            feats_label = feats_label.cpu().numpy()
            labels = labels.cpu().numpy()
            [rows, cols] = feats_label.shape
            num = np.zeros(data.n_known_cls)
            proto_l = np.zeros((data.n_known_cls, args.feat_dim))
            for i in range(rows):
                proto_l[labels[i]] += feats_label[i]
                num[labels[i]] += 1
            for i in range(data.n_known_cls):
                proto_l[i] = proto_l[i] / num[i]
            
            # update labeled prototypes
            self.proto_l = args.momentum_factor * self.proto_l + (1 - args.momentum_factor) * proto_l
            
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()   

            # 移除batch进度条，直接遍历dataloader
            for batch_idx, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                pooled = self.model(input_ids, segment_ids, input_mask, label_ids, mode='feature_extract')

                sim_mat = self.pairwise_cosine_sim(pooled, self.proto_u) / args.temperature
                s_dist = F.softmax(sim_mat, dim=1)
                cost_mat = self.EuclideanDistances(pooled, self.proto_u)
                loss_u = (cost_mat * s_dist).sum(1).mean()

                mask = torch.where(label_ids < len(self.proto_l), 1, 0).reshape((-1,1))
                sim_mat = self.pairwise_cosine_sim(pooled, torch.tensor(self.proto_l).float().to(self.device)) / args.temperature
                s_dist = F.softmax(sim_mat, dim=1)
                cost_mat = 1 - self.pairwise_cosine_sim(pooled, torch.tensor(self.proto_l).float().to(self.device))
                loss_l = (cost_mat * s_dist * mask).sum(1).mean()

                loss_pro = loss_u + args.gamma * loss_l
                
                try:
                    batch = next(labelediter)
                except StopIteration:
                    labelediter = iter(data.train_labeled_dataloader)
                    batch = next(labelediter)
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss_ce, _ = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train') 

                loss = loss_pro + loss_ce
                
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 只在epoch进度条显示平均loss
            avg_loss = tr_loss/nb_tr_steps
            epoch_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
            print(f'Epoch {epoch} loss: {avg_loss:.4f}')
            
            # --- Early Stopping for main training ---
            monitor_key = 'NMI'   # or 'ACC'/'H-Score'
            min_delta = 0.0
            metrics = self.quick_eval(args, self.data.eval_dataloader, self.data.known_lab)
            score = metrics[monitor_key]

            if not hasattr(self, "_best_main_score"):
                self._best_main_score = float("-inf")
                self._wait_main = 0
                self._best_main_model = copy.deepcopy(self.model)

            if score > self._best_main_score + min_delta:
                self._best_main_score = score
                self._wait_main = 0
                self._best_main_model = copy.deepcopy(self.model)
            else:
                self._wait_main += 1
                if self._wait_main >= args.wait_patient:
                    print(f"[EarlyStop] stop at epoch {epoch}: best {monitor_key}={self._best_main_score}")
                    self.model = self._best_main_model
                    return
              
    def load_pretrained_model(self):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)
        
    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model
    
    def freeze_parameters(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name or "encoder.layer.10" in name or "encoder.layer.9" in name:
                param.requires_grad = True

    def pairwise_cosine_sim(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return torch.matmul(x, y.T)

    def EuclideanDistances(self, a, b):
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))
    
    def predict_k(self, args, data):
        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.pretrained_model.to(self.device), args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels * 0.9

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('predict_K', num_labels)

        return num_labels

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
    warnings.filterwarnings('ignore')
    logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print('Data and Parameters Initialization...')
    parser = init_model()
    # 1. 新增 --config 参数
    parser.add_argument("--config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    # 2. 如果提供了 config 文件，则加载并分层应用它
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # 应用通用配置
        apply_config_updates(args, yaml_config, parser)
        
        # 应用数据集专属配置
        if 'dataset_specific_configs' in yaml_config:
            dataset_configs = yaml_config['dataset_specific_configs'].get(args.dataset, {})
            apply_config_updates(args, dataset_configs, parser)
    data = Data(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.pretrain:
        print('Pre-training begin...')
        manager_p = PretrainModelManager(args, data)
        manager_p.train()
        print('Pre-training finished!')
        manager = ModelManager(args, data, manager_p.model)
    else:
        manager = ModelManager(args, data, None)
    
    print('Training begin...')
    manager.train(args,data)
    print('Training finished!')

    
    print('Evaluation begin...')
    manager.evaluation(args, data)
    # 移除这行，因为 evaluation 方法中已经调用了 save_results
    # manager.save_results(args)
    print('Evaluation finished!')

