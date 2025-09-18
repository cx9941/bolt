from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
import yaml
import sys
import math
import json

class ModelManager:
    
    def __init__(self, args, data, pretrained_model=None):
        
        if pretrained_model is None:
            pretrained_model = BertForModel.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.n_known_cls)
            if os.path.exists(args.pretrain_dir):
                pretrained_model = self.restore_model(args, pretrained_model)
        self.pretrained_model = pretrained_model

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.cluster_num_factor > 1:
            self.num_labels = self.predict_k(args, data) 
        else:
            self.num_labels = data.num_labels       

        self.model = BertForModel.from_pretrained(args.bert_model, cache_dir = "", num_labels = self.num_labels)    
        
        if args.pretrain:
            self.load_pretrained_model(args)

        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)
            
        self.model.to(self.device)

        num_train_examples = len(data.train_labeled_examples) + len(data.train_unlabeled_examples)
        steps_per_epoch = max(1, math.ceil(num_train_examples / args.train_batch_size))
        epochs = max(1, int(args.num_train_epochs))
        self.num_train_optimization_steps = steps_per_epoch * epochs
        
        self.optimizer = self.get_optimizer(args)

        self.best_eval_score = 0
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def get_features_labels(self, dataloader, model, args):
        
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(input_ids, segment_ids, input_mask, feature_ext = True)

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def predict_k(self, args, data):
        
        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.pretrained_model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels
        print('drop',drop_out)

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('pred_num',num_labels)

        num_labels = max(2, num_labels)
        return num_labels
    
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
                         t_total = self.num_train_optimization_steps)   
        return optimizer

    def evaluation(self, args, data):
        feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = self.num_labels).fit(feats)
    
        y_pred = km.labels_
        y_true = labels.cpu().numpy()
    
        # 修复：添加缺失的 known_lab 参数，参考 SDC 的实现
        results = clustering_score(y_true, y_pred, data.known_lab)
        print('results',results)
        
        ind, _ = hungray_aligment(y_true, y_pred)
        map_ = {i[0]:i[1] for i in ind}
        y_pred = np.array([map_[idx] for idx in y_pred])

        cm = confusion_matrix(y_true,y_pred)   
        print('confusion matrix',cm)
        self.test_results = results
        

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

        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)        
            pseudo_labels = km.labels_ 

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)
        
        return pseudo_labels

    def update_pseudo_labels(self, pseudo_labels, args, data):
        train_data = TensorDataset(data.semi_input_ids, data.semi_input_mask, data.semi_segment_ids, pseudo_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size)
        return train_dataloader

    def train(self, args, data): 

        best_score = float("-inf")
        best_model = copy.deepcopy(self.model)  # 先备份一份，保证不为 None
        wait = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  

            feats, _ = self.get_features_labels(data.train_semi_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            km = KMeans(n_clusters = self.num_labels).fit(feats)
            
            score = metrics.silhouette_score(feats, km.labels_)
            print('score',score)

            if score > best_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_score = score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    if best_model is not None:
                        self.model = best_model
                    break 
            
            pseudo_labels = self.alignment(km, args)
            train_dataloader = self.update_pseudo_labels(pseudo_labels, args, data)
            
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for batch in train_dataloader:

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train')
                
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
            
            tr_loss = tr_loss / nb_tr_steps
            print('train_loss',tr_loss)
        

    def load_pretrained_model(self, args):
        pretrained_dict = self.pretrained_model.state_dict()
        pretrained_dict = {k.replace('module.', '') if k.startswith('module.') else k: v 
                       for k, v in pretrained_dict.items()}
        classifier_params = ['classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)
        

    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model
    
    def freeze_parameters(self,model):
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed, self.num_labels]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor','seed', 'K']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        results['args'] = json.dumps(vars(args), ensure_ascii=False)
        # keys = list(results.keys())
        # values = list(results.values())

        desired_order = [
            'method', 'dataset', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor', 
            'seed', 'ACC', 'H-Score', 'K-ACC', 'N-ACC', 'ARI', 'NMI', 'args'
        ]
        keys = desired_order
        values = [results[i] for i in desired_order]
        
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
            # 修复：使用 concat 替代已弃用的 _append 方法
            df1 = pd.concat([df1, new], ignore_index=True)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
        
        print('test_results', data_diagram)


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

    args.bert_model = '../../pretrained_models/bert-base-chinese' if args.dataset == 'ecdt' else args.bert_model
    data = Data(args)

    if args.pretrain:
        print('Pre-training begin...')
        manager_p = PretrainModelManager(args, data)
        manager_p.train(args, data)
        print('Pre-training finished!')
        manager = ModelManager(args, data, manager_p.model)
    else:
        manager = ModelManager(args, data)
    
    print('Training begin...')
    manager.train(args,data)
    print('Training finished!')

    print('Evaluation begin...')
    manager.evaluation(args, data)
    print('Evaluation finished!')

    manager.save_results(args)
    
