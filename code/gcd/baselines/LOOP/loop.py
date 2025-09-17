import os
from model import CLBert
from init_parameter import init_model
from dataloader import Data
from mtp import PretrainModelManager
from utils.tools import *
from utils.memory import MemoryBank, fill_memory_bank
from utils.neighbor_dataset import NeighborsDataset
from model import BertForModel
from transformers import logging, WEIGHTS_NAME
import warnings
from scipy.spatial import distance as dist
# from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import yaml
import sys
import json

warnings.filterwarnings('ignore')
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LoopModelManager:
    def __init__(self, args, data, pretrained_model=None):
        set_seed(args.seed)
        self.args = args
        n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = data.num_labels
        self.model = CLBert(args.bert_model, device=self.device, num_labels=data.n_known_cls)

        if n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        
        if pretrained_model is None:
            pretrained_model = BertForModel(args.pretrain_dir, num_labels=data.n_known_cls)
            # if os.path.exists(args.pretrain_dir):
            #     pretrained_model = self.restore_model(args, pretrained_model)
        self.pretrained_model = pretrained_model
        
        self.load_pretrained_model()
        
        if args.cluster_num_factor > 1:
            self.num_labels = self.predict_k(args, data) 
        else:
            self.num_labels = data.num_labels
        
        self.num_train_optimization_steps = int(len(data.train_semi_dataset) / args.train_batch_size) * args.num_train_epochs
        
        self.optimizer, self.scheduler = self.get_optimizer(args)
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)

    def get_neighbor_dataset(self, args, data, indices, query_index, pred):
        dataset = NeighborsDataset(args, data.train_semi_dataset, indices, query_index, pred)
        self.train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
        self.dataset = dataset

    def get_neighbor_inds(self, args, data, km):
        memory_bank = MemoryBank(len(data.train_semi_dataset), args.feat_dim, len(data.all_label_list), 0.1)
        fill_memory_bank(data.train_semi_dataloader, self.model, memory_bank)
        indices, query_index = memory_bank.mine_nearest_neighbors(args.topk, km.labels_, km.cluster_centers_)
        return indices, query_index
    
    def get_adjacency(self, args, inds, neighbors, targets):
        """get adjacency matrix"""
        adj = torch.zeros(inds.shape[0], inds.shape[0])
        for b1, n in enumerate(neighbors):
            adj[b1][b1] = 1
            for b2, j in enumerate(inds):
                if j in n:
                    adj[b1][b2] = 1 # if in neighbors
                # if (targets[b1] == targets[b2]) and (targets[b1]>=0) and (targets[b2]>=0):
                if (targets[b1] == targets[b2]) and (inds[b1] <= args.num_labeled_examples) and (inds[b2] <= args.num_labeled_examples):
                    adj[b1][b2] = 1 # if same labels
                    # this is useful only when both have labels
        return adj

    def evaluation(self, args, data, save_results=True, plot_cm=True):
        """final clustering evaluation on test set"""
        # get features
        feats_test, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats_test = feats_test.cpu().numpy()

        km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats_test)
        y_pred = km.labels_
        y_true = labels.cpu().numpy()
        results = clustering_score(y_true, y_pred, data.known_lab)

        print("----------------- Evaluation Results (LOOP) -----------------")
        print(f"Overall ACC: {results['ACC']:.2f}")
        print(f"H-Score:     {results['H-Score']:.2f}")
        print(f"Known ACC:   {results['K-ACC']:.2f}")
        print(f"Novel ACC:   {results['N-ACC']:.2f}")
        print(f"NMI:         {results['NMI']:.2f}")
        print(f"ARI:         {results['ARI']:.2f}")
        print("-------------------------------------------------------------")

        self.test_results = results
        
        # save results
        if save_results:
            self.save_results(args)
        return results
        # # Save results and features
        # if save_results:
        #     dir_name = f"{args.save_results_path}/{args.dataset}_{args.known_cls_ratio}_{args.labeled_ratio}"
        #     os.makedirs(dir_name, exist_ok=True)
            
        #     np.save(os.path.join(dir_name, "feats_test.npy"), feats_test)
        #     np.save(os.path.join(dir_name, "kmeans_centers_test.npy"), km.cluster_centers_)
        #     np.save(os.path.join(dir_name, "y_pred_test.npy"), y_pred)
        #     np.save(os.path.join(dir_name, "y_true_test.npy"), y_true)

        #     # Save metrics
        #     self.save_results(args)

    def train(self, args, data):
        if isinstance(self.model, nn.DataParallel):
            criterion = self.model.module.loss_cl
            ce = self.model.module.loss_ce
        else:
            criterion = self.model.loss_cl
            ce = self.model.loss_ce
        feats, labels = self.get_features_labels(data.train_semi_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        labels = labels.cpu().numpy()
        km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)
        
        # load neighbors for the first epoch
        indices, query_index = self.get_neighbor_inds(args, data, km)
        self.get_neighbor_dataset(args, data, indices, query_index, km.labels_)
        labelediter = iter(data.train_labeled_dataloader)

        best_model = copy.deepcopy(self.model)
        best_score = float("-inf")
        wait = 0
        metric_name = getattr(args, "es_metric", "NMI")
        es_patience = getattr(args, "es_patience", 5)
        es_min_delta = getattr(args, "es_min_delta", 0.0)
        
        for epoch in range(int(args.num_train_epochs)):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for _, batch in enumerate(self.train_dataloader):
                # 1. load data
                anchor = tuple(t.to(self.device) for t in batch["anchor"]) # anchor data
                neighbor = tuple(t.to(self.device) for t in batch["neighbor"]) # neighbor data
                pos_neighbors = batch["possible_neighbors"] # all possible neighbor inds for anchor
                data_inds = batch["index"] # neighbor data ind

                # 2. get adjacency matrix
                adjacency = self.get_adjacency(args, data_inds, pos_neighbors, batch["target"]) # (bz,bz)

                # 3. get augmentations
                if args.view_strategy == "rtr":
                    X_an = {"input_ids":self.generator.random_token_replace(anchor[0].cpu()).to(self.device), "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                    X_ng = {"input_ids":self.generator.random_token_replace(neighbor[0].cpu()).to(self.device), "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
                elif args.view_strategy == "shuffle":
                    X_an = {"input_ids":self.generator.shuffle_tokens(anchor[0].cpu()).to(self.device), "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                    X_ng = {"input_ids":self.generator.shuffle_tokens(neighbor[0].cpu()).to(self.device), "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
                elif args.view_strategy == "none":
                    X_an = {"input_ids":anchor[0], "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                    X_ng = {"input_ids":neighbor[0], "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
                else:
                    raise NotImplementedError(f"View strategy {args.view_strategy} not implemented!")
                
                # 4. compute loss and update parameters
                with torch.set_grad_enabled(True):
                    f_pos = torch.stack([self.model(X_an)["features"], self.model(X_ng)["features"]], dim=1)
                    loss_cl = criterion(f_pos, mask=adjacency, temperature=args.temp)
                    
                    try:
                        batch = next(labelediter)
                    except StopIteration:
                        labelediter = iter(data.train_labeled_dataloader)
                        batch = next(labelediter)
                    batch = tuple(t.to(self.device) for t in batch)
                    X_an = {"input_ids":batch[0], "attention_mask":batch[1], "token_type_ids":batch[2]}

                    logits = self.model(X_an)["logits"]
                    loss_ce = ce(logits, batch[3]) 

                    loss = 0.5 * loss_ce + loss_cl
                    tr_loss += loss.item()
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    nb_tr_examples += anchor[0].size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            self.dataset.count = 0

            # === 早停评估（每个 epoch 结束评估一次）===
            results = self.evaluation(args, data, save_results=False, plot_cm=False)
            score = results.get(metric_name, None)
            if score is None:
                # 兜底：若用户填了未知指标名，默认用 NMI
                score = results["NMI"]
            if score > best_score + es_min_delta:
                best_score = score
                best_model = copy.deepcopy(self.model)
                wait = 0
                # 可选：也可以在此处落盘一个 "model_epoch_best.pt"
            else:
                wait += 1
                if wait >= es_patience:
                    print(f"Early stopping triggered on metric {metric_name}. Best={best_score:.2f}")
                    break
                        
            # update neighbors every several epochs
            if ((epoch + 1) % args.update_per_epoch) == 0 and ((epoch + 1) != int(args.num_train_epochs)):
                self.evaluation(args, data, save_results=False, plot_cm=False)

                feats, labels = self.get_features_labels(data.train_semi_dataloader, self.model, args)
                feats = feats.cpu().numpy()
                # k-means clustering
                km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)
                indices, query_index = self.get_neighbor_inds(args, data, km)
                self.get_neighbor_dataset(args, data, indices, query_index, km.labels_)

    def get_optimizer(self, args):
        num_warmup_steps = int(args.warmup_proportion*self.num_train_optimization_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)
        return optimizer, scheduler
    
    def load_pretrained_model(self):
        """load the backbone of pretrained model"""
        if isinstance(self.pretrained_model, nn.DataParallel):
            pretrained_dict = self.pretrained_model.module.backbone.state_dict()
        else:
            pretrained_dict = self.pretrained_model.backbone.state_dict()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.backbone.load_state_dict(pretrained_dict, strict=False)
        else:
            self.model.backbone.load_state_dict(pretrained_dict, strict=False)

    def get_features_labels(self, dataloader, model, args):
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for _, batch in enumerate(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feature = model(X, output_hidden_states=True)["hidden_states"]

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels
            
    def save_results(self, args):
        # 1. 定义要保存的实验配置
        config_to_save = {
            'method': getattr(args, 'method', 'LOOP'), # 允许在运行时指定方法名
            'dataset': args.dataset,
            'known_cls_ratio': args.known_cls_ratio,
            'labeled_ratio': getattr(args, 'labeled_ratio', 0.1),
            'cluster_num_factor': getattr(args, 'cluster_num_factor', 1.0),
            'seed': args.seed,
            'K': self.num_labels
        }
        
        # 2. 合并配置和结果 (self.test_results 已经包含了新的6个指标)
        full_results = {**config_to_save, **self.test_results}
        full_results['args'] = json.dumps(vars(args), ensure_ascii=False)
        
        # 3. 定义结果文件路径
        save_path = args.save_results_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        results_file = os.path.join(save_path, "results.csv")
        
        # 4. 使用 Pandas 写入 CSV
        new_df = pd.DataFrame([full_results])
        
        if not os.path.exists(results_file):
            new_df.to_csv(results_file, index=False)
        else:
            # 使用 pd.concat 替代已弃用的 _append
            try:
                old_df = pd.read_csv(results_file)
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
                combined_df.to_csv(results_file, index=False)
            except Exception as e:
                # 如果读取或合并失败，用新数据覆盖
                print(f"Warning: Could not append to CSV due to {e}. Overwriting file.")
                new_df.to_csv(results_file, index=False)

        print(f"Results successfully saved to {results_file}")
    
    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def cluster_name(self, args, data):
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

        feats_gpu, _ = self.get_features_labels(data.train_semi_dataloader, self.model, args)
        feats = feats_gpu.cpu().numpy()
        
        km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)
        # unlabeled prototypes
        proto_u = km.cluster_centers_
        distance = dist.cdist(proto_l, proto_u, 'euclidean')
        _, col_ind = linear_sum_assignment(distance)

        novel_id = [i for i in range(self.num_labels) if i not in col_ind]
        cluster_centers = torch.tensor(km.cluster_centers_[novel_id])
        dis = self.EuclideanDistances(feats_gpu.cpu(), cluster_centers).T
        _, index = torch.sort(dis, dim=1)
        index = index[:, :3]
        cluster_name = []

        cluster_name_map_id = {}# ==== 新增 ==== 创建字典用于保存 cluster_id 与名称映射

        for i in range(len(index)):
            query = []
            for j in index[i]:
                query.append(data.train_semi_dataset.__getitem__(j)[0])
            name = self.query_llm(query)
            cluster_name.append(name)
            cluster_name_map_id[i] = name  # ==== 新增 ==== 保存 cluster_id -> cluster_name 的映射
        print(cluster_name)
            
        # ==== 新增 ==== 构建保存目录
        dir_name = f"{args.save_results_path}/{args.dataset}_{args.known_cls_ratio}_{args.labeled_ratio}"
        os.makedirs(dir_name, exist_ok=True)

        # ==== 新增 ==== 保存 JSON 格式的名称映射
        with open(os.path.join(dir_name, "cluster_names.json"), "w", encoding="utf-8") as f:
            json.dump(cluster_name_map_id, f, indent=4, ensure_ascii=False)
        preds = torch.argmin(dis.T, dim=-1).cpu().numpy()

        sample_idx = np.load(f'{dir_name}/sample_idx.npy') - 1

        new_feats = np.empty_like(feats)
        new_feats[sample_idx] = feats
        new_preds = np.empty_like(preds)
        new_preds[sample_idx] = preds

        np.save(os.path.join(dir_name, "feats_train.npy"), new_feats)
        np.save(os.path.join(dir_name, "y_pred_train.npy"), new_preds)

    def query_llm(self, a):
        s1 = self.tokenizer.decode(a[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        s2 = self.tokenizer.decode(a[1], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        s3 = self.tokenizer.decode(a[2], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        prompt_text = "Given the following customer utterances, return a word or a phrase to summarize the common intent of these utterances without explanation. \n Utterance 1: " + s1 + "\n Utterance 2: " + s2 + "\n Utterance 3: " + s3

        try:
            # 从环境变量获取 API Key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("[FATAL] Environment variable OPENAI_API_KEY is not set.")

            # 从 self.args 动态读取配置，现在我们确信它们是正确的了
            llm = ChatOpenAI(
                model=self.args.llm_model_name,      
                openai_api_key=api_key,
                openai_api_base=self.args.api_base, 
                temperature=0.0,
                max_retries=5,  # 设置最大重试次数为3次
                timeout=120,     # 将请求超时时间延长到30秒
            )

            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=prompt_text),
            ]
            
            response = llm.invoke(messages)
            return response.content
    
        except Exception as e:
            # 保留详细的错误打印，以备不时之需
            import traceback
            print("\n" + "="*50)
            print(f"[FATAL ERROR in query_llm]: An error occurred.")
            traceback.print_exc()
            print("="*50 + "\n")
            return "LLM_QUERY_ERROR"
                
    def EuclideanDistances(self, a, b):
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

    def predict_k(self, args, data):
        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.pretrained_model.cuda(), args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels * 0.9
        print('drop',drop_out)

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('pred_num',num_labels)

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

    print('Data and Parameters Initialization...')
    parser = init_model()
    # 1. 新增 --config 参数
    parser.add_argument("--config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    # 2. 如果提供了 config 文件，则加载并应用它
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # 应用 YAML 配置，这会作为默认值
        apply_config_updates(args, yaml_config, parser)

        if 'dataset_specific_configs' in yaml_config:
            dataset_configs = yaml_config['dataset_specific_configs'].get(args.dataset, {})
            apply_config_updates(args, dataset_configs, parser)

    # args.bert_model = '../../pretrained_models/bert-base-chinese' if args.dataset == 'ecdt' else args.bert_model
    # args.tokenizer = '../../pretrained_models/bert-base-chinese' if args.dataset == 'ecdt' else args.tokenizer
    args.bert_model = './pretrained_models/bert-base-chinese' if args.dataset == 'ecdt' else args.bert_model
    args.tokenizer = './pretrained_models/bert-base-chinese' if args.dataset == 'ecdt' else args.tokenizer


    data = Data(args)
    if os.path.exists(args.pretrain_dir):
        args.disable_pretrain = True # disable internal pretrain
    else:
        args.disable_pretrain = False

    if not args.disable_pretrain:
        print('Pre-training begin...')
        manager_p = PretrainModelManager(args, data)
        manager_p.train(args, data)
        print('Pre-training finished!')
        manager = LoopModelManager(args, data, manager_p.model)
    else:
        manager = LoopModelManager(args, data)
    
    if args.report_pretrain:
        method = args.method
        args.method = 'pretrain'
        manager.evaluation(args, data) # evaluate when report performance on pretrain
        args.method = method

    print('Training begin...')
    manager.train(args,data)
    print('Training finished!')

    print('Evaluation begin...')
    manager.evaluation(args, data)
    print('Evaluation finished!')
    # manager.cluster_name(args,data)
    print('Saving Model ...')
    if args.save_model:
        manager.model.save_backbone(args.save_model_path)
    print("Finished!")