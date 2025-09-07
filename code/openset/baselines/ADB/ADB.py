# ADB.py (已改造)

from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
from loss import *
import yaml  # <-- 新增
import sys   # <-- 新增

class ModelManager:
    
    def __init__(self, args, data, pretrained_model=None):
        
        self.model = pretrained_model

        if self.model is None:
            self.model = BertForModel.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.num_labels)
            self.restore_model(args)

        # GPU设置移到主程序入口，这里不再需要
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id     
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.best_eval_score = 0
        self.delta = None
        self.delta_points = []
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def open_classify(self, features, data): # <-- 修正：需要传入 data 对象
        logits = euclidean_metric(features, self.centroids)
        probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        preds[euc_dis >= self.delta[preds]] = data.unseen_token_id
        return preds

    def evaluation(self, args, data, mode="eval"):
        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        dataloader = data.eval_dataloader if mode == 'eval' else data.test_dataloader

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output, _ = self.model(input_ids, segment_ids, input_mask)
                preds = self.open_classify(pooled_output, data) # <-- 修正：传入 data

                total_labels = torch.cat((total_labels,label_ids))
                total_preds = torch.cat((total_preds, preds))
        
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        if mode == 'eval':
            cm = confusion_matrix(y_true, y_pred)
            eval_score = F_measure(cm)['f1-score']
            return eval_score

        elif mode == 'test':
            self.predictions = list([data.label_list[idx] for idx in y_pred])
            self.true_labels = list([data.label_list[idx] for idx in y_true])
            
            cm = confusion_matrix(y_true,y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['accuracy'] = acc
            self.test_results = results
            self.save_results(args) # <-- 路径管理已在 save_results 中实现

    def train(self, args, data):     
        criterion_boundary = BoundaryLoss(num_labels = data.num_labels, feat_dim = args.feat_dim)
        self.delta = F.softplus(criterion_boundary.delta)
        optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr = args.lr_boundary)
        self.centroids = self.centroids_cal(args, data)

        wait = 0
        best_delta, best_centroids = None, None

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    loss, self.delta = criterion_boundary(features, self.centroids, label_ids)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            self.delta_points.append(self.delta)
            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            eval_score = self.evaluation(args, data, mode="eval")
            print('eval_score',eval_score)
            
            if eval_score >= self.best_eval_score:
                wait = 0
                self.best_eval_score = eval_score
                best_delta = self.delta
                best_centroids = self.centroids
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
        
        self.delta = best_delta
        self.centroids = best_centroids

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num

    def centroids_cal(self, args, data):
        centroids = torch.zeros(data.num_labels, args.feat_dim).cuda()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        with torch.set_grad_enabled(False):
            for batch in data.train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                total_labels = torch.cat((total_labels, label_ids))
                for i in range(len(label_ids)):
                    label = label_ids[i]
                    centroids[label] += features[i]
                
        total_labels = total_labels.cpu().numpy()
        centroids /= torch.tensor(self.class_count(total_labels)).float().unsqueeze(1).cuda()
        return centroids

    def restore_model(self, args):
        # 路径现在由主程序入口的 output_dir 控制
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        self.model.load_state_dict(torch.load(output_model_file))
    
    def save_results(self, args):
        # 路径现在由主程序入口的 output_dir 控制
        results_path = os.path.join(args.output_dir, "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        var = [args.dataset, args.known_cls_ratio, args.labeled_ratio, args.seed]
        names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results, **vars_dict)
        
        np.save(os.path.join(results_path, 'centroids.npy'), self.centroids.detach().cpu().numpy())
        np.save(os.path.join(results_path, 'deltas.npy'), self.delta.detach().cpu().numpy())
        
        results_csv_path = os.path.join(results_path, 'results.csv')
        df_new_row = pd.DataFrame([results])

        if not os.path.exists(results_csv_path):
            df_new_row.to_csv(results_csv_path, index=False)
        else:
            df_existing = pd.read_csv(results_csv_path)
            df_updated = pd.concat([df_existing, df_new_row], ignore_index=True)
            df_updated.to_csv(results_csv_path, index=False)
        
        print('Test results saved to:', results_csv_path)
        print(df_new_row)


# --- 核心改造：将参数解析和配置注入逻辑放在主入口 ---
if __name__ == '__main__':
    
    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()

    def apply_config_updates(args, config_dict, parser):
        type_map = {action.dest: action.type for action in parser._actions}
        for key, value in config_dict.items():
            if f'--{key}' in sys.argv or not hasattr(args, key):
                continue
            expected_type = type_map.get(key)
            if expected_type and value is not None:
                try:
                    if expected_type is bool or expected_type.__name__ == 'str_to_bool':
                        value = str(value).lower() in ('true', '1', 't', 'yes')
                    else:
                        value = expected_type(value)
                except (ValueError, TypeError):
                    print(f"Warning: Could not cast YAML value '{value}' for key '{key}' to type {expected_type}.")
            setattr(args, key, value)

    # 执行配置注入
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        apply_config_updates(args, yaml_config, parser)
        if 'dataset_specific_configs' in yaml_config:
            dataset_configs = yaml_config['dataset_specific_configs'].get(args.dataset, {})
            apply_config_updates(args, dataset_configs, parser)
    
    # --- 路径管理和环境设置 ---
    # 使用动态 output_dir (如果命令行未提供，则使用YAML/默认值)
    # 这会覆盖掉 YAML 中的 pretrain_dir 和 save_results_path，实现集中管理
    args.pretrain_dir = os.path.join(args.output_dir, 'models')
    args.save_results_path = os.path.join(args.output_dir, 'results')
    os.makedirs(args.pretrain_dir, exist_ok=True)
    os.makedirs(args.save_results_path, exist_ok=True)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # 开始执行任务
    data = Data(args)

    print('Pre-training begin...')
    manager_p = PretrainModelManager(args, data)
    manager_p.train(args, data)
    print('Pre-training finished!')
    
    manager = ModelManager(args, data, manager_p.model)
    print('Training begin...')
    manager.train(args, data)
    print('Training finished!')
    
    print('Evaluation begin...')
    manager.evaluation(args, data, mode="test")  
    print('Evaluation finished!')
