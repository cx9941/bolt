# DOC.py (已改造)

import argparse
import os
import pandas as pd
import yaml  # <-- 新增
import sys   # <-- 新增
import json

# 1. --- 参数定义与配置注入 (SOP 标准化改造) ---
parser = argparse.ArgumentParser()

# -- 新增 config 参数
parser.add_argument('--config', type=str, default=None, help="Path to the YAML config file.")

# -- 基础设置
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_id', type=str, default='0')
# -- 数据与任务设置
parser.add_argument('--dataset', type=str, default='banking')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--known_cls_ratio', type=float, default=0.25)
parser.add_argument('--labeled_ratio', type=float, default=1.0)
parser.add_argument('--fold_idx', type=int, default=0)
parser.add_argument('--fold_num', type=int, default=5)
# -- 模型与训练设置
parser.add_argument('--num_train_epochs', type=int, default=20)
parser.add_argument('--train_batch_size', type=int, default=128)
# -- 输出目录设置
parser.add_argument('--output_dir', type=str, default='./outputs/openset/doc')

# --- 核心改造：配置注入逻辑 ---
args = parser.parse_args()

def apply_config_updates(args, config_dict, parser):
    """
    使用配置字典中的值更新 args 对象，同时进行类型转换。
    命令行中显式给出的参数不会被覆盖。
    """
    # 从 parser 中获取每个参数的期望类型
    type_map = {action.dest: action.type for action in parser._actions}
    
    for key, value in config_dict.items():
        # 如果参数在命令行中指定，或在 args 中不存在，则跳过
        if f'--{key}' in sys.argv or not hasattr(args, key):
            continue
        
        expected_type = type_map.get(key)
        if expected_type and value is not None:
            try:
                # 对布尔值进行特殊处理
                if expected_type is bool:
                    value = str(value).lower() in ('true', '1', 't', 'yes')
                else:
                    value = expected_type(value)
            except (ValueError, TypeError):
                print(f"Warning: Could not cast YAML value '{value}' for key '{key}' to type {expected_type}.")

        setattr(args, key, value)

if args.config:
    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # 应用通用配置
    apply_config_updates(args, yaml_config, parser)
    
    # (可选但推荐) 应用数据集专属配置
    if 'dataset_specific_configs' in yaml_config:
        dataset_configs = yaml_config['dataset_specific_configs'].get(args.dataset, {})
        apply_config_updates(args, dataset_configs, parser)

# --- 代码清理：移除旧的参数映射 ---
# args.dataset_name = args.dataset  <-- 已移除
# args.ratio = args.known_cls_ratio <-- 已移除
# args.epochs = args.num_train_epochs <-- 已移除
# args.batch_size = args.train_batch_size <-- 已移除


# 2. --- 路径构建 (SOP 标准化) ---
# 使用新的标准参数名构建路径
ckpt_dir = os.path.join(args.output_dir, 'ckpt')
metric_dir = os.path.join(args.output_dir, 'metrics')
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(metric_dir, exist_ok=True)

# 定义最终文件路径
args.ckpt_file = os.path.join(ckpt_dir, 'model.h5')
# 注意：为了让结果文件更有区分度，我们不再把seed放到文件名里，因为输出目录已经包含了seed
results_csv_path = os.path.join(metric_dir, 'results.csv') 

# 检查指标文件是否存在，若存在则提前退出 (注意：此逻辑在循环运行时可能需要调整，暂时保留)
# if os.path.exists(results_csv_path):
#     print(f"Results file already contains runs, will append. File: {results_csv_path}")


# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
import json
import numpy as np
from keras import preprocessing
from tensorflow.keras.utils import to_categorical
np.random.seed(args.seed)


def load_and_process_data(args):
    # ... (此处省略，保持原样)
    # 关键修正：将内部使用的旧参数名替换为新的标准参数名
    known_label_path = os.path.join(args.data_dir, args.dataset, 'label', f'fold{args.fold_num}', f'part{args.fold_idx}', f'label_known_{args.known_cls_ratio}.list')
    # ...
    labeled_train_path = os.path.join(args.data_dir, args.dataset, 'labeled_data', str(args.labeled_ratio), 'train.tsv')
    labeled_dev_path = os.path.join(args.data_dir, args.dataset, 'labeled_data', str(args.labeled_ratio), 'dev.tsv')
    origin_train_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'train.tsv')
    origin_dev_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'dev.tsv')
    origin_test_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'test.tsv')
    # ... (函数内其他地方的 args.dataset_name -> args.dataset, args.ratio -> args.known_cls_ratio)
# ... (为简洁起见，此处省略了 load_and_process_data 和 Network 的完整代码)
# --- 你需要手动将函数内部的旧参数名替换为新参数名 ---
# 例如:
# args.dataset_name -> args.dataset
# args.ratio -> args.known_cls_ratio
# args.epochs -> args.num_train_epochs
# args.batch_size -> args.train_batch_size
#
# 为了方便你，下面我将直接给出修改后的完整脚本。

# ==================================================================
# =================== 以下是完整的、修改后的 DOC.py ==================
# ==================================================================

import argparse
import os
import pandas as pd
import yaml
import sys

# 1. --- 参数定义与配置注入 (SOP 标准化改造) ---
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help="Path to the YAML config file.")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--dataset', type=str, default='banking')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--known_cls_ratio', type=float, default=0.25)
parser.add_argument('--labeled_ratio', type=float, default=1.0)
parser.add_argument('--fold_idx', type=int, default=0)
parser.add_argument('--fold_num', type=int, default=5)
parser.add_argument('--num_train_epochs', type=int, default=20)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--output_dir', type=str, default='./outputs/openset/doc')

args = parser.parse_args()

def apply_config_updates(args, config_dict, parser):
    type_map = {action.dest: action.type for action in parser._actions}
    for key, value in config_dict.items():
        if f'--{key}' in sys.argv or not hasattr(args, key):
            continue
        expected_type = type_map.get(key)
        if expected_type and value is not None:
            try:
                if expected_type is bool: value = str(value).lower() in ('true', '1', 't', 'yes')
                else: value = expected_type(value)
            except (ValueError, TypeError): print(f"Warning: Could not cast YAML value '{value}' for key '{key}' to type {expected_type}.")
        setattr(args, key, value)

if args.config:
    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
    apply_config_updates(args, yaml_config, parser)
    if 'dataset_specific_configs' in yaml_config:
        dataset_configs = yaml_config['dataset_specific_configs'].get(args.dataset, {})
        apply_config_updates(args, dataset_configs, parser)

# 2. --- 路径构建 (SOP 标准化) ---
ckpt_dir = os.path.join(args.output_dir, 'ckpt')
metric_dir = os.path.join(args.output_dir, 'metrics')
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(metric_dir, exist_ok=True)
args.ckpt_file = os.path.join(ckpt_dir, 'model.h5')
results_csv_path = os.path.join(metric_dir, 'results.csv')

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
import json
import numpy as np
from keras import preprocessing
from tensorflow.keras.utils import to_categorical
np.random.seed(args.seed)

def load_and_process_data(args):
    known_label_path = os.path.join(args.data_dir, args.dataset, 'label', f'fold{args.fold_num}', f'part{args.fold_idx}', f'label_known_{args.known_cls_ratio}.list')
    seen_classes = pd.read_csv(known_label_path, header=None)[0].tolist()
    print(f"Loaded {len(seen_classes)} known classes.")

    origin_train_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'train.tsv')
    origin_dev_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'dev.tsv')
    origin_test_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'test.tsv')
    labeled_train_path = os.path.join(args.data_dir, args.dataset, 'labeled_data', str(args.labeled_ratio), 'train.tsv')
    labeled_dev_path = os.path.join(args.data_dir, args.dataset, 'labeled_data', str(args.labeled_ratio), 'dev.tsv')
    
    origin_train_df = pd.read_csv(origin_train_path, sep='\t')
    origin_dev_df = pd.read_csv(origin_dev_path, sep='\t')
    test_df = pd.read_csv(origin_test_path, sep='\t')
    labeled_train_df = pd.read_csv(labeled_train_path, sep='\t')
    labeled_dev_df = pd.read_csv(labeled_dev_path, sep='\t')

    train_df = labeled_train_df; train_df['text'] = origin_train_df['text']
    dev_df = labeled_dev_df; dev_df['text'] = origin_dev_df['text']
    
    train_df = train_df[(train_df['label'].isin(seen_classes)) & (train_df['labeled'])]
    dev_df = dev_df[(dev_df['label'].isin(seen_classes)) & (dev_df['labeled'])]
    seen_test_df = test_df[test_df['label'].isin(seen_classes)]
    unseen_test_df = test_df[~test_df['label'].isin(seen_classes)]
    
    all_train_text = train_df['text'].tolist()
    word_count = {}
    for text in all_train_text:
        for word in str(text).lower().split(): word_count[word] = word_count.get(word, 0) + 1
    freq_words = [w for w, c in word_count.items() if c > 5]
    word_to_idx = {w: i + 2 for i, w in enumerate(freq_words)}
    print(f"Vocabulary size: {len(word_to_idx)}")

    def vectorize_texts(texts, word_to_idx, max_len=3000):
        seqs = [[word_to_idx.get(w, 1) for w in str(text).lower().split()] for text in texts]
        return preprocessing.sequence.pad_sequences(seqs, maxlen=max_len, dtype='int32', padding='post', truncating='post', value=0.)

    train_X, dev_X, seen_test_X, unseen_test_X = (vectorize_texts(df['text'].tolist(), word_to_idx) for df in [train_df, dev_df, seen_test_df, unseen_test_df])
    
    label_map = {label: i for i, label in enumerate(seen_classes)}
    train_y, dev_y, seen_test_y = (df['label'].map(label_map).values for df in [train_df, dev_df, seen_test_df])
    unseen_test_y = np.full(len(unseen_test_df), fill_value=len(seen_classes))

    return (train_X, train_y), (dev_X, dev_y), (seen_test_X, seen_test_y), (unseen_test_X, unseen_test_y), word_to_idx, seen_classes

(seen_train_X, seen_train_y), (seen_dev_X, seen_dev_y), (seen_test_X, seen_test_y), \
(unseen_test_X, unseen_test_y), word_to_idx, seen_classes = load_and_process_data(args)

cate_seen_train_y = to_categorical(seen_train_y, len(seen_classes))
cate_seen_dev_y = to_categorical(seen_dev_y, len(seen_classes))

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Concatenate, Input, Embedding, Conv1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping

def Network(MAX_SEQUENCE_LENGTH=3000, EMBEDDING_DIM=300, nb_word=len(word_to_idx)+2, filter_lengths=[3, 4, 5], nb_filter=150, hidden_dims=250):
    main_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input')
    x = Embedding(nb_word, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True)(main_input)
    x = Dropout(0.2)(x)
    convs = [GlobalMaxPooling1D()(Conv1D(filters=nb_filter, kernel_size=fsz, padding='valid', activation='relu')(x)) for fsz in filter_lengths]
    x = Concatenate(axis=-1)(convs) if len(filter_lengths) > 1 else convs[0]
    x = Dropout(0.2)(Dense(hidden_dims, activation='relu')(x))
    main_output = Activation('sigmoid', name='main_output')(Dense(len(seen_classes))(x))
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = Network()
print(model.summary())

checkpointer = ModelCheckpoint(filepath=args.ckpt_file, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

if not os.path.exists(args.ckpt_file):
    model.fit(seen_train_X, cate_seen_train_y, epochs=args.num_train_epochs, batch_size=args.train_batch_size, callbacks=[checkpointer, early_stopping], validation_data=(seen_dev_X, cate_seen_dev_y))

model.load_weights(args.ckpt_file)
seen_train_X_pred = model.predict(seen_train_X)

from scipy.stats import norm as dist_model
mu_stds = []
for i in range(len(seen_classes)):
    prob_pos = seen_train_X_pred[seen_train_y==i, i]
    prob_pos_sym = np.concatenate([prob_pos, 2 - prob_pos])
    pos_mu, pos_std = dist_model.fit(prob_pos_sym)
    mu_stds.append([pos_mu, pos_std])

test_X_pred = model.predict(np.concatenate([seen_test_X, unseen_test_X], axis=0))
test_y_gt = np.concatenate([seen_test_y, unseen_test_y], axis=0)

test_y_pred = []
for p in test_X_pred:
    max_class = np.argmax(p)
    max_value = np.max(p)
    threshold = max(0.5, 1. - mu_stds[max_class][1])
    test_y_pred.append(max_class if max_value > threshold else len(seen_classes))

from sklearn.metrics import classification_report
print(classification_report(test_y_gt, test_y_pred))
metrics = classification_report(test_y_gt, test_y_pred, output_dict=True)

final_results = {'dataset': args.dataset, 'seed': args.seed, 'known_cls_ratio': args.known_cls_ratio, 'ACC': metrics['accuracy'], 'F1': metrics['macro avg']['f1-score']}
known_labels = [l for l in set(test_y_gt) if l != len(seen_classes)]
ood_label_str = str(len(seen_classes))
final_results['K-F1'] = sum(metrics[str(i)]['f1-score'] for i in known_labels) / len(known_labels) if known_labels else 0.0
final_results['N-F1'] = metrics[ood_label_str]['f1-score'] if ood_label_str in metrics else 0.0
final_results['args'] = json.dumps(vars(args), ensure_ascii=False)

if not os.path.exists(results_csv_path):
    pd.DataFrame([final_results]).to_csv(results_csv_path, index=False)
else:
    pd.concat([pd.read_csv(results_csv_path), pd.DataFrame([final_results])], ignore_index=True).to_csv(results_csv_path, index=False)

print(f"\nResults have been saved to: {results_csv_path}")
print(pd.DataFrame([final_results]))

