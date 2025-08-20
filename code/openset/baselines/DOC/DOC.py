import argparse
import os
import pandas as pd

# 1. --- 参数定义 (SOP 标准化) ---
parser = argparse.ArgumentParser()
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
args = parser.parse_args()

# 为了兼容原代码，我们把标准参数名映射回原有的参数名
args.dataset_name = args.dataset
args.ratio = args.known_cls_ratio
args.epochs = args.num_train_epochs
args.batch_size = args.train_batch_size

# 2. --- 路径构建 (SOP 标准化) ---
# 使用 output_dir 构建检查点和指标文件的完整路径
ckpt_dir = os.path.join(args.output_dir, 'ckpt', f'{args.dataset_name}_{args.ratio}_{args.seed}')
metric_dir = os.path.join(args.output_dir, 'metrics', f'{args.dataset_name}_{args.ratio}')
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(metric_dir, exist_ok=True)

# 定义最终文件路径
args.ckpt_file = os.path.join(ckpt_dir, 'model.h5')
args.metric_file = os.path.join(metric_dir, f'{args.seed}.json')

# 检查指标文件是否存在，若存在则提前退出
if os.path.exists(args.metric_file):
    print(f"Metric file already exists, skipping: {args.metric_file}")
    exit()

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
import json
import numpy as np
from keras import preprocessing
# from keras.utils.np_utils import to_categorical  (旧的)
from tensorflow.keras.utils import to_categorical # (新的)
np.random.seed(args.seed)

def load_and_process_data(args):
    """
    SOP标准化的数据加载与预处理函数 for DOC. (已修正)
    """
    # 1. 加载已知类列表
    known_label_path = os.path.join(args.data_dir, args.dataset_name, 'label', f'fold{args.fold_num}', f'part{args.fold_idx}', f'label_known_{args.ratio}.list')
    seen_classes = pd.read_csv(known_label_path, header=None)[0].tolist()
    print(f"Loaded {len(seen_classes)} known classes.")

    # 2. 加载数据文件 (关键修正：分别加载 origin 和 labeled 数据)
    # 路径定义
    origin_train_path = os.path.join(args.data_dir, args.dataset_name, 'origin_data', 'train.tsv')
    origin_dev_path = os.path.join(args.data_dir, args.dataset_name, 'origin_data', 'dev.tsv')
    origin_test_path = os.path.join(args.data_dir, args.dataset_name, 'origin_data', 'test.tsv')
    
    labeled_train_path = os.path.join(args.data_dir, args.dataset_name, 'labeled_data', str(args.labeled_ratio), 'train.tsv')
    labeled_dev_path = os.path.join(args.data_dir, args.dataset_name, 'labeled_data', str(args.labeled_ratio), 'dev.tsv')

    # 读取包含文本的 origin data
    origin_train_df = pd.read_csv(origin_train_path, sep='\t')
    origin_dev_df = pd.read_csv(origin_dev_path, sep='\t')
    test_df = pd.read_csv(origin_test_path, sep='\t') # 测试集直接使用 origin data

    # 读取包含标签划分的 labeled data
    labeled_train_df = pd.read_csv(labeled_train_path, sep='\t')
    labeled_dev_df = pd.read_csv(labeled_dev_path, sep='\t')
    
    # 关键修正：将文本(origin)和标签(labeled)合并
    train_df = labeled_train_df
    train_df['text'] = origin_train_df['text']
    
    dev_df = labeled_dev_df
    dev_df['text'] = origin_dev_df['text']

    # 3. 筛选数据 (此部分逻辑不变)
    # 训练集和验证集只包含已知类
    train_df = train_df[train_df['label'].isin(seen_classes)]
    dev_df = dev_df[dev_df['label'].isin(seen_classes)]
    
    # 测试集分为已知类和未知类
    seen_test_df = test_df[test_df['label'].isin(seen_classes)]
    unseen_test_df = test_df[~test_df['label'].isin(seen_classes)]
    
    # 4. 构建词汇表 (只在训练数据上构建)
    all_train_text = train_df['text'].tolist()
    word_count = {}
    for text in all_train_text:
        for word in str(text).lower().split():
            word_count[word] = word_count.get(word, 0) + 1
    
    freq_words = [w for w, c in word_count.items() if c > 5]
    word_to_idx = {w: i + 2 for i, w in enumerate(freq_words)} # 0 for padding, 1 for OOV
    print(f"Vocabulary size: {len(word_to_idx)}")

    # 5. 文本向量化与填充
    def vectorize_texts(texts, word_to_idx, max_len=3000):
        seqs = []
        for text in texts:
            seq = [word_to_idx.get(w, 1) for w in str(text).lower().split()]
            seqs.append(seq)
        return preprocessing.sequence.pad_sequences(seqs, maxlen=max_len, dtype='int32', padding='post', truncating='post', value=0.)

    train_X = vectorize_texts(train_df['text'].tolist(), word_to_idx)
    dev_X = vectorize_texts(dev_df['text'].tolist(), word_to_idx)
    seen_test_X = vectorize_texts(seen_test_df['text'].tolist(), word_to_idx)
    unseen_test_X = vectorize_texts(unseen_test_df['text'].tolist(), word_to_idx)
    
    # 6. 标签处理
    label_map = {label: i for i, label in enumerate(seen_classes)}
    train_y = train_df['label'].map(label_map).values
    dev_y = dev_df['label'].map(label_map).values
    seen_test_y = seen_test_df['label'].map(label_map).values
    unseen_test_y = np.full(len(unseen_test_df), fill_value=len(seen_classes))

    return (train_X, train_y), (dev_X, dev_y), (seen_test_X, seen_test_y), (unseen_test_X, unseen_test_y), word_to_idx, seen_classes
# --- 调用新的数据加载函数 ---
(seen_train_X, seen_train_y), (seen_dev_X, seen_dev_y), (seen_test_X, seen_test_y), \
(unseen_test_X, unseen_test_y), word_to_idx, seen_classes = load_and_process_data(args)

# 将训练和验证标签转为 one-hot 编码
cate_seen_train_y = to_categorical(seen_train_y, len(seen_classes))
cate_seen_dev_y = to_categorical(seen_dev_y, len(seen_classes))
#Network, in the paper, I use pretrained google news embedding, here I do not use it and set the embedding layer trainable
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding, Input, Concatenate
from keras.layers import Conv1D, GlobalMaxPooling1D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import backend as K

def Network(MAX_SEQUENCE_LENGTH = 3000, EMBEDDING_DIM = 300, nb_word = len(word_to_idx)+2, filter_lengths = [3, 4, 5],
    nb_filter = 150, hidden_dims =250):
    
    graph_in = Input(shape=(MAX_SEQUENCE_LENGTH,  EMBEDDING_DIM))
    convs = []
    for fsz in filter_lengths:
        conv = Conv1D(filters=nb_filter,
                                 kernel_size=fsz,
                                 padding='valid',
                                 activation='relu')(graph_in)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)

    if len(filter_lengths)>1:
        out = Concatenate(axis=-1)(convs)
    else:
        out = convs[0]

    graph = Model(inputs=graph_in, outputs=out) #convolution layers
    
    emb_layer = [Embedding(nb_word,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True),
                 Dropout(0.2)
        ]
    conv_layer = [
            graph,
        ]
    feature_layers1 = [
            Dense(hidden_dims),
            Dropout(0.2),
            Activation('relu')
    ]
    feature_layers2 = [
            Dense(len(seen_classes)),
            Dropout(0.2),
    ]
    output_layer = [
            Activation('sigmoid')
    ]

    model = Sequential(emb_layer+conv_layer+feature_layers1+feature_layers2+output_layer)
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model

model = Network()    
print(model.summary())

checkpointer = ModelCheckpoint(filepath=args.ckpt_file, verbose=1, save_best_only=True)
early_stopping=EarlyStopping(monitor='val_loss', patience=5)


if not os.path.exists(args.ckpt_file):
    model.fit(seen_train_X, cate_seen_train_y,
          epochs=args.epochs, batch_size=args.batch_size, 
          callbacks=[checkpointer, early_stopping], 
          validation_data=(seen_dev_X, cate_seen_dev_y))

model.load_weights(args.ckpt_file)

seen_train_X_pred = model.predict(seen_train_X)
print(seen_train_X_pred.shape)

#fit a gaussian model
from scipy.stats import norm as dist_model
def fit(prob_pos_X):
    prob_pos = [p for p in prob_pos_X]+[2-p for p in prob_pos_X]
    pos_mu, pos_std = dist_model.fit(prob_pos)
    return pos_mu, pos_std


#calculate mu, std of each seen class
mu_stds = []
for i in range(len(seen_classes)):
    pos_mu, pos_std = fit(seen_train_X_pred[seen_train_y==i, i])
    mu_stds.append([pos_mu, pos_std])

print(mu_stds)

#predict on test examples
test_X_pred = model.predict(np.concatenate([seen_test_X,unseen_test_X], axis = 0))
test_y_gt = np.concatenate([seen_test_y,unseen_test_y], axis = 0)
print(test_X_pred.shape, test_y_gt.shape)

#get prediction based on threshold
test_y_pred = []
scale = 1.
for p in test_X_pred:# loop every test prediction
    max_class = np.argmax(p)# predicted class
    max_value = np.max(p)# predicted probability
    threshold = max(0.5, 1. - scale * mu_stds[max_class][1])#find threshold for the predicted class
    if max_value > threshold:
        test_y_pred.append(max_class)#predicted probability is greater than threshold, accept
    else:
        test_y_pred.append(len(seen_classes))#otherwise, reject
  

# evaluate
from sklearn.metrics import classification_report
# 仍然在控制台打印完整报告，方便实时查看
print(classification_report(test_y_gt, test_y_pred)) 

metrics = classification_report(test_y_gt, test_y_pred, output_dict=True)
label_list = list(set(test_y_gt.tolist()))

# --- 步骤1：创建一个只包含最终指标的“单行”结果字典--
final_results = {}

# a. 添加实验参数元数据
final_results['dataset'] = args.dataset
final_results['seed'] = args.seed
final_results['known_cls_ratio'] = args.known_cls_ratio

# b. 添加核心的总览性能指标
final_results['ACC'] = metrics['accuracy']
final_results['F1'] = metrics['macro avg']['f1-score']

# c. 添加自定义的 K-F1 和 N-F1 指标
known_labels = [l for l in label_list if l != len(seen_classes)]
ood_label = len(seen_classes)

if known_labels:
    final_results['K-F1'] = sum([metrics[str(i)]['f1-score'] for i in known_labels]) / len(known_labels)
else:
    final_results['K-F1'] = 0.0
final_results['N-F1'] = metrics[str(ood_label)]['f1-score'] if str(ood_label) in metrics else 0.0

# --- 步骤2：将“单行”结果追加保存到主 results.csv 文件 (模仿ADB逻辑) ---
results_path = os.path.join(metric_dir, 'results.csv')

if not os.path.exists(results_path):
    # 如果文件不存在，直接创建并写入（包含表头）
    df_to_save = pd.DataFrame([final_results])
    df_to_save.to_csv(results_path, index=False)
else:
    # 如果文件存在，则读取->追加->写回，以保证格式统一
    existing_df = pd.read_csv(results_path)
    new_row_df = pd.DataFrame([final_results])
    # 使用 pd.concat 替代已废弃的 ._append
    updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    updated_df.to_csv(results_path, index=False)

print(f"\nResults have been saved to: {results_path}")
print("Appended new result row:")
print(pd.DataFrame([final_results]))