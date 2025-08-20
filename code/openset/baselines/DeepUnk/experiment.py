import os
import argparse
import json
import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model

from models import BiLSTM_LMCL
from utils import set_allow_growth

def main(args):
    # 1. 初始化和环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    set_allow_growth(args.gpu_id)
    
    np.random.seed(args.seed)
    rn.seed(args.seed)
    tf.random.set_seed(args.seed)

    # 2. (核心改造) 标准化数据加载
    print("Loading standardized data...")
    # a. 加载已知类列表
    known_label_path = os.path.join(args.data_dir, args.dataset, 'label', f'fold{args.fold_num}', f'part{args.fold_idx}', f'label_known_{args.known_cls_ratio}.list')
    seen_labels = pd.read_csv(known_label_path, header=None)[0].tolist()
    n_class_seen = len(seen_labels)

    # b. 加载 .tsv 数据文件
    origin_train_df = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'origin_data', 'train.tsv'), sep='\t')
    origin_valid_df = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'origin_data', 'dev.tsv'), sep='\t')
    origin_test_df = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'origin_data', 'test.tsv'), sep='\t')
    
    labeled_train_df = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'labeled_data', str(args.labeled_ratio), 'train.tsv'), sep='\t')
    
    df_train = labeled_train_df
    df_train['text'] = origin_train_df['text']
    df_valid = origin_valid_df # 使用完整的 dev.tsv 作为验证集
    df_test = origin_test_df  # 使用完整的 test.tsv 作为测试集

    # c. 筛选数据
    train_seen_df = df_train[df_train['label'].isin(seen_labels)]
    valid_seen_df = df_valid[df_valid['label'].isin(seen_labels)]
    
    all_labels = df_train['label'].unique().tolist()
    y_cols_unseen = [l for l in all_labels if l not in seen_labels]

    # 3. (核心改造) 文本预处理 (修复数据泄露)
    print("Preprocessing text data...")
    tokenizer = Tokenizer(num_words=10000, oov_token="<UNK>", filters='!"#$%&()*+-/:;<=>@[\]^_`{|}~')
    # 仅在训练集上训练tokenizer
    tokenizer.fit_on_texts(train_seen_df['text'].astype(str))
    word_index = tokenizer.word_index
    
    X_train_seen = pad_sequences(tokenizer.texts_to_sequences(train_seen_df['text'].astype(str)), padding='post', truncating='post')
    X_valid_seen = pad_sequences(tokenizer.texts_to_sequences(valid_seen_df['text'].astype(str)), padding='post', truncating='post')
    X_test = pad_sequences(tokenizer.texts_to_sequences(df_test['text'].astype(str)), padding='post', truncating='post')
    
    # 4. 标签编码
    le = LabelEncoder()
    le.fit(train_seen_df['label'])
    y_train_idx = le.transform(train_seen_df['label'])
    y_valid_idx = le.transform(valid_seen_df['label'])
    
    y_train_onehot = to_categorical(y_train_idx, num_classes=n_class_seen)
    y_valid_onehot = to_categorical(y_valid_idx, num_classes=n_class_seen)
    
    y_test_mask = df_test['label'].copy()
    y_test_mask[~y_test_mask.isin(seen_labels)] = 'unseen'

    # 5. 加载Glove词向量 (逻辑保留)
    print("Loading GloVe embeddings...")
    MAX_FEATURES = min(10000, len(word_index)) + 1
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(args.embedding_file))
    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embedding_matrix = np.random.normal(emb_mean, emb_std, (MAX_FEATURES, 300))
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    # 6. 模型训练 (逻辑保留)
    print("Training model...")
    # (核心改造) 标准化输出路径
    os.makedirs(os.path.join(args.output_dir, 'ckpt'), exist_ok=True)
    filepath = os.path.join(args.output_dir, 'ckpt', f'model_{args.dataset}_{args.known_cls_ratio}_{args.seed}.h5')
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', save_weights_only=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=20, mode='auto') 
    
    model = BiLSTM_LMCL(
        max_seq_len=None,
        max_features=MAX_FEATURES,
        embedding_dim=300,
        output_dim=n_class_seen,
        embedding_matrix=embedding_matrix,
        learning_rate=args.learning_rate,
        model_img_path=None
    )
    model.fit(X_train_seen, y_train_onehot, epochs=args.n_epochs, batch_size=args.train_batch_size, 
              validation_data=(X_valid_seen, y_valid_onehot), shuffle=True, verbose=1, callbacks=[checkpoint, early_stop])

    # 7. 评估 (逻辑保留)
    print("Evaluating model...")
    y_pred_proba = model.predict(X_test)
    
    get_deep_feature = Model(inputs=model.inputs, outputs=model.layers[-3].output)
    feature_train = get_deep_feature.predict(X_train_seen)
    feature_test = get_deep_feature.predict(X_test)
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)
    lof.fit(feature_train)
    y_pred_lof = pd.Series(lof.predict(feature_test))
    
    df_seen = pd.DataFrame(y_pred_proba, columns=le.classes_)
    y_pred = df_seen.idxmax(axis=1)
    y_pred[y_pred_lof[y_pred_lof==-1].index] = 'unseen'
    
    # 8. (核心改造) 标准化结果输出
    print("Saving results...")
    report = classification_report(y_test_mask, y_pred, output_dict=True, zero_division=0)
    
    final_results = {}
    final_results['dataset'] = args.dataset
    final_results['seed'] = args.seed
    final_results['known_cls_ratio'] = args.known_cls_ratio
    final_results['ACC'] = report['accuracy']
    final_results['F1'] = report['macro avg']['f1-score']
    
    known_f1_scores = [report[label]['f1-score'] for label in le.classes_ if label in report]
    final_results['K-F1'] = sum(known_f1_scores) / len(known_f1_scores) if known_f1_scores else 0.0
    final_results['N-F1'] = report['unseen']['f1-score'] if 'unseen' in report else 0.0

    metric_dir = os.path.join(args.output_dir, 'metrics')
    os.makedirs(metric_dir, exist_ok=True)
    results_path = os.path.join(metric_dir, 'results.csv')

    if not os.path.exists(results_path):
        df_to_save = pd.DataFrame([final_results])
        df_to_save.to_csv(results_path, index=False)
    else:
        existing_df = pd.read_csv(results_path)
        new_row_df = pd.DataFrame([final_results])
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        updated_df.to_csv(results_path, index=False)
        
    print(f"\nResults have been saved to: {results_path}")
    print("Appended new result row:")
    print(pd.DataFrame([final_results]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 添加所有在YAML中定义的参数
    parser.add_argument("--dataset", type=str, default="banking")
    parser.add_argument("--known_cls_ratio", type=float, default=0.25)
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--labeled_ratio", type=float, default=1.0)
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--fold_num", type=int, default=5)
    parser.add_argument("--embedding_file", type=str, default="./pretrained_models/glove.6B.300d.txt")
    parser.add_argument("--output_dir", type=str, default="./outputs/openset/deepunk")
    
    args = parser.parse_args()
    main(args)