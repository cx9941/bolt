import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="news", type=str)
parser.add_argument("--proportion", default=0.25, type=float)
parser.add_argument("--n_epochs", default=8, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu_id", default="0", type=str)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--batch_size", default=16, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

from utils import *
set_allow_growth()


import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Modeling
from models import BiLSTM_LMCL
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import random as rn

# Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor



embedding_path = './glove_embeddings/'
EMBEDDING_FILE = os.path.join(embedding_path, 'glove.6B.300d.txt')
if not os.path.exists(f"metrics/{args.dataset_name}"):
    os.makedirs(f"metrics/{args.dataset_name}")
args.metric_path = f"metrics/{args.dataset_name}/{args.proportion}_{args.seed}.json"
if os.path.exists(args.metric_path):
    exit()
MAX_SEQ_LEN = None
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 300

np.random.seed(args.seed)
rn.seed(args.seed)
tf.set_random_seed(args.seed)

df, partition_to_n_row = load_data(args.dataset_name)
df['text'] = df['text'].apply(lambda x: ' '.join(x.split(' ')[:200]))

df['content_words'] = df['text'].apply(lambda s: word_tokenize(s))
texts = df['content_words'].apply(lambda l: " ".join(l)) 

# Do not filter out "," and "."
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<UNK>", filters='!"#$%&()*+-/:;<=>@[\]^_`{|}~') 

tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
sequences_pad = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

# Train-valid-test split
idx_train = (None, partition_to_n_row['train'])
idx_valid = (partition_to_n_row['train'], partition_to_n_row['train'] + partition_to_n_row['valid'])
idx_test = (partition_to_n_row['train'] + partition_to_n_row['valid'], None)

X_train = sequences_pad[idx_train[0]:idx_train[1]]
X_valid = sequences_pad[idx_valid[0]:idx_valid[1]]
X_test = sequences_pad[idx_test[0]:idx_test[1]]

df_train = df[idx_train[0]:idx_train[1]]
df_valid = df[idx_valid[0]:idx_valid[1]]
df_test = df[idx_test[0]:idx_test[1]]

y_train = df_train.label.reset_index(drop=True)
y_valid = df_valid.label.reset_index(drop=True)
y_test = df_test.label.reset_index(drop=True)
print("train : valid : test = %d : %d : %d" % (X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

print("Load pre-trained GloVe embedding...")
MAX_FEATURES = min(MAX_NUM_WORDS, len(word_index)) + 1  # +1 for PAD

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embedding_matrix = np.random.normal(emb_mean, emb_std, (MAX_FEATURES, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_FEATURES: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

n_class = y_train.unique().shape[0]

# 已知类
# y_cols_seen = y_train.value_counts().index[:n_class_seen]
# y_cols_unseen = y_train.value_counts().index[n_class_seen:]

y_cols_seen = pd.read_csv(f'data/{args.dataset_name}/known_class_{args.proportion}.txt', header=None)[0].tolist()
y_cols_unseen = list(set(y_train.value_counts().index) - set(y_cols_seen))

n_class_seen = len(y_cols_seen)

train_seen_idx = y_train[y_train.isin(y_cols_seen)].index
valid_seen_idx = y_valid[y_valid.isin(y_cols_seen)].index

print(train_seen_idx)
print(valid_seen_idx)

X_train_seen = X_train[train_seen_idx]
y_train_seen = y_train[train_seen_idx]
X_valid_seen = X_valid[valid_seen_idx]
y_valid_seen = y_valid[valid_seen_idx]

le = LabelEncoder()
le.fit(y_train_seen)
y_train_idx = le.transform(y_train_seen)
y_valid_idx = le.transform(y_valid_seen)

y_train_onehot = to_categorical(y_train_idx)
y_valid_onehot = to_categorical(y_valid_idx)

y_test_mask = y_test.copy()
y_test_mask[y_test_mask.isin(y_cols_unseen)] = 'unseen'

filepath = 'ckpt/BiLSTM_' + args.dataset_name + "_" + str(args.proportion) + f'-{args.seed}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, 
                             save_best_only=True, mode='auto', save_weights_only=False)
early_stop = EarlyStopping(monitor='val_loss', patience=20, mode='auto') 
callbacks_list = [checkpoint, early_stop]

train_data = (X_train_seen, y_train_onehot)
valid_data = (X_valid_seen, y_valid_onehot)
test_data = (X_test, y_test_mask)

## If you want to plot the model
# model = BiLSTM_LMCL(MAX_SEQ_LEN, MAX_FEATURES, EMBEDDING_DIM, n_class_seen, 'img/model.png', embedding_matrix)
model = BiLSTM_LMCL(MAX_SEQ_LEN, MAX_FEATURES, EMBEDDING_DIM, n_class_seen, None, embedding_matrix)
history = model.fit(train_data[0], train_data[1], epochs=args.n_epochs, batch_size=args.batch_size, 
                    validation_data=valid_data, shuffle=True, verbose=1, callbacks=callbacks_list)
                    
y_pred_proba = model.predict(test_data[0])
y_pred_proba_train = model.predict(train_data[0])
classes = list(le.classes_) + ['unseen']

method = 'LOF (LMCL)'
get_deep_feature = Model(inputs=model.input, 
                         outputs=model.layers[-3].output)
feature_test = get_deep_feature.predict(test_data[0])
feature_train = get_deep_feature.predict(train_data[0])
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)
lof.fit(feature_train)

y_pred_lof = pd.Series(lof.predict(feature_test))
df_seen = pd.DataFrame(y_pred_proba, columns=le.classes_)
df_seen['unseen'] = 0

y_pred = df_seen.idxmax(axis=1)
y_pred[y_pred_lof[y_pred_lof==-1].index]='unseen'

from sklearn.metrics import classification_report 
import json

metrics = classification_report(test_data[1], y_pred, output_dict=True)
seen_classes = [i for i in test_data[1].tolist() if i != 'unseen']
print(classification_report(test_data[1], y_pred))
final_metrics = {}
final_metrics['N-F1'] = metrics['unseen']['f1-score']
final_metrics['K-F1'] = sum([metrics[i]['f1-score'] for i in seen_classes]) / len(seen_classes)
final_metrics['ACC'] = metrics['accuracy']
final_metrics['F1'] = metrics['macro avg']['f1-score']
json.dump(final_metrics, open(args.metric_path, 'w'))