import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='atis')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--ratio', type=float, default=0.5)
args = parser.parse_args()

args.metric_file = f"metrics/{args.dataset_name}/{args.ratio}/{args.seed}.json"
args.ckpt_file = f"ckpt/{args.dataset_name}/{args.ratio}/{args.seed}.h5"

if os.path.exists(args.metric_file):
    exit()

os.makedirs(f"ckpt/{args.dataset_name}/{args.ratio}", exist_ok=True)
os.makedirs(f"metrics/{args.dataset_name}/{args.ratio}", exist_ok=True)
    
import os
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
import json
import numpy as np
from keras import preprocessing
np.random.seed(args.seed)

fn = f'data/{args.dataset_name}.json' #origial review documents, there are 50 classes
with open(fn, 'r') as infile:
        docs = json.load(infile)
X = docs['X']
y = np.asarray(docs['y'])
num_classes = len(docs['target_names'])

#count each word's occurance
def count_word(X):
    word_count = dict()
    for d in X:
        for w in d.lower().split(' '): #lower
            if w in word_count:
                word_count[w] += 1
            else:
                word_count[w] = 1            
    return word_count

word_count = count_word(X)
print('total words: ', len(word_count))

#get frequent words
freq_words = [w  for w, c in word_count.items() if c > 10]
print('frequent word size = ', len(freq_words))

#word index
word_to_idx = {w: i+2  for i, w in enumerate(freq_words)} # index 0 for padding, index 1 for unknown/rare words
idx_to_word = {i:w for w, i in word_to_idx.items()}

def index_word(X):
    seqs = []
    max_length = 0
    for d in X:
        seq = []
        for w in d.lower().split():
            if w in word_to_idx:
                seq.append(word_to_idx[w])
            else:
                seq.append(1) #rare word index = 1
        seqs.append(seq)
    return seqs

#index documents and pad each review to length = 3000
indexed_X = index_word(X)
padded_X = preprocessing.sequence.pad_sequences(indexed_X, maxlen=3000, dtype='int32', padding='post', truncating='post', value = 0.)

#split review into training and testing set
train_X,  test_X = padded_X[docs[f"X_{args.ratio}_train"]], padded_X[docs[f"X_{args.ratio}_test"]]
train_y,  test_y = y[docs[f"X_{args.ratio}_train"]], y[docs[f"X_{args.ratio}_test"]]
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#split reviews into seen classes and unseen classes
def splitSeenUnseen(X, y, seen, unseen):
    seen_mask = np.in1d(y, seen)# find examples whose label is in seen classes
    unseen_mask = np.in1d(y, unseen)# find examples whose label is in unseen classes
    
    print(np.array_equal(np.logical_and(seen_mask, unseen_mask), np.zeros((y.shape), dtype= bool)))#expect to see 'True', check two masks are exclusive
    
    # map elements in y to [0, ..., len(seen)] based on seen, map y to unseen_label when it belongs to unseen classes
    to_seen = {l:i for i, l in enumerate(seen)}
    unseen_label = len(seen)
    to_unseen = {l:unseen_label for l in unseen}
        
    return X[seen_mask], np.vectorize(to_seen.get)(y[seen_mask]), X[unseen_mask], np.vectorize(to_unseen.get)(y[unseen_mask])

seen = docs[f"target_{args.ratio}"]
unseen = [i for i in range(num_classes) if i not in seen]

seen_train_X, seen_train_y, _, _ = splitSeenUnseen(train_X, train_y, seen, unseen)
seen_test_X, seen_test_y, unseen_test_X, unseen_test_y = splitSeenUnseen(test_X, test_y, seen, unseen)


# from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical
cate_seen_train_y = to_categorical(seen_train_y, len(seen))#make train y to categorial/one hot vectors

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
            Dense(len(seen) ),
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
              epochs=args.epochs, batch_size=args.batch_size, callbacks=[checkpointer, early_stopping], validation_split=0.2)

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
for i in range(len(seen)):
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
        test_y_pred.append(len(seen))#otherwise, reject
  

#evaluate
from sklearn.metrics import classification_report
print(classification_report(test_y_gt, test_y_pred))
label_list = list(set(test_y_gt.tolist()))
metrics = classification_report(test_y_gt, test_y_pred, output_dict=True)
final_metrics = metrics['macro avg']
final_metrics['F1'] = final_metrics['f1-score']
final_metrics['ACC'] = metrics['accuracy']
final_metrics['K-F1'] = sum([metrics[str(i)]['f1-score'] for i in label_list if i !=0]) / (len(label_list) - 1)
final_metrics['N-F1'] = metrics[str(max(label_list))]['f1-score']
json.dump(final_metrics, open(args.metric_file, 'w'))