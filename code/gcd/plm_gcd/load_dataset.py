import torch
import pandas as pd
from torch.utils.data import DataLoader
from configs import args
from datasets import concatenate_datasets

known_label_list = pd.read_csv(f'{args.data_dir}/{args.dataset}/label/fold{args.fold_num}/part{args.fold_idx}/label_known_{args.known_cls_ratio}.list', header=None)[0].tolist()
all_label_list = pd.read_csv(f'{args.data_dir}/{args.dataset}/label/label.list', header=None)[0].tolist()
all_label_list = known_label_list + [i for i in all_label_list if i not in known_label_list]
## origin data
origin_train_data = pd.read_csv(f'{args.data_dir}/{args.dataset}/origin_data/train.tsv', sep='\t')
origin_eval_data = pd.read_csv(f'{args.data_dir}/{args.dataset}/origin_data/dev.tsv', sep='\t')
origin_test_data = pd.read_csv(f'{args.data_dir}/{args.dataset}/origin_data/test.tsv', sep='\t')


## id train data
train_data = pd.read_csv(f'{args.data_dir}/{args.dataset}/labeled_data/{args.labeled_ratio}/train.tsv', sep='\t')
train_data['text'] =  origin_train_data['text']
train_data = train_data[(train_data['label'].isin(known_label_list)) & (train_data['labeled'])].drop('labeled',  axis=1)

## id eval data
eval_data = pd.read_csv(f'{args.data_dir}/{args.dataset}/labeled_data/{args.labeled_ratio}/dev.tsv', sep='\t')
eval_data['text'] =  origin_eval_data['text']
data_in_eval = eval_data[(eval_data['label'].isin(known_label_list)) & (eval_data['labeled'])].drop('labeled',  axis=1)

## all test data
test_data = pd.read_csv(f'{args.data_dir}/{args.dataset}/origin_data/test.tsv', sep='\t')
data_in_test = test_data[test_data['label'].isin(known_label_list)]
data_out_test = test_data[~test_data['label'].isin(known_label_list)]
# data_out_test['label'] = 'ood'


train_data['labels'] = train_data['label'].apply(lambda x: known_label_list.index(x) if x in known_label_list else -1)
data_in_eval['labels'] = data_in_eval['label'].apply(lambda x: known_label_list.index(x) if x in known_label_list else -1)
data_in_test['labels'] = data_in_test['label'].apply(lambda x: all_label_list.index(x) if x in known_label_list else -1)
data_out_test['labels'] = data_out_test['label'].apply(lambda x: all_label_list.index(x))

args.num_labels = len(known_label_list)
args.all_num_labels = len(all_label_list)
# 检查数据格式
print(train_data.head())

from datasets import Dataset
from transformers import AutoTokenizer

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_special_tokens({'eos_token': '[END]'})

# 将Pandas DataFrame转换为 Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data.reset_index(drop=True))
dataset_in_eval = Dataset.from_pandas(data_in_eval.reset_index(drop=True))
dataset_in_test = Dataset.from_pandas(data_in_test.reset_index(drop=True))
dataset_out_test = Dataset.from_pandas(data_out_test.reset_index(drop=True))


# 定义tokenizer函数
def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True, max_length=60)

# 对数据集进行tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
dataset_in_test = dataset_in_test.map(tokenize_function, batched=True)
dataset_in_eval = dataset_in_eval.map(tokenize_function, batched=True)
dataset_out_test = dataset_out_test.map(tokenize_function, batched=True)


# 删除无关列
train_dataset = train_dataset.remove_columns(["text", 'label'])
dataset_in_test = dataset_in_test.remove_columns(["text", 'label'])
dataset_in_eval = dataset_in_eval.remove_columns(["text", 'label'])
dataset_out_test = dataset_out_test.remove_columns(["text", 'label'])


test_dataset = concatenate_datasets([dataset_in_test, dataset_out_test])

# 设置数据格式为 PyTorch tensors
train_dataset.set_format("torch")
dataset_in_test.set_format("torch")
dataset_in_eval.set_format("torch")
dataset_out_test.set_format("torch")

test_dataset.set_format("torch")

def collate_batch(batch, max_len=512):
    ans = {}
    max_len = max([i['input_ids'].shape[0] for i in batch])
    for key in batch[0]:
        if key in ['input_ids', 'attention_mask', 'token_type_ids']:
            padded = [i[key].tolist() + (max_len - len(i[key])) * [0 if key != 'input_ids' else tokenizer.pad_token_id] for i in batch]
            ans[key] = torch.tensor(padded, dtype=torch.long)
        else:
            ans[key] = torch.stack([i[key] for i in batch])
    return ans

loader_in_train = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_batch)
loader_in_test = DataLoader(dataset_in_test, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_batch)
loader_in_eval = DataLoader(dataset_in_eval, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_batch)

test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.eval_batch_size, collate_fn=collate_batch)
