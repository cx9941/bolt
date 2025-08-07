import torch
import pandas as pd
from torch.utils.data import DataLoader
from configs import args
from datasets import concatenate_datasets


train_data = pd.read_csv(f'data/{args.dataset_name}/{args.dataset_name}_{args.rate}/train.csv', sep='\t')
data_in_eval = pd.read_csv(f'data/{args.dataset_name}/{args.dataset_name}_{args.rate}/eval_seen.csv', sep='\t')
data_out_eval = pd.read_csv(f'data/{args.dataset_name}/{args.dataset_name}_{args.rate}/eval_unseen.csv', sep='\t')
data_in_test = pd.read_csv(f'data/{args.dataset_name}/{args.dataset_name}_{args.rate}/test_seen.csv', sep='\t')
data_out_test = pd.read_csv(f'data/{args.dataset_name}/{args.dataset_name}_{args.rate}/test_unseen.csv', sep='\t')

taxonomy = pd.read_csv(f'data/{args.dataset_name}/{args.dataset_name}_{args.rate}/taxonomy.csv', sep='\t').reset_index()
levels = train_data.columns[:-1].tolist()
device = 'cuda'
train_data = pd.merge(train_data, taxonomy, on=levels)[['index', 'text']]
data_in_test = pd.merge(data_in_test, taxonomy, on=levels)[['index', 'text']]
data_in_eval = pd.merge(data_in_eval, taxonomy, on=levels)[['index', 'text']]
data_out_eval['index'] = -1
data_out_eval = data_out_eval[['index', 'text']]
data_out_test['index'] = -1
data_out_test = data_out_test[['index', 'text']]

args.num_labels = train_data['index'].nunique()
# 检查数据格式
print(train_data.head())

from datasets import Dataset
from transformers import AutoTokenizer

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# 将Pandas DataFrame转换为 Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data)
dataset_in_test = Dataset.from_pandas(data_in_test)
dataset_in_eval = Dataset.from_pandas(data_in_eval)
dataset_out_test = Dataset.from_pandas(data_out_test)
dataset_out_eval = Dataset.from_pandas(data_out_eval)

# 定义tokenizer函数
def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True, max_length=512)

# 对数据集进行tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
dataset_in_test = dataset_in_test.map(tokenize_function, batched=True)
dataset_in_eval = dataset_in_eval.map(tokenize_function, batched=True)
dataset_out_test = dataset_out_test.map(tokenize_function, batched=True)
dataset_out_eval = dataset_out_eval.map(tokenize_function, batched=True)

# 删除无关列
train_dataset = train_dataset.remove_columns(["text"])
dataset_in_test = dataset_in_test.remove_columns(["text"])
dataset_in_eval = dataset_in_eval.remove_columns(["text"])
dataset_out_test = dataset_out_test.remove_columns(["text"])
dataset_out_eval = dataset_out_eval.remove_columns(["text"])

# 将label列重命名为 'labels'
train_dataset = train_dataset.rename_column("index", "labels")
dataset_in_test = dataset_in_test.rename_column("index", "labels")
dataset_in_eval = dataset_in_eval.rename_column("index", "labels")
dataset_out_test = dataset_out_test.rename_column("index", "labels")
dataset_out_eval = dataset_out_eval.rename_column("index", "labels")
eval_dataset = concatenate_datasets([dataset_in_eval, dataset_out_eval])
test_dataset = concatenate_datasets([dataset_in_test, dataset_out_test])

# 设置数据格式为 PyTorch tensors
train_dataset.set_format("torch")
dataset_in_test.set_format("torch")
dataset_in_eval.set_format("torch")
dataset_out_test.set_format("torch")
eval_dataset.set_format("torch")
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
eval_loader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.eval_batch_size, collate_fn=collate_batch)
