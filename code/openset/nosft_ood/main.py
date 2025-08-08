import os
import pandas as pd
from init_args import parse_args
from utils import load_data, get_label_pool
from query_llm import LLMQuerier
from tqdm import tqdm
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
lock = threading.Lock()

def get_similar_examples(idx, train_data, labels, pred_label, related_items, topk=20):
    train_data = train_data.reset_index(drop=True).reset_index()
    label_idx = labels.index(pred_label)
    related_idx = related_items[idx][label_idx]
    sub_df = train_data[(train_data['index'].isin(related_idx))]['text'].tolist()
    label_df= train_data[(train_data['index'].isin(related_idx))]['label'].tolist()
    return sub_df[:topk], label_df[:topk]

def process_instance(idx, row, train_data, labels, related_items, querier, prompt_method, output_path, origin_pred=None, max_tries=0):
    if max_tries > 3:
        return
    text = row['text']
    gt_label = row['label']
    
    try:
        if prompt_method == 'cot':
            response = querier.query_cot_prompt(text, labels, similar_exs, similar_labels)
        elif prompt_method == 'simple':
            response = querier.query_simple_prompt(text, labels)
        elif prompt_method == 'analogy':
            if origin_pred is None:
                response = querier.query_simple_prompt(text, labels)
            else:
                response = origin_pred['pred_label'][idx]
            
            response = re.findall(r"|".join(labels), response.split('\n')[-1])[0] if len(re.findall(r"|".join(labels),  response.split('\n')[-1])) > 0 else 'It is 000.OOD'
            if response in labels:
                similar_exs, similar_labels = get_similar_examples(idx, train_data, labels, response, related_items)
                response = querier.query_analogy_prompt(text, labels, response, similar_exs, similar_labels)
            else:
                response = 'It is 000.OOD'
        else:
            assert False, f"{prompt_method} not correct"
    except Exception as e:
        print(f"[Retry] Error at index {idx}: {e}")
        time.sleep(20)
        process_instance(idx, row, train_data, labels, related_items, querier, prompt_method, output_path, origin_pred,max_tries=max_tries+1)

    # 多线程写入需加锁
    with lock:
        new_row = pd.DataFrame([[text, gt_label, response]], columns=['text', 'gt_label', 'pred_label'])
        new_row.to_csv(output_path, mode='a', sep='\t', header=False, index=False)

def main():
    args = parse_args()
    
    label_file = os.path.join(args.data_dir, args.dataset, f"{args.dataset}_{args.mask_ratio}.txt")
    test_file = os.path.join(args.data_dir, args.dataset, "origin_data", "test.tsv")
    train_file = os.path.join(args.data_dir, args.dataset, "origin_data", "train.tsv")

    labels = get_label_pool(label_file)
    test_data = load_data(test_file)
    train_data = load_data(train_file)
    with open(label_file, 'r') as f:
        valid_labels = [line.strip() for line in f if line.strip()]
    train_data = train_data[train_data['label'].isin(valid_labels)]
    train_data['label_idx']= train_data['label'].apply(lambda x: valid_labels.index(x))
    # train_data = train_data.reset_index(drop=True)

    origin_pred = None
    related_items = None
    if args.prompt_method == 'analogy':
        if os.path.exists(args.output_path.replace('analogy', 'simple')):
            origin_pred = pd.read_csv(args.output_path.replace('analogy', 'simple'), sep='\t')
            origin_pred = pd.merge(test_data, origin_pred[['text', 'pred_label']], on=['text'])
        related_items = np.load(f"outputs/sim_matrix/{args.dataset}_{args.mask_ratio}.npy")
        

    

    if os.path.exists(args.output_path):
        output_df = pd.read_csv(args.output_path, sep='\t')
        finished_indices = set(output_df.index)
    else:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        pd.DataFrame(columns=['text', 'gt_label', 'pred_label']).to_csv(args.output_path, sep='\t', index=False)
        finished_indices = set()

    querier = LLMQuerier(args.llm_url, args.llm_name)

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for idx, row in test_data.iterrows():
            if idx in finished_indices:
                continue
            futures.append(executor.submit(process_instance, idx, row, train_data, labels, related_items, querier, args.prompt_method, args.output_path, origin_pred))
        
        # 进度条跟踪
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

if __name__ == '__main__':
    main()