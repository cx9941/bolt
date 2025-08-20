# baselines/analogy/utils.py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained('../../llms/bert-base-uncased')
model = AutoModel.from_pretrained('../../llms/bert-base-uncased')
model.eval()

def get_label_pool(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_data(path):
    return pd.read_csv(path, sep='\t')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)  # CLS

def get_similar_examples(query, df, top_k=3):
    query_vec = get_bert_embedding(query)
    sims = []
    for _, row in df.iterrows():
        cand_vec = get_bert_embedding(row['text'])
        sim = F.cosine_similarity(query_vec, cand_vec, dim=0).item()
        sims.append((sim, row['text'], row['label']))
    sims.sort(reverse=True)
    return sims[:top_k]