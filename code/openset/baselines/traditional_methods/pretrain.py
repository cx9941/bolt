import os
os.environ["WANDB_DISABLED"]="true"
from configs import args
from utils import get_best_checkpoint
if os.path.exists(args.checkpoint_path) and get_best_checkpoint(args.checkpoint_path) is not None:
    exit()
import torch
import pandas as pd
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification
from load_dataset import train_dataset, dataset_in_test, collate_batch, dataset_in_eval
import logging
from sklearn.metrics import classification_report
import numpy as np

def compute_metrics(eval_predictions):
    preds, golds = eval_predictions
    preds = np.argmax(preds, axis=1)
    metrics = classification_report(preds, golds, output_dict=True)
    metrics['macro avg'].update({'accuracy': metrics['accuracy']})
    return metrics['macro avg']

# 加载预训练的BERT模型，并指定分类任务的输出层数量
model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels, device_map='cuda')

from transformers import Trainer, TrainingArguments

# 设置训练参数
training_args = TrainingArguments(
    output_dir=args.checkpoint_path,          # 模型输出目录
    evaluation_strategy="epoch",     # 每个epoch结束后进行评估
    logging_strategy="epoch",     # 每个epoch结束后进行评估
    per_device_train_batch_size=16,   # 每个设备上的批大小
    per_device_eval_batch_size=32,    # 测试时的批大小
    num_train_epochs=args.n_epochs,              # 训练周期
    weight_decay=0.01,               # 权重衰减
    save_strategy='epoch',
    save_total_limit=1,
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dataset_in_eval,
    compute_metrics=compute_metrics,
    data_collator=collate_batch
)
trainer.train()
results = trainer.evaluate()
logging.info(results)