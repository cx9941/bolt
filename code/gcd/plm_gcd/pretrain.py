# pretrain.py  —— 清理后的稳定版
import os
from configs import args
from utils import get_best_checkpoint, create_model
import torch
import logging
import numpy as np
import pandas as pd

from load_dataset import (
    train_dataset,
    dataset_in_eval,
    collate_batch,
    tokenizer,
)

from sklearn.metrics import classification_report
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

# 若已有checkpoint则直接退出（避免重复训练/覆盖）
if os.path.exists(args.checkpoint_path) and get_best_checkpoint(args.checkpoint_path) is not None:
    exit()

def compute_metrics(eval_predictions):
    preds, golds = eval_predictions
    preds = np.argmax(preds, axis=1)
    metrics = classification_report(golds, preds, output_dict=True)
    # 保证包含 'accuracy'，供 metric_for_best_model 使用
    metrics['macro avg'].update({'accuracy': metrics['accuracy']})
    return metrics['macro avg']

# LoRA 配置
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"] if 'bert' in args.model_path else ["q_proj", "v_proj"],
)

# 基座模型（已含4bit量化等设置）
base_model = create_model(model_path=args.model_path, num_labels=args.num_labels)
model = get_peft_model(base_model, peft_config)

# tokenizer 对齐（避免特殊符号缺失）
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# —— 关键：只保留一份 TrainingArguments，参数名写对 —— #
training_args = TrainingArguments(
    output_dir=args.checkpoint_path,
    eval_strategy="epoch",                 
    logging_strategy="epoch",
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.num_pretrain_epochs,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=1,

    # 最佳模型 + 监控指标（compute_metrics 返回了 'accuracy'）
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    # 关闭外部上报，消除 WANDB_DISABLED 弃用提示
    report_to="none",
)

callbacks = [
    EarlyStoppingCallback(
        early_stopping_patience=getattr(args, "es_patience", 3),
        early_stopping_threshold=getattr(args, "es_min_delta", 0.0),
    )
]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dataset_in_eval,
    compute_metrics=compute_metrics,
    data_collator=collate_batch,
    tokenizer=tokenizer,   # v5 会提示用 processing_class，可暂时忽略
    callbacks=callbacks,
)

# 评估-训练-再评估
results = trainer.evaluate()
logging.info("initial %s", results)
trainer.train()
results = trainer.evaluate()
logging.info("final %s", results)
