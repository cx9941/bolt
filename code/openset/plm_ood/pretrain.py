# pretrain.py (最终修正版)

import os
os.environ["WANDB_DISABLED"]="true"

import torch
import pandas as pd
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
import logging
from sklearn.metrics import classification_report
import numpy as np

from utils import get_best_checkpoint, create_model
# --- 改动 1: 导入新的数据加载函数 ---
from load_dataset import load_and_prepare_datasets
from reg_trainer import RegTrainer
from configs import get_plm_ood_config

def run_pretraining(args):
    """
    预训练的主函数，包含了所有的核心逻辑。
    """
    # --- 改动 2: 在函数开头调用函数，获取所有数据对象 ---
    logging.info("Loading and preparing datasets for pretraining...")
    data = load_and_prepare_datasets(args)
    # 从字典中解包出我们需要的变量
    tokenizer = data['tokenizer']
    train_dataset = data['train_dataset']
    dataset_in_eval = data['dataset_in_eval']
    collate_batch = data['collate_batch'] # 注意：collate_fn 也从这里获取
    logging.info("Datasets loaded successfully.")
    # --- 改动结束 ---
    
    # 检查点存在则退出
    if os.path.exists(args.checkpoint_path) and get_best_checkpoint(args.checkpoint_path) is not None:
        logging.warning(f"Checkpoint already exists at {args.checkpoint_path}. Exiting pretraining.")
        return

    def compute_metrics(eval_predictions):
        preds, golds = eval_predictions
        preds = np.argmax(preds, axis=1)
        metrics = classification_report(preds, golds, output_dict=True)
        metrics['macro avg'].update({'accuracy': metrics['accuracy']})
        return metrics['macro avg']

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"] if 'bert' in args.model_path else ["q_proj", "v_proj"]
    )
    data = load_and_prepare_datasets(args)
    tokenizer = data['tokenizer']
    base_model = create_model(model_path=args.model_path, num_labels=args.num_labels, tokenizer=tokenizer)
    model = get_peft_model(base_model, peft_config)
    
    # --- 改动 3: 使用从 'data' 字典中获取的对象 ---
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.checkpoint_path,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.n_epochs,
        weight_decay=0.01,
        save_strategy='epoch',
        save_total_limit=1,
    )

    # 定义Trainer
    trainer = RegTrainer(
        reg_loss=args.reg_loss,
        num_labels=args.num_labels,
        device=args.device,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dataset_in_eval,
        compute_metrics=compute_metrics,
        data_collator=collate_batch
    )
    # --- 改动结束 ---

    logging.info("Evaluating before training...")
    results = trainer.evaluate()
    logging.info(f"Initial results: {results}")

    logging.info("Starting training...")
    trainer.train()

    logging.info("Evaluating after training...")
    results = trainer.evaluate()
    logging.info(f"Final results: {results}")

if __name__ == '__main__':
    config_args = get_plm_ood_config()
    run_pretraining(config_args)