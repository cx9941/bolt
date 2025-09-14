import os
os.environ["WANDB_DISABLED"]="true"
from configs import args
from utils import get_best_checkpoint
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification
from load_dataset import train_dataset, dataset_in_test, collate_batch, dataset_in_eval, test_loader
import logging
from sklearn.metrics import classification_report
import numpy as np
import os
os.environ["WANDB_DISABLED"]="true"
from configs import args
from utils import get_best_checkpoint, create_model
if os.path.exists(args.checkpoint_path) and get_best_checkpoint(args.checkpoint_path) is not None:
    exit()
import torch
import pandas as pd
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification
from load_dataset import train_dataset, dataset_in_test, collate_batch, dataset_in_eval, tokenizer
import logging
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig, Trainer

def compute_metrics(eval_predictions):
    preds, golds = eval_predictions
    preds = np.argmax(preds, axis=1)
    # metrics = classification_report(preds, golds, output_dict=True)
    metrics = classification_report(golds, preds, output_dict=True)
    metrics['macro avg'].update({'accuracy': metrics['accuracy']})
    return metrics['macro avg']

peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,         # 序列分类任务
        inference_mode=False,               # 训练模式
        r=8,                                 # LoRA 低秩维度
        lora_alpha=32,                       # LoRA scaling 参数
        lora_dropout=0.1,                    # LoRA dropout
        target_modules=["query", "key", "value"] if 'bert' in args.model_path else ["q_proj", "v_proj"]
    )

base_model = create_model(model_path=args.model_path, num_labels=args.num_labels)

model = get_peft_model(base_model, peft_config)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

from transformers import TrainingArguments, EarlyStoppingCallback

# 设置训练参数
training_args = TrainingArguments(
    output_dir=args.checkpoint_path,          # 模型输出目录
    eval_strategy="epoch",     # 每个epoch结束后进行评估
    # eval_steps=50,     # 每个epoch结束后进行评估
    logging_strategy="steps",     # 每个epoch结束后进行评估
    logging_steps=20,     # 每个epoch结束后进行评估
    per_device_train_batch_size=args.train_batch_size,   # 每个设备上的批大小
    per_device_eval_batch_size=args.eval_batch_size,    # 测试时的批大小
    num_train_epochs=args.n_epochs,              # 训练周期
    weight_decay=0.01,               # 权重衰减
    save_strategy='epoch',
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model=args.metric_for_best,  # 例如 "accuracy"
    greater_is_better=True,             # 用 accuracy 时应为 True
)
# 设置训练参数
training_args = TrainingArguments(
    output_dir=args.checkpoint_path,          # 模型输出目录
    eval_strategy="epoch",     # 每个epoch结束后进行评估
    logging_strategy="epoch",     # 每个epoch结束后进行评估
    per_device_train_batch_size=16,   # 每个设备上的批大小
    per_device_eval_batch_size=32,    # 测试时的批大小
    num_train_epochs=args.n_epochs,              # 训练周期
    weight_decay=0.01,               # 权重衰减
    save_strategy='epoch',
    save_total_limit=1,
)

callbacks = [EarlyStoppingCallback(
    early_stopping_patience=args.es_patience,
    early_stopping_threshold=args.es_min_delta
)]

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dataset_in_eval,
    compute_metrics=compute_metrics,
    data_collator=collate_batch,
    callbacks=callbacks
)
results = trainer.evaluate()
logging.info("initial\n", results)
trainer.train()
results = trainer.evaluate()
logging.info("final\n", results)