# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script_v0 on your own text classification task. Pointers for this are left as comments.
import json
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import dataclasses
import inspect
import logging
import random
import sys

# import fitlog
from dataclasses import dataclass, field
from sklearn.metrics import matthews_corrcoef
from typing import Optional

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PretrainedConfig,
    EvalPrediction,
    set_seed,
)
#from trainer_knn import SimpleTrainer
from trainer import SimpleTrainer
from evaluate import Evaluation

from training_args import TrainingArguments
#from model import (
#   ContrastiveOrigin,
#    ContrastiveMoCoKnnBert
#)
from model import (
    ContrastiveOrigin,
    ContrastiveMoCoKnnBert
)
from transformers.trainer_utils import is_main_process

from transformers import AutoModelForSequenceClassification
import torch

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    valid_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None,
    )

    # def __post_init__(self):
    #     if self.task_name is not None:
    #         self.task_name = self.task_name.lower()
    #         if self.task_name not in task_to_keys.keys():
    #             raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
    #     elif self.train_file is None or self.valid_file is None:
    #         raise ValueError("Need either a GLUE task or a training/validation file.")
    #     else:
    #         extension = self.train_file.split(".")[-1]
    #         assert extension in ["csv", "json", "tsv"], "`train_file` should be a csv or a json file."
    #         extension = self.valid_file.split(".")[-1]
    #         assert extension in ["csv", "json", "tsv"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    train_pattern: str = field(default="further_pretrain")

# @dataclass
# class FitLogArguments:
#     #task: str = field(default='mrpc')
#     negative_num: int = field(default=96)
#     positive_num: int = field(default=3)
#     queue_size: int = field(default=32000)
#     top_k: int = field(default=20)
#     end_k: int = field(default=1)
#     m: float = field(default=0.999)
#     contrastive_rate_in_training: float = field(default=0.1)
#     contrastive_rate_in_inference: float = field(default=0.1)


def data_collator(features):
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    first = features[0]
    batch = {}
    if "original_text" in first:
        batch["original_text"] = [f["original_text"] for f in features]
    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script_v0.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))


    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, fitlog_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    

    # model_args, data_args, training_args, fitlog_args = parser.parse_json_file(json_file=os.path.abspath('json/snips/0.25/0.json'))

    # results_path = './model_output/' + '_'.join([training_args.data, str(training_args.known_ratio), str(training_args.seed)]) + '.csv'
    # if os.path.exists(results_path):
    #     exit()

    # model_args, data_args, training_args, fitlog_args = parser.parse_json_file(json_file='json/demo.json')

    # ========================================================================
    # --- 全新的、符合SOP标准的数据加载与处理逻辑 (已更新为新参数名) ---
    # ========================================================================

    # 1. 根据SOP标准从YAML参数构建文件路径
    origin_train_path = os.path.join(training_args.data_dir, training_args.dataset, 'origin_data', 'train.tsv')
    origin_valid_path = os.path.join(training_args.data_dir, training_args.dataset, 'origin_data', 'dev.tsv')
    origin_test_path = os.path.join(training_args.data_dir, training_args.dataset, 'origin_data', 'test.tsv')
    
    labeled_train_path = os.path.join(training_args.data_dir, training_args.dataset, 'labeled_data', str(training_args.labeled_ratio), 'train.tsv')
    labeled_valid_path = os.path.join(training_args.data_dir, training_args.dataset, 'labeled_data', str(training_args.labeled_ratio), 'dev.tsv')

    # 2. 使用pandas加载标准化的TSV文件
    origin_train_df = pd.read_csv(origin_train_path, sep='\t', dtype=str)
    origin_valid_df = pd.read_csv(origin_valid_path, sep='\t', dtype=str)
    df_test = pd.read_csv(origin_test_path, sep='\t', dtype=str)

    labeled_train_df = pd.read_csv(labeled_train_path, sep='\t', dtype=str)
    labeled_valid_df = pd.read_csv(labeled_valid_path, sep='\t', dtype=str)
    
    # 3. 组合文本和标签信息
    df_train = labeled_train_df
    df_train['text'] = origin_train_df['text']
    
    df_valid = labeled_valid_df
    df_valid['text'] = origin_valid_df['text']

    # 4. (核心修改) 从标准化的.list文件加载已知类
    known_label_path = os.path.join(
        training_args.data_dir,
        training_args.dataset,
        'label',
        f'fold{training_args.fold_num}',
        f'part{training_args.fold_idx}',
        f'label_known_{training_args.known_cls_ratio}.list'
    )
    seen_labels = pd.read_csv(known_label_path, header=None)[0].tolist()

    # 5. (保留原逻辑) 使用新的 seen_labels 列表来筛选和划分数据
    # 训练集和验证集：必须是已知类别 & 已标注
    df_train_seen = df_train[(df_train.label.isin(seen_labels)) & (df_train['labeled'].astype(bool))]
    df_valid_seen = df_valid[(df_valid.label.isin(seen_labels)) & (df_valid['labeled'].astype(bool))]

    df_valid_oos = df_valid[~df_valid.label.isin(seen_labels)]
    df_valid_oos.loc[:, "label"] = 'oos'
    df_test.loc[~df_test.label.isin(seen_labels), "label"] = 'oos'

    # 6. (保留原逻辑) 将处理好的DataFrame转换为Hugging Face的Dataset对象
    data = dict()
    data["train"] = Dataset.from_pandas(df_train_seen, preserve_index=False)
    data["valid_seen"] = Dataset.from_pandas(df_valid_seen, preserve_index=False)
    data["valid_oos"] = Dataset.from_pandas(df_valid_oos, preserve_index=False)
    data["test"] = Dataset.from_pandas(df_test, preserve_index=False)
    datasets = DatasetDict(data)

    # ========================================================================
    # --- 数据处理逻辑改造完成 ---
    # ========================================================================

    # Labels
    if data_args.task_name is not None: # False
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            features = datasets["train"].features["label"]
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list += ['oos']
            num_labels = len(label_list)

    if training_args.load_trained_model: # False
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )

        config.model_name = model_args.model_name_or_path
        config.negative_num = training_args.negative_num
        config.positive_num = training_args.positive_num
        config.queue_size = training_args.queue_size
        config.train_multi_head = training_args.train_multi_head
        config.contrastive_rate_in_training = training_args.contrastive_rate_in_training
        config.load_trained_model = training_args.load_trained_model
        config.multi_head_num = training_args.multi_head_num
        config.num_labels = num_labels
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path
        )
        if training_args.load_model_pattern == "original_model":
            model = AutoModelForSequenceClassification.from_pretrained(
                training_args.model_path,
                config=config
            )
        elif training_args.load_model_pattern == "knn_bert":
            config.knn_num = training_args.top_k
            model = ContrastiveMoCoKnnBert(config=config)
            logger.info("loading model form " + training_args.model_path + "pytorch_model.bin")
            state_dict = torch.load(training_args.model_path + "pytorch_model.bin")
            model.load_state_dict(state_dict)
        else:
            logger.warning("your model should in list [original_model moco roberta_moco knn_bert knn_roberta]")
    else: # True
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )

        config.model_name = model_args.model_name_or_path
        config.negative_num = training_args.negative_num
        config.positive_num = training_args.positive_num
        # config.m = fitlog_args.m
        config.m = training_args.m
        config.queue_size = training_args.queue_size
        config.train_multi_head = training_args.train_multi_head
        config.contrastive_rate_in_training = training_args.contrastive_rate_in_training
        config.load_trained_model = training_args.load_trained_model
        config.multi_head_num = training_args.multi_head_num
        config.lmcl = training_args.lmcl
        config.cl_mode = training_args.cl_mode
        config.rnn_number_layers = training_args.rnn_number_layers
        config.sup_cont = training_args.sup_cont
        config.num_labels = num_labels
        config.norm_coef = training_args.norm_coef
        config.hidden_dim = training_args.hidden_dim
        config.device = training_args.device
        config.T = training_args.temperature
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
        # This model is the original contrastive learning model
        if training_args.load_model_pattern == "original_model":
            model = ContrastiveOrigin(config=config)
        # This model is the KNN model
        elif training_args.load_model_pattern == "knn_bert":
            config.knn_num = training_args.top_k
            config.end_k = training_args.end_k
            model = ContrastiveMoCoKnnBert(config=config)
        else:
            logger.warning("your model should in list [original_model moco roberta_moco knn_bert knn_roberta scl_model]")

    # Preprocessing the datasets
    # --- 修正：绕过脆弱的自动列名检测，直接指定文本列 ---
    # 我们的数据集是单句分类，文本列永远是 'text'
    sentence1_key, sentence2_key = "text", None
    logger.info(f"Set sentence key to: {sentence1_key}")
    # --- 修正结束 ---

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        result["sent_id"] = [index for index, i in enumerate(examples["label"])]
        result["original_text"] = examples[sentence1_key]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]
    eval_dataset = datasets["valid_seen"]
    test_dataset = datasets["test"] if training_args.do_predict else None
    eval_oos_dataset = datasets["valid_oos"]
    label_eval_oos_dataset = eval_oos_dataset['label']

    # Initialize our Trainer
    training_args.num_labels = config.num_labels
    trainer = SimpleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        test_dataset=test_dataset if training_args.do_predict else None,
        compute_metrics=None,
        tokenizer=tokenizer,
        number_labels=num_labels,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=data_collator
    )

    # Training
    if training_args.do_train:
        if training_args.load_model_pattern == 'knn_bert':
            trainer.train_mocoknn(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
        else:
            trainer.train_origin(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
        #trainer.save_model()  # Saves the tokenizer too for easy upload
        #fitlog.finish()

    # Evalution
    if training_args.do_predict:
        evaler = Evaluation(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            eval_oos_dataset=eval_oos_dataset,
            tokenizer=tokenizer,
            number_labels=num_labels,
            data_collator=data_collator
        )
        evaler.evaluation(model_path=model_args.model_name_or_path)

    return None


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
