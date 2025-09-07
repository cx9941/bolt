import torch
from transformers import AutoTokenizer
import random
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import csv


class Data:

    def __init__(self, args):
        self.set_seed(args.seed)
        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        all_label_path = os.path.join(self.data_dir, 'label', 'label.list')
        self.all_label_list = pd.read_csv(all_label_path, header=None)[0].tolist()

        self.num_known = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = pd.read_csv(f'{args.data_dir}/{args.dataset}/label/fold{args.fold_num}/part{args.fold_idx}/label_known_{args.known_cls_ratio}.list', header=None)[0].tolist()
        
        self.known_lab = [self.all_label_list.index(a) for a in self.known_label_list]
        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)
        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(processor, args, 'train')
        print('num_labeled_samples', len(self.train_labeled_examples))
        print('num_unlabeled_samples', len(self.train_unlabeled_examples))
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')
        
        self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids = self.get_semi(self.train_labeled_examples, self.train_unlabeled_examples, args)

        self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples, args, 'train')
        self.pretrain_labeled_dataloader = self.get_loader(self.train_labeled_examples, args, 'pretrain')
        self.train_semi_dataloader = self.get_semi_loader(self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids, args)
        self.pretrain_semi_dataloader = self.get_semi_loader(self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids, args, 'pretrain')
        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')
    
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def get_examples(self, processor, args, mode='train'):
    # --- 对于训练集，执行严谨的双重过滤 ---
        if mode == 'train':
            # 1. 定义并读取两个核心文件
            origin_data_path = os.path.join(self.data_dir, 'origin_data', 'train.tsv')
            labeled_info_path = os.path.join(self.data_dir, 'labeled_data', str(args.labeled_ratio), 'train.tsv')

            origin_data = pd.read_csv(origin_data_path, sep='\t')
            labeled_info = pd.read_csv(labeled_info_path, sep='\t')
            
            # 2. 合并信息：将原始文本添加到标签信息中
            # 假设两个文件行数和顺序完全对应
            merged_data = labeled_info
            merged_data['text'] = origin_data['text']
            
            # 3. 创建双重过滤的布尔掩码 (boolean mask)
            # 条件：标签必须在已知类别列表里 & labeled 字段必须为 True
            is_labeled_known = (merged_data['label'].isin(self.known_label_list)) & (merged_data['labeled'])
            
            # 4. 根据掩码分割为“有标签”和“无标签”两个 DataFrame
            labeled_df = merged_data[is_labeled_known]
            unlabeled_df = merged_data[~is_labeled_known] # ~is_labeled_known 表示取反

            # 5. 将两个 DataFrame 分别转换为 InputExample 对象列表
            train_labeled_examples = []
            for i, row in labeled_df.iterrows():
                guid = f"train_labeled-{i}"
                train_labeled_examples.append(InputExample(guid=guid, text=row['text'], label=row['label']))
            
            train_unlabeled_examples = []
            for i, row in unlabeled_df.iterrows():
                guid = f"train_unlabeled-{i}"
                train_unlabeled_examples.append(InputExample(guid=guid, text=row['text'], label=row['label']))
                
            return train_labeled_examples, train_unlabeled_examples

        else: # mode is 'eval' or 'test'
            ori_examples = processor.get_examples(self.data_dir, mode)
            
            if mode == 'eval':
                # 验证集：只包含已知类别的样本
                eval_examples = []
                for example in ori_examples:
                    if example.label in self.known_label_list:
                        eval_examples.append(example)
                return eval_examples

            elif mode == 'test':
                # 测试集：包含所有样本
                return ori_examples

    def get_semi(self, labeled_examples, unlabeled_examples, args):
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=True)    
        labeled_features = self.convert_examples_to_features(labeled_examples, self.known_label_list, args.max_seq_length, tokenizer)
        unlabeled_features = self.convert_examples_to_features(unlabeled_examples, self.all_label_list, args.max_seq_length, tokenizer)

        labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
        labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
        labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
        labeled_label_ids = torch.tensor([f.label_id for f in labeled_features], dtype=torch.long)      

        unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
        unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_label_ids = torch.tensor([f.label_id for f in unlabeled_features], dtype=torch.long)      

        semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
        semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
        semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
        semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
        return semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids

    def get_semi_loader(self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args, mode='train'):
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids)
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size = args.train_batch_size) 
        if mode == 'pretrain':
            semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size = args.pretrain_batch_size)
        return semi_dataloader

    def get_loader(self, examples, args, mode = 'train'):
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=True)    
        
        if mode == 'train' or mode == 'eval' or mode == 'pretrain':
            features = self.convert_examples_to_features(examples, self.known_label_list, args.max_seq_length, tokenizer)
        elif mode == 'test':
            features = self.convert_examples_to_features(examples, self.all_label_list, args.max_seq_length, tokenizer)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
        
        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = args.train_batch_size)    
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = args.eval_batch_size) 
        elif mode == 'pretrain':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = args.pretrain_batch_size)
        return dataloader

    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):      
        label_map = {}
        for i, label in enumerate(label_list):
            label_map[label] = i

        features = []
        for (index, example) in enumerate(examples):
            tokens = tokenizer(example.text, padding='max_length', max_length=max_seq_length, truncation=True)

            input_ids = tokens['input_ids']
            input_mask = tokens['attention_mask']
            segment_ids = tokens['token_type_ids']
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label_map[example.label]
            features.append(
                InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=label_id))
        return features


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DatasetProcessor():
    
    def get_examples(self, data_dir, mode):
        if mode == 'train':
            file_path = os.path.join(data_dir, "origin_data", "train.tsv")
        elif mode == 'eval':
            file_path = os.path.join(data_dir, "origin_data", "dev.tsv")
        elif mode == 'test':
            file_path = os.path.join(data_dir, "origin_data", "test.tsv")
        else:
            raise ValueError("Invalid mode %s" % mode)

        return self._create_examples(self._read_tsv(file_path), mode)

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label'], dtype=str))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines