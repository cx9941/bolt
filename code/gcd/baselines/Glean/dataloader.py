from utils.tools import *

# max_seq_lengths = {'clinc':30, 'stackoverflow':45, 'banking':55}
# TOPK = {'clinc':50, 'stackoverflow':500, 'banking':50}
# task = {'clinc': 'intent', 'stackoverflow': 'intent', 'banking': 'intent'}

class Data:
    
    def __init__(self, args):
        set_seed(args.seed)
        processor = DatasetProcessor()
        args.cluster_num_factor = 1
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        all_label_path = os.path.join(self.data_dir, 'label', 'label.list')
        self.all_label_list = pd.read_csv(all_label_path, header=None)[0].tolist()
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        
        self.known_label_list = pd.read_csv(f'{args.data_dir}/{args.dataset}/label/fold{args.fold_num}/part{args.fold_idx}/label_known_{args.known_cls_ratio}.list', header=None)[0].tolist()

        # self.known_lab = [int(np.where(self.all_label_list== a)[0]) for a in self.known_label_list]
        self.known_lab = [self.all_label_list.index(a) for a in self.known_label_list]

        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)
     
        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(processor, args, 'train')
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')
        print('dataset', args.dataset)
        print('known_cls_ratio', args.known_cls_ratio)
        print('label_setting', args.label_setting)
        print('labeled_shot', args.labeled_shot)
        print('labeled_ratio', args.labeled_ratio)
        print('num_known_cls', self.n_known_cls)
        print('num_labeled_samples', len(self.train_labeled_examples))
        print('num_unlabeled_samples', len(self.train_unlabeled_examples))
        args.num_labeled_examples = len(self.train_labeled_examples)

        if self.n_known_cls > 0:
            self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples, args, 'train')
        else:
            self.train_labeled_dataloader = None

        self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids = self.get_semi(self.train_labeled_examples, self.train_unlabeled_examples, args)
        self.train_semi_dataset, self.train_semi_dataloader = self.get_semi_loader(self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids, args)

        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')
        print('\nlabel_map_train\n', args.label_map_train)
        # print('\nlabel_map_test\n', args.label_map_test)
        print('\nlabel_map_semi\n', args.label_map_semi)

        ## Construct Demonstration Data
        if args.flag_demo or args.flag_demo_c:
            print('\nConstruct Demonstration Data')
            # read from dev set
            df_demo_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'dev.tsv')
            df_demo = pd.read_csv(df_demo_path, sep='\t')
            
            known_classes = self.known_label_list
            print('Num of known classes: ', len(known_classes))
            print('Known classes: ', known_classes)

            # sample demon data per known class
            demo_data = df_demo[df_demo['label'].isin(known_classes)].groupby('label').head(args.known_demo_num_per_class).reset_index(drop=True)
            demo_data_c = df_demo[~df_demo['label'].isin(known_classes)].groupby('label').head(args.known_demo_num_per_class_c).reset_index(drop=True)
            print('Demo data shape: ', demo_data.shape)
            print('Demo data shape_c: ', demo_data_c.shape)

            # Construct demonstration prompt
            args.prompt_demo = ""
            args.prompt_demo_c = ""
            # Add demonstration in the format of Text: [text], Label: [label]
            if args.flag_demo:
                args.prompt_demo += "\n\nTask Context: \n"
                for i in range(len(demo_data)):
                    args.prompt_demo += f"Text: {demo_data['text'][i]}\t Label: {demo_data['label'][i]}\n"
            if args.flag_demo_c:
                args.prompt_demo_c += "\n\nTask Context: \n"
                for i in range(len(demo_data_c)):
                    if args.dataset != 'stackexchange':
                        args.prompt_demo_c += f"Text: {demo_data_c['text'][i]}\t Label: {demo_data_c['label'][i]}\n"
                    else:
                        args.prompt_demo_c += f"Text: {demo_data_c['text'][i]}\t \n"

            print(args.prompt_demo_c)


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
                train_labeled_examples.append(InputExample(guid=guid, text_a=row['text'], label=row['label']))
            
            train_unlabeled_examples = []
            for i, row in unlabeled_df.iterrows():
                guid = f"train_unlabeled-{i}"
                train_unlabeled_examples.append(InputExample(guid=guid, text_a=row['text'], label=row['label']))
                
            return train_labeled_examples, train_unlabeled_examples

        # --- 对于验证集和测试集，使用旧的、但正确的加载逻辑 ---
        # （因为它们不需要复杂的分割，只需读取 origin_data 即可）
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
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if self.n_known_cls > 0:
            labeled_features = convert_examples_to_features(args, labeled_examples, self.known_label_list, args.max_seq_length, tokenizer)
            labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
            labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
            labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
            labeled_label_ids = torch.tensor([f.label_id for f in labeled_features], dtype=torch.long)  
        else:
            labeled_features = None

        unlabeled_features = convert_examples_to_features(args, unlabeled_examples, self.all_label_list, args.max_seq_length, tokenizer, mode='semi')
        unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
        unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_label_ids = torch.tensor([f.label_id for f in unlabeled_features], dtype=torch.long)     

        if self.n_known_cls > 0:
            semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
            semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
            semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
            semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
        else:
            semi_input_ids = unlabeled_input_ids
            semi_input_mask = unlabeled_input_mask
            semi_segment_ids = unlabeled_segment_ids
            semi_label_ids = unlabeled_label_ids
        return semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids

    def get_semi_loader(self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args):
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids)
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size = args.train_batch_size) 

        return semi_data, semi_dataloader

    def get_loader(self, examples, args, mode = 'train'):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        
        if mode == 'train' or mode == 'eval':
            features = convert_examples_to_features(args, examples, self.known_label_list, args.max_seq_length, tokenizer, mode)
        elif mode == 'test':
            features = convert_examples_to_features(args, examples, self.all_label_list, args.max_seq_length, tokenizer, mode)
        else:
            raise NotImplementedError(f"Mode {mode} not found")

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
        
        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = args.pretrain_batch_size)    
        elif mode in ["eval", "test"]:
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = args.eval_batch_size)
        else:
            raise NotImplementedError(f"Mode {mode} not found")
        
        return dataloader


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class DatasetProcessor(DataProcessor):

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
        docs = os.listdir(data_dir)
        if "train.tsv" in docs:
            import pandas as pd
            test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
            labels = [str(label).lower() for label in test['label']]
            labels = np.unique(np.array(labels))
        elif "dataset.json" in docs:
            with open(os.path.join(data_dir, "dataset.json"), 'r') as f:
                dataset = json.load(f)
                dataset = dataset[list(dataset.keys())[0]]
            labels = []
            for dom in dataset:
                for ind, data in enumerate(dataset[dom]):
                    label = data[1][0]
                    labels.append(str(label).lower())
            labels = np.unique(np.array(labels))
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
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(args, examples, label_list, max_seq_length, tokenizer, mode=None):      
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
    if mode == 'train':
        args.label_map_train = label_map
        args.get_label_name_train = {v: k for k, v in label_map.items()}
    elif mode == 'eval':
        args.label_map_eval = label_map
        args.get_label_name_eval = {v: k for k, v in label_map.items()}
    elif mode == 'test':
        args.label_map_test = label_map
        args.get_label_name_test = {v: k for k, v in label_map.items()}
    elif mode == 'semi':
        args.label_map_semi = label_map
        args.get_label_name_semi = {v: k for k, v in label_map.items()}

    features = []
    for (index, example) in enumerate(examples):
        tokens = tokenizer(example.text_a, padding='max_length', max_length=max_seq_length, truncation=True)

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


