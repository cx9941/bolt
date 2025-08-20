from util import *
import torch
from torch.utils.data import DataLoader, Dataset
from ctDataset import ContrastiveDataset
# import json_lines
import pandas as pd
import json
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
class Data:
    
    def __init__(self, args):
        set_seed(args.seed)
        
        # ========================================================================
        # --- 全新的、符合SOP标准的数据加载与处理逻辑 ---
        # ========================================================================
        
        # 1. 根据SOP标准从YAML参数构建文件路径
        origin_train_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'train.tsv')
        origin_eval_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'dev.tsv')
        origin_test_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'test.tsv')
        
        labeled_train_path = os.path.join(args.data_dir, args.dataset, 'labeled_data', str(args.labeled_ratio), 'train.tsv')
        
        # 2. 使用pandas加载标准化的TSV文件
        origin_train_df = pd.read_csv(origin_train_path, sep='\t')
        origin_eval_df = pd.read_csv(origin_eval_path, sep='\t')
        origin_test_df = pd.read_csv(origin_test_path, sep='\t')
        labeled_train_df = pd.read_csv(labeled_train_path, sep='\t')
        
        df_train = labeled_train_df
        df_train['text'] = origin_train_df['text']
        
        # 3. 从标准化的.list文件加载已知类
        known_label_path = os.path.join(
            args.data_dir,
            args.dataset,
            'label',
            f'fold{args.fold_num}',
            f'part{args.fold_idx}',
            f'label_known_{args.known_cls_ratio}.list'
        )
        self.known_label_list = pd.read_csv(known_label_path, header=None)[0].tolist()
        self.num_labels = len(self.known_label_list)

        self.p, self.n = args.p, args.n

        # 4. 筛选数据，构建examples
        # Boundary Adjustment阶段的训练集只使用已知类
        train_examples_df = df_train[df_train.label.isin(self.known_label_list)]
        eval_examples_df = origin_eval_df[origin_eval_df.label.isin(self.known_label_list)]
        
        # 测试集包含所有类别，并将未知类标签替换为'unseen'
        test_examples_df = origin_test_df
        test_examples_df.loc[~test_examples_df['label'].isin(self.known_label_list), 'label'] = '<UNK>'

        # 将DataFrame转换为旧代码期望的InputExample格式
        processor = DatasetProcessor()
        self.train_examples = processor._create_examples_from_df(train_examples_df, 'train')
        self.eval_examples = processor._create_examples_from_df(eval_examples_df, 'eval')
        self.test_examples = processor._create_examples_from_df(test_examples_df, 'test')
        
        if args.dataset == 'oos':
            self.unseen_token = 'oos'
        else:
            self.unseen_token = '<UNK>'
        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_token]

        # 5. 获取DataLoader (保留原逻辑)
        self.train_dataloader = self.get_loader(self.train_examples, args, 'train')
        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')

        # 6. 设置其他属性 (保留原逻辑)
        if args.dataset == 'oos':
            self.unseen_token = 'oos'
        else:
            self.unseen_token = '<UNK>'
        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_token]

    def get_examples(self, processor, args, mode='train'):
        ori_examples = processor.get_examples(self.data_dir, mode)
        examples = []
        if mode == 'train':
            for example in ori_examples:
                if (example.label in self.known_label_list) and (np.random.uniform(0, 1) <= args.labeled_ratio):
                    examples.append(example)
        elif mode == 'eval':
            for example in ori_examples:
                if (example.label in self.known_label_list):
                    examples.append(example)
        elif mode == 'test':
            for example in ori_examples:
                if (example.label in self.label_list) and (example.label is not self.unseen_token):
                    examples.append(example)
                else:
                    example.label = self.unseen_token
                    examples.append(example)
        return examples

    def get_loader(self, examples, args, mode='train', aug_examples=None):
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_dir, do_lower_case=True)
        features = convert_examples_to_features(examples, self.label_list, args.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        guid = [x.guid for x in examples]
        neg_featrues = None
        neg_guid =None
        if mode == 'train' and args.neg_from_gen:
            neg_featrues = convert_examples_to_features(aug_examples, self.label_list, args.max_seq_length, tokenizer)
            neg_guid=[x.guid for x in aug_examples]
        if self.p + self.n == 0:
            datatensor = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
            if mode == 'train':
                sampler = RandomSampler(datatensor)
                dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.train_batch_size)    
            elif mode == 'eval' or mode == 'test':
                sampler = SequentialSampler(datatensor)
                dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.eval_batch_size) 
        else:
            if mode == 'train':
                datatensor = ContrastiveDataset(input_ids, input_mask, segment_ids, label_ids, self.p, self.n, neg_from_gen=args.neg_from_gen,
                guid=guid, neg_examples=neg_featrues, neg_guid=neg_guid)
                sampler = RandomSampler(datatensor)
                dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.train_batch_size, collate_fn=datatensor.collate_fn)
            elif mode == 'eval' or mode == 'test':
                datatensor = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
                sampler = SequentialSampler(datatensor)
                dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.eval_batch_size) 
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

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv_old(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        df = pd.read_csv(input_file, sep='\t')
        df = df.T.to_dict()
        lines = []
        for index, line in df.items():
            lines.append([index, (line['text'], line['label'])])
        return lines

class DatasetProcessor(DataProcessor):

    def _create_examples_from_df(self, df, set_type):
        """从DataFrame创建examples的辅助函数"""
        examples = []
        for i, row in df.iterrows():
            guid = "%s-%s" % (set_type, i)
            text_a = row['text']
            label = row['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))
            
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines:
            # if i == 0:
            #     continue
            i, line = line
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            # if set_type == 'aug':
            #     label = "<GENOTHER>"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

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


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()

if __name__ == '__main__':
    from init_parameter import init_model
    parser = init_model()
    args = parser.parse_args()
    args.neg_from_gen = False
    data = Data(args)
    iter_train = iter(data.train_dataloader)
    print(iter_train.next())