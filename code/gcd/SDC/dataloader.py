from utils.utils import *
import pandas as pd
class Data:

    def __init__(self, args):
        set_seed(args.seed)
        max_seq_lengths = {'hwu': 20, 'clinc': 30, 'banking': 65, 'stackoverflow': 65, 'mcid': 65, 'ecdt': 65}
        beta_list = {'hwu': 0.05, 'clinc': 0.42, 'banking': 0.03, 'stackoverflow': 0.05, 'mcid': 0.05, 'ecdt': 0.05}
        args.max_seq_length = max_seq_lengths[args.dataset]
        self.beta = beta_list[args.dataset]
        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.all_label_list = processor.get_labels(self.data_dir)
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        np.random.seed(args.seed)
        # self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        self.known_label_list = pd.read_csv(f"{self.data_dir}/label/label_{args.known_cls_ratio}.list", header=None)[0].tolist()
        
        self.known_train_sample = pd.read_csv(f"{self.data_dir}/labeled_data/train_{args.labeled_ratio}.tsv", sep='\t')
        self.known_train_sample = self.known_train_sample[self.known_train_sample['label'].isin(self.known_label_list)]

        self.known_eval_sample = pd.read_csv(f"{self.data_dir}/labeled_data/dev_{args.labeled_ratio}.tsv", sep='\t')
        self.known_eval_sample = self.known_eval_sample[self.known_eval_sample['label'].isin(self.known_label_list)]

        self.known_lab = [i for i in range(len(self.known_label_list))]
        # self.known_lab = [self.all_label_list.tolist().index(i) for i in self.known_label_list]
        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)

        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(processor, args, 'train')
        print('num_labeled_samples', len(self.train_labeled_examples))
        print('num_unlabeled_samples', len(self.train_unlabeled_examples))
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')
        self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples, args, 'train')
        self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids = self.get_semi(self.train_labeled_examples, self.train_unlabeled_examples, args)
        self.train_semi_dataset, self.train_semi_dataloader = self.get_semi_loader(self.semi_input_ids, self.semi_input_mask,
        self.semi_segment_ids, self.semi_label_ids, args)
        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')

    def get_examples(self, processor, args, mode='train'):
        ori_examples = processor.get_examples(self.data_dir, mode)

        if mode == 'train':
            # train_labels = np.array([example.label for example in ori_examples])
            # train_labeled_ids = []
            # for label in self.known_label_list:
            #     num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
            #     pos = list(np.where(train_labels == label)[0])
            #     random.seed(args.seed)
            #     train_labeled_ids.extend(random.sample(pos, num))
                
            # train_labeled_examples, train_unlabeled_examples = [], []
            # for idx, example in enumerate(ori_examples):
            #     if idx in train_labeled_ids:
            #         train_labeled_examples.append(example)
            #     else:
            #         train_unlabeled_examples.append(example)

            train_labeled_examples, train_unlabeled_examples = [], []
            for idx, example in enumerate(ori_examples):
                if example.text_a in self.known_train_sample['text'].tolist() and example.label in self.known_train_sample['label'].tolist():
                    train_labeled_examples.append(example)
                else:
                    train_unlabeled_examples.append(example)

            return train_labeled_examples, train_unlabeled_examples

        elif mode == 'eval':
            eval_examples = []
            for example in ori_examples:
                if example.label in self.known_label_list:
                    eval_examples.append(example)
            return eval_examples

        elif mode == 'test':
            return ori_examples

    def get_semi(self, labeled_examples, unlabeled_examples, args):

        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        labeled_features = convert_examples_to_features(labeled_examples, self.known_label_list, args.max_seq_length, tokenizer)
        unlabeled_features = convert_examples_to_features_semi(unlabeled_examples, self.known_label_list, 
        self.all_label_list, args.max_seq_length, tokenizer)

        labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
        labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
        labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
        labeled_label_ids = torch.tensor([f.label_id for f in labeled_features], dtype=torch.long)

        unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
        unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
        # unlabeled_label_ids = torch.tensor([-1 for f in unlabeled_features], dtype=torch.long)
        unlabeled_label_ids = torch.tensor([f.label_id for f in unlabeled_features], dtype=torch.long)

        semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
        semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
        semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
        semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
        return semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids

    def get_semi_loader(self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args):
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids)
        semi_sampler = RandomSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size=args.train_batch_size)

        return semi_data, semi_dataloader

    def get_loader(self, examples, args, mode='train'):
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

        if mode == 'train' or mode == 'eval':
            features = convert_examples_to_features(examples, self.known_label_list, args.max_seq_length, tokenizer)
        elif mode == 'test':
            features = convert_examples_to_features_semi(examples, self.known_label_list, self.all_label_list, args.max_seq_length, tokenizer)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

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
        if data_dir == 'data/banking':
            labels = np.unique(np.array(test['label']))
        else:
            labels = [str(label) for label in test['label']]
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
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
            
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)


        input_mask = [1] * len(input_ids)


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

def convert_examples_to_features_semi(examples, labeled_list, all_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(labeled_list):
        label_map[label] = i
    
    novel_list = [label for label in all_list if label not in labeled_list]
    for i, label in enumerate(novel_list):
        label_map[label] = i + len(labeled_list)
    

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)


        input_mask = [1] * len(input_ids)

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
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()