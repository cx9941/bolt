from utils.tools import *

max_seq_lengths = {'clinc':30, 'stackoverflow':45, 'banking':55, 'hwu':55, 'mcid':55, 'ecdt':55}
TOPK = {'clinc':50, 'stackoverflow':500, 'banking':50, 'hwu':55, 'mcid':55, 'ecdt':55}
class Data:
    
    def __init__(self, args):
        set_seed(args.seed)
        args.max_seq_length = max_seq_lengths[args.dataset]
        args.pretrain_dir = './pretrain_model/premodel_' + args.dataset + '_' + str(args.seed)
        args.save_model_path = './model_' + args.dataset + '_' + str(args.seed)
        args.topk = TOPK[args.dataset]
        processor = DatasetProcessor()
        args.cluster_num_factor = 1
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.all_label_list = processor.get_labels(self.data_dir)
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        # if self.n_known_cls > 0:
        #     self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        # else:
        #     self.known_label_list = []
        self.known_label_list = pd.read_csv(f"{self.data_dir}/label/label_{args.known_cls_ratio}.list", header=None)[0].tolist()

        self.known_train_sample = pd.read_csv(f"{self.data_dir}/labeled_data/train_{args.labeled_ratio}.tsv", sep='\t')
        self.known_train_sample = self.known_train_sample[self.known_train_sample['label'].isin(self.known_label_list)]

        self.known_eval_sample = pd.read_csv(f"{self.data_dir}/labeled_data/dev_{args.labeled_ratio}.tsv", sep='\t')
        self.known_eval_sample = self.known_eval_sample[self.known_eval_sample['label'].isin(self.known_label_list)]

        self.known_lab = [int(np.where(self.all_label_list== a)[0]) for a in self.known_label_list] 
        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)
        
        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(processor, args, 'train')
        print('num_labeled_samples',len(self.train_labeled_examples))
        print('num_unlabeled_samples',len(self.train_unlabeled_examples))
        args.num_labeled_examples = len(self.train_labeled_examples)
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')

        if self.n_known_cls > 0:
            self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples, args, 'train')
        else:
            self.train_labeled_dataloader = None

        self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids, self.semi_idx_ids = self.get_semi(self.train_labeled_examples, self.train_unlabeled_examples, args)


        dir_name = f"{args.save_results_path}/{args.dataset}_{args.known_cls_ratio}_{args.labeled_ratio}"
        os.makedirs(dir_name, exist_ok=True)
        np.save(f"{dir_name}/sample_idx.npy", self.semi_idx_ids)

        self.train_semi_dataset, self.train_semi_dataloader = self.get_semi_loader(self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids, args)

        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')
        
    def get_examples(self, processor, args, mode = 'train'):
        ori_examples = processor.get_examples(self.data_dir, mode)
        
        if mode == 'train':
            # 将 known_train_sample 也转为全小写来匹配，并使用高效的 set
            known_samples_set = set(zip(self.known_train_sample['text'].str.lower(), self.known_train_sample['label'].str.lower()))

            train_labeled_examples, train_unlabeled_examples = [], []
            for example in ori_examples:
                # 现在两边都是小写，可以正确匹配了
                if (example.text_a, example.label) in known_samples_set:
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
        
        else:
            raise NotImplementedError(f"Mode {mode} not found")

    def get_semi(self, labeled_examples, unlabeled_examples, args):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if self.n_known_cls > 0:
            labeled_features = convert_examples_to_features(labeled_examples, self.known_label_list, args.max_seq_length, tokenizer)
            labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
            labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
            labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
            labeled_label_ids = torch.tensor([f.label_id for f in labeled_features], dtype=torch.long)  
        else:
            labeled_features = None

        unlabeled_features = convert_examples_to_features(unlabeled_examples, self.all_label_list, args.max_seq_length, tokenizer)
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

        idx_list = np.array([int(_.guid.split('-')[-1]) for _ in labeled_examples] + [int(_.guid.split('-')[-1]) for _ in unlabeled_examples])

        return semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, idx_list

    def get_semi_loader(self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args):
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids)
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size = args.train_batch_size) 

        return semi_data, semi_dataloader

    def get_loader(self, examples, args, mode = 'train'):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        
        if mode == 'train' or mode == 'eval':
            features = convert_examples_to_features(examples, self.known_label_list, args.max_seq_length, tokenizer)
        elif mode == 'test':
            features = convert_examples_to_features(examples, self.all_label_list, args.max_seq_length, tokenizer)
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
                line = [l.lower() for l in line]
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
        else:
            raise NotImplementedError(f"Mode {mode} not found")

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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):      
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

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


