import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BertForSequenceClassification
import torch
from utils import get_best_checkpoint

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = args.backbone
        checkpoint_path = get_best_checkpoint(args.checkpoint_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        if args.backbone == 'bert':
            self.llm = self.model.bert
        elif args.backbone == 'roberta':
            self.llm = self.model.roberta
        self.fc = self.model.classifier

    def features(self, x):
        outputs = self.llm(**x)
        if self.backbone == 'bert':
            features = outputs.pooler_output
        elif self.backbone == 'roberta':
            features = outputs.last_hidden_state
        return features

    def forward(self, x):
        outputs = self.model(**x)
        return outputs.logits
