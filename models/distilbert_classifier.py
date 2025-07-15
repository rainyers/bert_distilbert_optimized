
import torch
from torch import nn
from transformers import DistilBertModel

class DistilBERTClassifier(nn.Module):
    def __init__(self, pretrained_model="distilbert-base-uncased", num_labels=4):
        super(DistilBERTClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits
