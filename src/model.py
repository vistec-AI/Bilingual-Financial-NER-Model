import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Tuple, Dict, List
from huggingface_hub import PyTorchModelHubMixin

class TokenClassificationModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, pretrained_model_name, id2l=3, ne_scheme="BIO", dropout=0.5):
        super(TokenClassificationModel, self).__init__()
        self.id2l = id2l
        self.dropout = dropout
        self.ne_scheme = ne_scheme
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.droupout = nn.Dropout(dropout)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(self.id2l))

    def forward(self, features):
        outputs = self.bert(**features)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(self.droupout(sequence_output))

        return (logits, sequence_output)
