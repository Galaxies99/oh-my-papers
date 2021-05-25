import os
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class Bert(nn.Module):
    def __init__(self, seq_dim = 0, feature_dim = -1):
        super(Bert, self).__init__()
        
        if seq_dim not in [-1, 0]:
            raise AttributeError('seq_dim should be 0 (beginning) or -1 (ending).')

        # Potential Exception which occurs in macOS
        if platform.system() == 'Darwin':
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        bert_hidden = 768
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.bert_model = AutoModel.from_pretrained("bert-base-cased")
        self.seq_dim = seq_dim

        if feature_dim == -1:
            self.linear = None
        else:
            self.linear = nn.Linear(bert_hidden, feature_dim)

    def forward(self, context):
        tokens = self.tokenizer(context, return_tensors = 'pt')
        res = self.bert_model(**tokens).last_hidden_state[:, self.seq_dim, :]
        return self.linear(res) if self.linear is not None else res


class Specter(nn.Module):
    def __init__(self):
        super(Specter, self).__init__()
        
        # Potential Exception which occurs in macOS
        if platform.system() == 'Darwin':
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
        self.specter = AutoModel.from_pretrained("allenai/specter")
    
    def forward(self, papers):
        title_abs = [(paper.get('title', '') + self.tokenizer.sep_token + paper.get('abstract', '')) for paper in papers]
        inputs = self.tokenizer(title_abs, padding = True, truncation = True, return_tensors = "pt", max_length = 512)
        res = self.specter(**inputs).last_hidden_state[:, 0, :]
        return res