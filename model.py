import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertTokenizer
import torch


bert_path = "../models/TORCH_BERT"
tokenizer_path = "../models/tokenizer"

class ClassificationModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", cache_dir=bert_path
        )   
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
            cache_dir= tokenizer_path,
        )
        self.linear1 = nn.Linear(in_features=768, out_features=256)
        self.dropout1 = nn.Dropout()
        self.linear2 = nn.Linear(in_features=256, out_features=64)
        self.dropout2 = nn.Dropout()
        self.linear3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        tokens = self.tokenizer(x, max_length = 512, padding = "max_length", truncation = True, return_tensors="pt")
        # with torch.no_grad():
        x = self.bert(**tokens)[1]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x
        





        