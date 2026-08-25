import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class Model(nn.Module):
    def __init__(self, model_id, device="cpu"):
        super(Model, self).__init__()
        self.model_id = model_id
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, device_map=self.device)

    def forward(self, x):
        tokenized_input = self.tokenizer(x, return_tensors="pt").to(self.device)
        output = self.model(**tokenized_input)
        return output

    