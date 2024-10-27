# model/model.py

import torch
from transformers import AutoModel

class LegalRetrievalModel(torch.nn.Module):
    def __init__(self, tokenizer, num_cls_tokens=3):
        super(LegalRetrievalModel, self).__init__()
        self.bert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.num_cls_tokens = num_cls_tokens
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size * num_cls_tokens, 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer  # Ensure the tokenizer is passed and assigned
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_outputs = outputs.last_hidden_state[:, :self.num_cls_tokens, :]
        cls_pooled = cls_outputs.view(cls_outputs.size(0), -1)
        return self.classifier(cls_pooled)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert(**inputs)
            cls_outputs = outputs.last_hidden_state[:, :self.num_cls_tokens, :]
            return cls_outputs.mean(dim=1).cpu().numpy()
