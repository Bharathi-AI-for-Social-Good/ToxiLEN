from torch import nn
from transformers import AutoModel, AutoTokenizer
import re, torch

class RobertaEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.model.to(self.device)

    def forward(self, text, return_tokens=False):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        if return_tokens:
            return outputs.last_hidden_state  # [B, L, 768]
        else:
            return outputs.last_hidden_state[:, 0, :]  # [CLS]


class TokenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        special = ["<HATE_SPAN>", "</HATE_SPAN>",
                   "<EMO_SPAN>",  "</EMO_SPAN>",
                   "<NO_SPAN>"]

        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.tokenizer.add_tokens(special, special_tokens=True)

        self.model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

    def forward(self, prompt, return_tokens=False):
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        output = self.model(**encoding).last_hidden_state  # (B, L, 768)

        if return_tokens:
            return output  # All token embeddings
        else:
            return output[:, 0]