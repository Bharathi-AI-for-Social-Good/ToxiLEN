import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import json

class ToxiMMDataset(Dataset):
    def __init__(self, config):
        self.image_dir = config.get("image_dir", "data/toximm/images")
        data_path = config.get("data", "data/toximm/train.json")

        with open(data_path, "r") as f:
            self.data = json.loads(f.read())

        self.labels = [int(row["label"]) for row in self.data]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        image_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(image_path).convert("RGB")

        return {
            'image': image,
            'prompt': row['prompt'],
            'explanation': row['explanation'],
            'caption': row['caption'],
            'label': torch.tensor(row['label'], dtype=torch.long)
        }
