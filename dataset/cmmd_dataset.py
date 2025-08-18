import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import json

class MemeDataset(Dataset):
    def __init__(self, config):
        self.image_dir = config.get("image_dir")
        data_path = config.get("data")

        self.data = pd.read_csv(data_path)

        self.labels = self.data["label"].astype(int).tolist()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx] 

        image_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(image_path).convert("RGB")

        caption = row['captions']
        if not isinstance(caption, str):
          caption = "no caption available"

        return {
            'image': image,
            'prompt': row['prompt'],
            'explanation': row['background'],
            'caption': caption,
            'label': torch.tensor(row['label'], dtype=torch.long)
        }
