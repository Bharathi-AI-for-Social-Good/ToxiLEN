import pandas as pd
import os

file_path = {
    "train":"data/cindy/all/train.csv",
    "test":"data/cindy/all/test.csv",
    "val":"data/cindy/all/dev.csv"
}

os.makedirs("data/cindy/json", exist_ok=True)

for key, value in file_path.items():
    df = pd.read_csv(value)
    df.to_json(f"data/cindy/json/{key}.json", orient="records", force_ascii=False)

