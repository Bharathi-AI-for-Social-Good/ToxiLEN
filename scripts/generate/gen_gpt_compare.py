import os, json, pandas as pd
from tqdm import tqdm
from app.pipeline import gpt_test_pipeline


data = pd.read_csv("data/counter_narrative/data/test.csv", encoding="utf-8") \
         .to_dict(orient="records")

prompt_types = ["few"]
models       = ["gpt-4o"]

os.makedirs("data/counter_narrative/results", exist_ok=True)

def generate_responses(dataset, model_name, prompt_type):
    outputs = []
    for row in tqdm(dataset, desc=f"{model_name} | {prompt_type}", leave=False):
        context = {
            "filename":   row.get("filename", ""),
            "hate_speech": row.get("text", ""),
            "background":  row.get("background", "")
        }
        res = gpt_test_pipeline(context, model=model_name, type=prompt_type)
        outputs.append({
            "filename": context["filename"],
            "hate_speech": context["hate_speech"],
            "response": res
        })
    return outputs


for m in models:
    for t in prompt_types:
        outs = generate_responses(data, m, t)
        out_path = f"data/counter_narrative/results/{m.replace('/','_')}_{t}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(outs, f, ensure_ascii=False, indent=2)
