import os, json, pandas as pd
from tqdm import tqdm
from app.pipeline import evaluate_pipeline


def generate_evaluations():
    # 组合路径
    response_path = f"data/counter_narrative/results/gen_results/ours.json"

    if not os.path.exists(response_path):
        print(f"[WARNING] Missing response file: {response_path}")
        return []

    # 读取这个组合对应的 response 数据
    with open(response_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = pd.DataFrame(dataset)
    background = pd.read_csv("data/cindy/background/all_background.csv", encoding="utf-8")
    dataset = pd.merge(dataset, background, on="filename", how="left")
    data = dataset.to_dict(orient="records")
    
    outputs = []
    for row in tqdm(data, desc="gpt-4o", leave=False):
        context = {
            "filename": row['filename'],
            "hate_speech": row['hate_speech'],
            "response": row['response'],
        }
        
        print("[DEBUG context]", context)  # 确保 response 不为空
        evaluations = evaluate_pipeline(context)
        outputs.append({
            "filename": context["filename"],
            "hate_speech": context["hate_speech"],
            "evaluations": evaluations
        })
    return outputs



outs = generate_evaluations()
out_path = f"data/counter_narrative/results/evaluations/ours.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(outs, f, ensure_ascii=False, indent=2)


