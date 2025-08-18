import os, json, pandas as pd
from tqdm import tqdm
from app.pipeline import evaluate_pipeline


prompt_types = ["knowledge", "few", "zero"]
models       = ["gpt-4o","llama-3.3-70b-versatile",
                "qwen/qwen3-32b",
                "deepseek-r1-distill-llama-70b"]

os.makedirs("data/counter_narrative/results/evaluations", exist_ok=True)

def generate_evaluations(model_name, prompt_type):
    # 组合路径
    response_path = f"data/counter_narrative/results/gen_results/{model_name.replace('/', '_')}_{prompt_type}.json"

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
    for row in tqdm(data, desc=f"{model_name} | {prompt_type}", leave=False):
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


for m in models:
    for t in prompt_types:
        outs = generate_evaluations(m, t)
        out_path = f"data/counter_narrative/results/evaluations/{m.replace('/','_')}_{t}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(outs, f, ensure_ascii=False, indent=2)
