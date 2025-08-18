import json
import pandas as pd
import openai
import time
import os

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key="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")  # ← 请替换为你自己的 key

def build_prompt(text_inputs):
    return [
        {
            "role": "system",
            "content": "You are a Chinese linguist skilled at analyzing implicit meaning, social attitude, and emotional undertones."
        },
        {
            "role": "user",
            "content": f"""
Analyze the hidden meaning, social attitude, or emotional tone conveyed by the following Chinese text.

Keep your response concise and informative. Respond in no more than **3 short sentences**.

Text:
{text_inputs}
"""
        }
    ]

# 调用 OpenAI GPT 接口
def call_gpt_api(messages, model="gpt-4.1-mini-2025-04-14"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API error: {e}")
        return None

# 解析模型返回结果（支持 JSON / 普通文本）
def format_meaning_output(resp_str):
    try:
        data = json.loads(resp_str)
        if isinstance(data, dict):
            return data.get("implicit_meaning", "").strip() or "No implicit meaning"
        return resp_str.strip()
    except Exception:
        return resp_str.strip() if resp_str else "No implicit meaning"

# 主处理流程
def process_file(input_json_path, output_json_path, model="gpt-4.1-mini-2025-04-14"):
    # 读取输入 JSON 文件
    with open(input_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # 加载已处理结果（断点续跑）
    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as rf:
            results = json.load(rf)
        done_set = set([item["filename"] for item in results])
        print(f"Resuming from checkpoint. Already processed: {len(done_set)}")
    else:
        results = []
        done_set = set()

    total = len(items)
    for idx, item in enumerate(items):
        filename = item.get("filename", "").strip()
        if not filename or filename in done_set:
            continue

        text = item.get("text", "").strip()
        label = item.get("label", "")

        if not text:
            span = "No implicit meaning"
        else:
            prompt = build_prompt(text)
            resp = call_gpt_api(prompt, model=model)
            background = format_meaning_output(resp)
            time.sleep(1.1)

        result_item = {
            "filename": filename,
            "text": text,
            "label": label,
            "background": background
        }
        results.append(result_item)

        # 每条写一次（可防止中断丢失）
        with open(output_json_path, "w", encoding="utf-8") as wf:
            json.dump(results, wf, ensure_ascii=False, indent=2)

        print(f"[{len(results)}/{total}] {filename} done")

    print(f"\n✔️ All done! Results saved to {output_json_path}")

# ✅ 使用示例（请根据你的实际路径调整）
process_file(
    input_json_path="data/toxi_mm/json/test.json",
    output_json_path="data/test_toxi_mm_background.json",
    model="gpt-4.1-mini-2025-04-14"
)
