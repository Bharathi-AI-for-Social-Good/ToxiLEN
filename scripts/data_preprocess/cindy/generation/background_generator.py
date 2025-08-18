import json
import pandas as pd
import openai
import time
import os
import random
from groq import Groq

client = openai.OpenAI(api_key="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")  # ← 请替换为你自己的 key

groq_client = Groq(api_key="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")

def build_prompt(text_inputs, prompt_type="zero_shot"):
    sys_msg = {
        "role": "system",
        "content": "You are a Chinese linguist skilled at analyzing implicit meaning, social attitude, and emotional undertones."
    }
    user_msg = {
        "role": "user",
        "content": f"""
Analyze the hidden meaning, social attitude, or emotional tone conveyed by the following Chinese text.

Keep your response concise and informative. Respond in no more than **3 short sentences**.

Text:
{text_inputs}
"""
    }
    if prompt_type == "zero_shot":
        return [sys_msg, user_msg]
    elif prompt_type == "few_shot":
        few_shot_examples = [
            {
                "role": "user",
                "content": """
Text:
你怎么又在刷微博？时间都去哪了？

Response:
这句话带有调侃和幽默的语气，隐含对信息泛滥和刷屏现象的讽刺，表达了对浪费时间的无奈。
"""
            },
            {
                "role": "user",
                "content": """
Text:
今天也是貌美如花元气满满的一天。

Response:
这句话表达了积极向上的情绪，透露出对生活的乐观态度，语气轻松愉快。
"""
            }
        ]
        return [sys_msg] + few_shot_examples + [user_msg]
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

def call_gpt_api(messages, model="gpt-4o"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

def call_llama_api(messages, model="llama3.1:8b"):
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API error: {e}")
        return None

def run_comparative_eval(input_json_path, output_json_path, sample_size=20):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sampled_data = random.sample(data, min(sample_size, len(data)))
    all_results = []

    groq_models = [
        "qwen/qwen3-32b",
        "deepseek-r1-distill-llama-70b",
        "llama3-70b-8192"
    ]

    for idx, item in enumerate(sampled_data):
        filename = item.get("filename")
        text = item.get("text", "")
        label = item.get("label", "")

        entry = {
            "filename": filename,
            "text": text,
            "label": label,
            "results": []
        }

        for prompt_type in ["zero_shot", "few_shot"]:
            messages = build_prompt(text, prompt_type=prompt_type)

            # OpenAI models
            for model_name in ["gpt-4o", "gpt-4.1-mini-2025-04-14"]:
                response = call_gpt_api(messages, model=model_name)
                time.sleep(1.1)

                entry["results"].append({
                    "prompt_type": prompt_type,
                    "model": model_name,
                    "response": response
                })

            # Groq models
            for groq_model in groq_models:
                response = call_llama_api(messages, model=groq_model)
                time.sleep(0.5)

                entry["results"].append({
                    "prompt_type": prompt_type,
                    "model": groq_model,
                    "response": response
                })

        all_results.append(entry)
        print(f"[{idx+1}/{sample_size}] ✅ Processed: {filename}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 All results saved to: {output_json_path}")

# ---------- 执行任务 -------------
run_comparative_eval(
    input_json_path="data/cindy/json/train.json",
    output_json_path="data/cindy/analysis/sample_eval.json",
    sample_size=20
)