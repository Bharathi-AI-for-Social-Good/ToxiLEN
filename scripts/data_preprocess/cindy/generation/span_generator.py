import json
import pandas as pd
import openai
import time
import os

# 你的 OpenAI API Key
client = openai.OpenAI(api_key="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")

# 你的 GPT-4.1-mini 情感抽取 prompt
def build_prompt(text_inputs):
    return [
        {
            "role": "system",
            "content": "你是一个中文情绪识别专家，擅长从文本中识别出针对女性的贬低情绪，以及其他一般性情绪表达。"
        },
        {
            "role": "user",
            "content": """
请从以下中文文本中提取两类情绪词或短语：

1. 贬低女性的情绪词（字段名："misogynistic"）：
   - 包括侮辱、物化、歧视、控制、攻击女性的表达；
   - 包括带有“讽刺、反语、调侃、社会性别压迫”意味的表达；
   - 包括模糊但指向女性贬低或性别规训的内容，例如“你不结婚以后怎么办”。

2. 非贬低女性的情绪词（字段名："general_emotion"）：
   - 包括普通的正向或负向情绪词，如愤怒、快乐、焦虑、安心、喜欢等；
   - 不针对女性或性别群体，仅表达一般情绪。

输出要求：
- 返回 JSON 格式，包含两个字段："misogynistic" 和 "general_emotion"；
- 每个字段为字符串列表；
- 如果某类情绪不存在，返回空列表；
- 仅输出 JSON，不要添加解释说明或其他内容。

以下是参考示例：

示例1：
文本：你这种女人就该被管教，天天想着找男人。
返回：
{
  "misogynistic": ["该被管教", "找男人"],
  "general_emotion": []
}

示例2：
文本：她长得很漂亮，性格也温柔。
返回：
{
  "misogynistic": [],
  "general_emotion": ["漂亮", "温柔"]
}

示例3：
文本：她真是个臭婊子，还敢出来讲话，简直不要脸。
返回：
{
  "misogynistic": ["臭婊子", "不要脸"],
  "general_emotion": []
}

示例4：
文本：婚前说孩子我带，婚后却让我一个人扛。
返回：
{
  "misogynistic": ["让我一个人扛"],
  "general_emotion": []
}

现在请提取以下文本中的情绪词：
文本：\n""" + text_inputs
        }
    ]

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

def process_file(input_json_path, labels_csv_path, output_json_path, model="gpt-4.1-mini-2025-04-14"):
    import math

    # 1. 读取输入数据
    df = pd.read_csv(input_json_path, encoding="gbk")
    print("df columns:", df.columns)
    labels_df = pd.read_csv(labels_csv_path, encoding="gbk")
    print("labels_df columns:", labels_df.columns)

    items = pd.merge(df, labels_df, on="images_name", how="left")
    items = items.to_dict(orient="records")
    items = [ {
        "filename": item.get("filename", ""),
        "text": item.get("text", ""),
    } for item in items ]

    # 2. 加载断点
    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as rf:
            results = json.load(rf)
        done_set = set([item["filename"] for item in results])
        print(f"已处理: {len(done_set)} 条，继续断点续跑。")
    else:
        results = []
        done_set = set()

    total = len(items)
    for idx, item in enumerate(items):
        filename = item.get("filename", "")
        if filename in done_set:
            continue

        text = item.get("text", "")
        if isinstance(text, str) and text.strip():
            prompt = build_prompt(text)
            resp = call_gpt_api(prompt, model=model)
            try:
                span_data = json.loads(resp)
                mis_span = span_data.get("misogynistic", [])
                general_emotion = span_data.get("general_emotion", [])
            except:
                mis_span, general_emotion = [], []
            time.sleep(1.1)
        else:
            mis_span, general_emotion = [], []

        result_item = {
            "filename": filename,
            "text": text if isinstance(text, str) else "",
            "mis_span": mis_span,
            "general_emotion": general_emotion
        }
        results.append(result_item)

        # 每次保存（断点续跑）
        with open(output_json_path, "w", encoding="utf-8") as wf:
            json.dump(results, wf, ensure_ascii=False, indent=2)

        print(f"[{len(results)}/{total}] {filename} done")

    print(f"\n✅ 全部完成！结果已保存到：{output_json_path}")

# 示例调用
process_file(
    "data/cindy/json/train.json",
    "data/cindy/labels/train_labels.csv",
    "data/cindy/json/train_result.json",
    model="gpt-4.1-mini-2025-04-14"
)