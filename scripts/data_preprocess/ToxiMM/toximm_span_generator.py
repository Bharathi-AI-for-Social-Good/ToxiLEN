import json
import pandas as pd
import openai
import time

# 你的 OpenAI API Key
client = openai.OpenAI(api_key="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")

# 你的 GPT-4.1-mini 情感抽取 prompt
def build_prompt(text_inputs):
    return [
        {
            "role": "system",
            "content": "You are an expert in Chinese emotion recognition, especially skilled at identifying hateful and general emotional expressions from text."
        },
        {
            "role": "user",
            "content": """
            Please extract two types of emotional words or phrases (spans) from the following Chinese text:

            1. Hateful emotional expressions (field name: "hateful"):
              - Includes expressions that show hatred, discrimination, humiliation, verbal attack, or social oppression toward individuals or groups;
              - Can target any group (e.g., women, men, LGBTQ+, regions, professions, races, or individuals);
              - Includes both direct and implicit expressions of hate, mockery, sarcasm, and social exclusion.

            2. General emotional expressions (field name: "general_emotion"):
              - Includes typical positive or negative emotions such as anger, happiness, anxiety, relief, affection, etc.;
              - Should not express hate or target a specific group;
              - Purely reflects general emotional states.

            Output format:
            - Return a JSON object with two fields: "hateful" and "general_emotion";
            - Each field should be a list of strings;
            - If a category does not exist, return an empty list;
            - Output **only** the JSON result, with no additional explanation or comments.

            Reference examples:

            --- Chinese Examples ---

            Example 1:
            Text: 这些穷鬼根本就不该活着，社会的垃圾。
            Return:
            {
              "hateful": ["穷鬼", "不该活着", "社会的垃圾"],
              "general_emotion": []
            }

            Example 2:
            Text: 今天心情特别好，阳光真棒！
            Return:
            {
              "hateful": [],
              "general_emotion": ["心情特别好", "阳光真棒"]
            }

            Example 3:
            Text: 这帮外地人一天到晚就知道占便宜，滚回去吧！
            Return:
            {
              "hateful": ["占便宜", "滚回去"],
              "general_emotion": []
            }

            Example 4:
            Text: 我这几天很焦虑，总觉得事情要出问题。
            Return:
            {
              "hateful": [],
              "general_emotion": ["焦虑", "事情要出问题"]
            }

            Example 5:
            Text: 他简直像个变态一样盯着别人看，太恶心了。
            Return:
            {
              "hateful": ["变态", "恶心"],
              "general_emotion": []
            }

            --- English Examples ---

            Example 6:
            Text: These idiots should be locked up—they’re a plague on society.
            Return:
            {
              "hateful": ["idiots", "should be locked up", "plague on society"],
              "general_emotion": []
            }

            Example 7:
            Text: I’m feeling really anxious about the presentation tomorrow.
            Return:
            {
              "hateful": [],
              "general_emotion": ["anxious"]
            }

            Example 8:
            Text: Go back to where you came from. You people ruin everything.
            Return:
            {
              "hateful": ["Go back to where you came from", "You people ruin everything"],
              "general_emotion": []
            }

            Example 9:
            Text: I’m super excited to see my family again after all this time.
            Return:
            {
              "hateful": [],
              "general_emotion": ["super excited"]
            }

            Example 10:
            Text: He acts like some kind of creep. Totally disgusting.
            Return:
            {
              "hateful": ["creep", "Totally disgusting"],
              "general_emotion": []
            }

            Now, please extract the emotional spans from the following text:
            Text:\n""" + text_inputs
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
    
def format_span_output(json_str):
    try:
        data = json.loads(json_str)
        hateful = data.get("hateful", [])
        general = data.get("general_emotion", [])
        if not hateful and not general:
            return "No emotion spans detected"
        span_str = ""
        if hateful:
            span_str += "Detected Hateful spans: " + ", ".join(hateful)
        if general:
            if span_str:
                span_str += "; "
            span_str += "Detected general emotions spans: " + ", ".join(general)
        return span_str
    except Exception as e:
        return "No emotion spans detected"

def process_file(input_json_path, output_json_path, model="gpt-4.1-mini-2025-04-14"):
    with open(input_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    results = []
    for idx, item in enumerate(items):
        filename = item.get("path", "")
        text = item.get("text", "")
        label = item.get("label", "")
        prompt = build_prompt(text)
        resp = call_gpt_api(prompt, model=model)
        span = format_span_output(resp)
        result_item = {
            "filename": filename,
            "text": text,
            "label": label,
            "span": span
        }
        results.append(result_item)

        # 每条写一次
        with open(output_json_path, "w", encoding="utf-8") as wf:
            json.dump(results, wf, ensure_ascii=False, indent=2)

        print(f"[{idx+1}/{len(items)}] {filename} done")
        time.sleep(1.1)

    print(f"\nSaved to {output_json_path}")

# 用法举例
process_file(
    "data/train_data_discription_2.0.json",
    "data/train_data_discription_2.0_result.json",
    model="gpt-4.1-mini-2025-04-14"
)