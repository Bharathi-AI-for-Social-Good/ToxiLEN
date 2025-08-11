import json

def clean_span_field(input_path: str, output_path: str):
    """
    清理 span 字段：
    - 对 label == 1 且包含 hateful span 的，保留仇恨表达内容（去除前缀与多余尾部）
    - 其他所有样本，将 span 字段清空
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned = []
    for item in data:
        label = item.get("label", 0)
        raw_span = item.get("span", "").strip()

        keep_span = ""
        if label == 1 and raw_span.startswith("Detected Hateful spans:"):
            # 去掉 hateful 前缀
            raw_span = raw_span.replace("Detected Hateful spans:", "").strip()

            # 去掉后续的任何 Detected 提示（如情绪、讽刺等）
            if "; Detected" in raw_span:
                raw_span = raw_span.split("; Detected")[0].strip()

            # 最终赋值
            keep_span = raw_span

        # 更新清洗后的字段
        item["span"] = keep_span
        cleaned.append(item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned data saved to: {output_path}")


input_json = "data/toxi_mm/json/test.json"        # 替换为原始 JSON 路径
output_json = "data/toxi_mm/json/test_cleaned.json"   # 清理后的输出路径
clean_span_field(input_json, output_json)