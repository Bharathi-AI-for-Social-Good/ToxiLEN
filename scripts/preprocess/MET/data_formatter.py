import json
import re
from typing import Dict, List, Tuple

def parse_spans(span_text: str) -> Tuple[List[str], List[str]]:
    """
    解析span文本，提取仇恨言论片段和情感片段
    
    Args:
        span_text: 类似 "Detected Hateful spans: 草泥马, 甘霖娘; Detected general emotions spans: 暴晒, 热死了"
    
    Returns:
        (hate_spans, emotion_spans): 仇恨言论片段列表和情感片段列表
    """
    hate_spans = []
    emotion_spans = []
    
    if span_text == "No emotion spans detected":
        return hate_spans, emotion_spans
    
    # 提取仇恨言论片段
    hate_pattern = r"Detected Hateful spans:\s*([^;]+)"
    hate_match = re.search(hate_pattern, span_text)
    if hate_match:
        hate_spans_text = hate_match.group(1).strip()
        hate_spans = [span.strip() for span in hate_spans_text.split(",") if span.strip()]
    
    # 提取情感片段
    emotion_pattern = r"Detected general emotions spans:\s*(.+?)(?:$|;)"
    emotion_match = re.search(emotion_pattern, span_text)
    if emotion_match:
        emotion_spans_text = emotion_match.group(1).strip()
        emotion_spans = [span.strip() for span in emotion_spans_text.split(",") if span.strip()]
    
    return hate_spans, emotion_spans

def create_prompt(text: str, hate_spans: List[str], emotion_spans: List[str]) -> str:
    """
    根据文本和片段创建带有特殊标记的prompt
    
    Args:
        text: 原始文本
        hate_spans: 仇恨言论片段列表
        emotion_spans: 情感片段列表
    
    Returns:
        格式化后的prompt
    """
    if not hate_spans and not emotion_spans:
        return text
    
    # 合并所有片段并去重
    all_spans = list(set(hate_spans + emotion_spans))
    
    if hate_spans and emotion_spans:
        # 如果同时有仇恨和情感片段，先显示仇恨片段，再显示情感片段
        hate_text = ", ".join(hate_spans)
        emo_text = ", ".join(emotion_spans)
        return f"<HATE_SPAN> {hate_text} </HATE_SPAN> <EMO_SPAN> {emo_text} </EMO_SPAN> {text}"
    elif hate_spans:
        hate_text = ", ".join(hate_spans)
        return f"<HATE_SPAN> {hate_text} </HATE_SPAN> {text}"
    else:
        emo_text = ", ".join(emotion_spans)
        return f"<EMO_SPAN> {emo_text} </EMO_SPAN> {text}"

def convert_data_format(input_data: Dict) -> Dict:
    """
    将输入数据转换为目标格式
    
    Args:
        input_data: 包含filename, text, label, span的字典
    
    Returns:
        转换后的数据字典
    """
    filename = input_data["filename"]
    text = input_data["text"]
    label = input_data["label"]
    span = input_data["span"]
    
    # 解析spans
    hate_spans, emotion_spans = parse_spans(span)
    
    # 创建prompt
    prompt = create_prompt(text, hate_spans, emotion_spans)
    
    # 返回转换后的数据
    return {
        "filename": filename,
        "original_text": text,
        "label": label,
        "span": span,
        "prompt": prompt
    }

def batch_convert(input_file: str, output_file: str):
    """
    批量转换数据文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        converted_data = []
        
        if isinstance(data, list):
            for item in data:
                converted_item = convert_data_format(item)
                converted_data.append(converted_item)
        else:
            converted_data = convert_data_format(data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        print(f"转换完成！共处理 {len(converted_data) if isinstance(converted_data, list) else 1} 条数据")
        
    except Exception as e:
        print(f"转换过程中出现错误: {e}")

# 示例使用
if __name__ == "__main__":
    # 测试单个数据转换
    sample_data = {
        "filename": "Image_(1007).jpg",
        "text": "5月17日，天气，暴晒啊！草泥马！甘霖娘！热死了啊！！！！！！！",
        "label": 1,
        "span": "Detected Hateful spans: 草泥马, 甘霖娘; Detected general emotions spans: 暴晒, 热死了"
    }
    
    result = convert_data_format(sample_data)
    print("转换结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # 测试其他格式
    test_cases = [
        {
            "filename": "test1.jpg",
            "text": "今天心情很好",
            "label": 0,
            "span": "Detected general emotions spans: 心情很好"
        },
        {
            "filename": "test2.jpg",
            "text": "这个人真是个白痴",
            "label": 1,
            "span": "Detected Hateful spans: 白痴"
        },
        {
            "filename": "test3.jpg",
            "text": "普通的一天",
            "label": 0,
            "span": "No emotion spans detected"
        }
    ]
    
    print("更多测试用例:")
    for i, test_case in enumerate(test_cases, 1):
        result = convert_data_format(test_case)
        print(f"\n测试用例 {i}:")
        print(f"输入: {test_case['text']}")
        print(f"输出: {result['prompt']}")

    
    batch_convert('data/MET/data.json', 'data/MET/converted_data.json')