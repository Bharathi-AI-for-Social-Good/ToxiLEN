import json
import re
from pathlib import Path

# ---------------- 正则与分隔符 ----------------
HATE_PREFIX = "detected hateful spans:"
EMO_PREFIX  = "detected general emotions spans:"
SEP_RE      = re.compile(r"[;,、]\s*|\s+")       # 切分多 span 词

# ---------------- 提取子串 ----------------
def _section(txt: str, start_kw: str, other_kw: str) -> str:
    low = txt.lower()
    s   = low.find(start_kw)
    if s == -1:
        return ""
    s += len(start_kw)
    e   = low.find(other_kw, s) if low.find(other_kw, s) != -1 else len(txt)
    return txt[s:e].strip(" ;,")                 # 去前后分隔符

# ---------------- 解析 span ----------------
def parse_span(span_field: str):
    span_field = (span_field or "").strip()
    hate_raw   = _section(span_field, HATE_PREFIX, EMO_PREFIX)
    emo_raw    = _section(span_field, EMO_PREFIX,  HATE_PREFIX)

    hate_sp = [w for w in SEP_RE.split(hate_raw) if w]
    emo_sp  = [w for w in SEP_RE.split(emo_raw)  if w]

    if hate_sp and emo_sp:
        return "mixed", hate_sp, emo_sp
    if hate_sp:
        return "hate",  hate_sp, []
    if emo_sp:
        return "emo",   [],      emo_sp
    return "none", [], []

# ---------------- 生成 prompt ----------------
def make_prompt(text: str, span_field: str) -> str:
    typ, hate, emo = parse_span(span_field)

    if typ == "hate":
        return f"<HATE_SPAN> {', '.join(hate)} </HATE_SPAN> {text}"

    if typ == "emo":
        return f"<EMO_SPAN> {', '.join(emo)} </EMO_SPAN> {text}"

    if typ == "mixed":
        parts = [
            f"<HATE_SPAN> {', '.join(hate)} </HATE_SPAN>" if hate else "",
            f"<EMO_SPAN> {', '.join(emo)} </EMO_SPAN>"     if emo  else ""
        ]
        return " ".join(p for p in parts if p) + " " + text  # 先 hate 再 emo

    return f"<NO_SPAN> {text}"

# ---------------- 主清洗函数 ----------------
def build_prompt_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)                              # 期望整体是 list[dict]

    processed = []
    for item in data:
        item = dict(item)                               # 避免原地修改
        item["span"] = make_prompt(item.get("text", ""),
                                     item.get("span", ""))
        processed.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"✅  Saved {len(processed)} samples → {output_path}")

# ---------------- CLI（可选） ----------------
if __name__ == "__main__":
    in_path  = "data/toxi_mm/json/test.json"            # 修改为你的原始文件
    out_path = "data/toxi_mm/json/test_prompt.json"     # 生成含 prompt 的文件
    build_prompt_file(in_path, out_path)
