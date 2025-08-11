import re

SEP = "<<<SEP>>>"
SEP_PATTERN = re.escape(SEP)
TAG_PATTERN = r'\[([A-Z]+)\]'

def extract_fields(text: str):
    """
    支持格式：
      · 只有一组 TYPE+CN
      · 多组 TYPE+CN
    返回字段:
      HS, KNOWLEDGE, pairs(List[Dict])
    """
    # 1. 先按分隔符切块并去掉空白
    blocks = [b.strip() for b in re.split(SEP_PATTERN, text) if b.strip()]

    result = {"HS": None, "KNOWLEDGE": None, "pairs": []}
    pending_type = None   # 暂存最近遇到的 TYPE

    for blk in blocks:
        m = re.match(TAG_PATTERN, blk)
        if not m:
            continue
        tag = m.group(1)
        content = blk[m.end():].strip()

        if tag == "HS":
            result["HS"] = content
        elif tag == "KNOWLEDGE":
            result["KNOWLEDGE"] = content
        elif tag == "TYPE":
            pending_type = content              # 等待下一个 CN
        elif tag == "CN":
            # 有 TYPE 才配对；没有就略过
            if pending_type:
                result["pairs"].append({"TYPE": pending_type, "CN": content})
                pending_type = None

    # 如果文本只含单个 TYPE 而缺 CN，可在此处追加异常处理

    return result
