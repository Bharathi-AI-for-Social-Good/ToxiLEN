import json
import torch
import re
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

from models.main_model import MemeMultimodalDetector
from app.repo.prompt_loader import load_prompt_by_name
from app.api import call_gpt_api, call_groq_api
from app.utils.logger import get_logger
from app.utils.regex import extract_fields
from app.search import InContextSearcher
from app.trainer.model_factory import build_model
from app.trainer.predict import single_predict

logger = get_logger(__name__)

searcher = InContextSearcher()

model = build_model("default")
model.load_state_dict(torch.load("toxilen.pth", map_location=device))

processor =  BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
image_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def clean_json_response(response: str) -> str:
    """
    清理 GPT 输出中的 Markdown 代码块包裹
    """
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        return match.group(1)
    return response.strip()



def agent_knowledge_extraction(context):
    logger.info("Extracting knowledge from context")
    prompt = load_prompt_by_name("knowledge_extractor", variables=context)
    response = call_gpt_api(prompt, max_tokens=512)

    return {"explanation": response}


def agent_span_extraction(context):
    logger.info("Extracting spans from context")
    prompt = load_prompt_by_name("span_extractor", variables=context)
    response = call_gpt_api(prompt, max_tokens=512)

    if isinstance(response, str):
        try:
            response = clean_json_response(response)
            response = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Illegal JSON: {response}") from e

    data = response.get("item", response)

    hateful_span = data.get('misogynistic', [])
    general_emo = data.get('general_emotion', [])

    spans = ""
    if hateful_span:
        spans += f"<HATE_SPAN> {' | '.join(hateful_span)} </HATE_SPAN>"
    if general_emo:
        spans += f"<EMO_SPAN> {' | '.join(general_emo)} </EMO_SPAN>"
    if not hateful_span and not general_emo:
        spans += "<NO_SPAN>"

    spans += context['text_inputs']

    return { "prompt": spans }


def agent_caption_generation(context):
    logger.info("Generating caption for image")
    image_path = context.get("image_path")
    if not image_path:
        raise ValueError("Image path is required")

    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image,return_tensors='pt').to(device)
        out = image_model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"生成字幕时出错 {image_path}: {e}")
        context["caption"] = "Error generating caption"
        return context

    return {"caption": caption}


def agent_prediction(context):
    logger.info("Running prediction on Memes")
    image = Image.open(context["image_path"]).convert("RGB")

    context['image'] = image
    pred, log_probs = single_predict(model,context,device=device)

    context["prediction"] = pred
    context["log_probs"] = log_probs
    return context

def agent_sample(context):
    query = context.get("query", "Sample query")
    ids, examples = searcher.search(query)

    prompts = ""
    for i, raw in enumerate(examples):
        fields = extract_fields(raw)
        if not fields or not fields["pairs"]:
            continue  # 无法解析或没有任何配对时跳过

        prompts += f"Example {i+1}:\n[HS] {fields['HS']}\n"

        # 遍历所有 TYPE-CN 配对
        for pair in fields["pairs"]:
            prompts += f"[TYPE] {pair['TYPE']}\n[CN] {pair['CN']}\n"

        prompts += "\n"  # 每个样本之间空一行

    return prompts




def agent_generator(context: dict, max_tokens=512):
    prompt = load_prompt_by_name("generator", variables=context)
    response = call_gpt_api(prompt, max_tokens=max_tokens)

    results = {}

    if response and response.strip():
        try:
            cleaned = clean_json_response(response)
            parsed = json.loads(cleaned)

            candidates = parsed.get("candidates", [])
            if not isinstance(candidates, list):
                raise ValueError("`candidates` must be a list")

            results = {
                "filename": context.get("filename", "unknown"),
                "hate_speech": context.get("hate_speech", ""),
                "background": context.get("background", ""),
                "counters": candidates
            }

        except Exception as err:
            logger.error("[agent_generator] Failed to parse JSON")
            logger.error("Raw response: %s", repr(response))
            raise err

    else:
        logger.warning("[agent_generator] Empty response for: %s", context.get("hate_speech", "unknown"))

    return results


def _default_evaluation(reason=""):
    return {
        "Specificity": {"score": 0, "explanation": reason},
        "Opposition": {"score": 0, "explanation": reason},
        "Relatedness": {"score": 0, "explanation": reason},
        "Toxicity": {"score": 0, "explanation": reason},
        "Fluency": {"score": 0, "explanation": reason}
    }


def evaluators(context, max_tokens=512):
    """
    对 counters(List[str]) 中的每一句 counter-narrative 调用 evaluator-prompt 打分。
    返回值中的 evaluations 仍保持 {c1: ..., c2: ...} 的 dict 结构，便于后续筛选。
    """
    hate_speech = context["hate_speech"]
    counters = context["counters"]

    if not isinstance(counters, list):
        raise ValueError("Expected `counters` to be a list of counter-narratives (str).")

    evaluations = {}
    for i, counter in enumerate(counters):
        key = f"c{i+1}"                             # c1, c2, ...
        prompt = load_prompt_by_name(
            "evaluator",
            {"hate_speech": hate_speech, "counter_narrative": counter, "background": context['background']}
        )
        response = call_gpt_api(
            prompt,
            max_tokens=max_tokens,
            model="gpt-4o",
            temperature=0.3
        )
        if not response:
            evaluations[key] = _default_evaluation("API call failed")
            continue
        
        try:
            evaluations[key] = json.loads(response)
        except Exception:
            evaluations[key] = _default_evaluation("Parsing failed")

    return {
        "filename": context.get("filename", "unknown"),
        "hate_speech": hate_speech,
        "counters": counters,           # List[str]
        "evaluations": evaluations      # Dict[cX → scores]
    }


def agent_filter(results):
    """
    接收 evaluators 输出（单个 dict 或其列表），
    返回平均加权得分最高的 counter-narrative 记录。
    """
    if not results:
        return None
    # 统一为 list 迭代
    if not isinstance(results, list):
        results = [results]

    top_counter, max_score = None, -1

    for item in results:
        hate_speech = item.get("hate_speech", "")
        cn_texts    = item.get("counters", [])           # List[str]
        eval_dict   = item.get("evaluations", {})        # Dict[cX → metrics]

        for ctype, metrics in eval_dict.items():
            # 提取各维度得分
            spec = metrics.get("Specificity", {}).get("score", 0)
            opp  = metrics.get("Opposition",  {}).get("score", 0)
            rel  = metrics.get("Relatedness", {}).get("score", 0)
            tox  = metrics.get("Toxicity",    {}).get("score", 0)
            flu  = metrics.get("Fluency",     {}).get("score", 0)

            # 加权平均
            weighted = [
                spec * 1.122,
                opp  * 1.072,
                rel  * 1.142,
                tox  * 1.000,
                flu  * 1.005
            ]
            avg_score = sum(weighted) / len(weighted)

            # 取出对应的 counter-narrative 文本
            idx = int(ctype[1:]) - 1            # 把 c1 → 0, c2 → 1 ...
            text = cn_texts[idx] if 0 <= idx < len(cn_texts) else ""

            # 更新最佳
            if avg_score > max_score:
                max_score = avg_score
                top_counter = {
                    "filename": item.get("filename", "unknown"),
                    "hate_speech": hate_speech,
                    "counter_type": ctype,
                    "counter_text": text,
                    "evaluation": metrics,
                    "average_score": avg_score
                }

    return top_counter




def default_gen_agent(context: dict, model_name="qwen/qwen3-32b", prompt_type="zero") -> str | None:
    """
    Default agent that generates a counter-narrative based on the hate speech text.
    
    Args:
        context (dict): Dictionary with keys: filename, hate_speech, background, samples.
        model_name (str): Groq model name to use for generation.
        prompt_type (str): Prompt variant, such as 'zero', 'few', etc.
    
    Returns:
        str | None: Generated counter-narrative, or None on failure.
    """
    if not context or not isinstance(context, dict):
        return None

    context = {
        "filename": context.get("filename", "unknown"),
        "hate_speech": context.get("hate_speech", ""),
        "background": context.get("background", ""),
        "samples": context.get("samples", ""),
    }

    messages = load_prompt_by_name(prompt_type, variables=context)
    response = call_groq_api(messages, model=model_name)
    return response


def default_gpt_gen_agent(context: dict, model_name="gpt-4o", prompt_type="zero") -> str | None:
    """
    Default agent that generates a counter-narrative based on the hate speech text.
    
    Args:
        context (dict): Dictionary with keys: filename, hate_speech, background, samples.
        model_name (str): Groq model name to use for generation.
        prompt_type (str): Prompt variant, such as 'zero', 'few', etc.
    
    Returns:
        str | None: Generated counter-narrative, or None on failure.
    """
    if not context or not isinstance(context, dict):
        return None

    context = {
        "filename": context.get("filename", "unknown"),
        "hate_speech": context.get("hate_speech", ""),
        "background": context.get("background", ""),
        "samples": context.get("samples", ""),
    }

    messages = load_prompt_by_name(prompt_type, variables=context)
    response = call_gpt_api(messages, model=model_name)
    return response


def default_evaluators(context, max_tokens=512):
    hate_speech = context["hate_speech"]
    counter = re.sub(r"<think>.*?</think>", "", context["response"], flags=re.DOTALL).strip()

    
    prompt = load_prompt_by_name(
        "evaluator",
        {"hate_speech": hate_speech, "counter_narrative": counter, "background": context.get("background", "")}
    )
    response = call_gpt_api(
        prompt,
        max_tokens=max_tokens,
        model="gpt-4o",
        temperature=0.3
    )

    try:
        json_response = json.loads(response) if isinstance(response, str) else response
    except json.JSONDecodeError as e:
        print("[JSON ERROR]", e)
        print("[RAW OUTPUT]", response)
        json_response = {
            "Specificity": {"score": 0, "explanation": "Invalid JSON response."},
            "Opposition":  {"score": 0, "explanation": "Invalid JSON response."},
            "Relatedness": {"score": 0, "explanation": "Invalid JSON response."},
            "Toxicity":    {"score": 0, "explanation": "Invalid JSON response."},
            "Fluency":     {"score": 0, "explanation": "Invalid JSON response."}
        }

    return {
        "filename": context.get("filename", "unknown"),
        "hate_speech": hate_speech,
        "counter_speech": counter,
        "evaluations": json_response
    }
