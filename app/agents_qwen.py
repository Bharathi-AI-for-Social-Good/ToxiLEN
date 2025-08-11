import json


from app.repo.prompt_loader import load_prompt_by_name
from app.api import call_gpt_api, call_groq_api
from app.utils.logger import get_logger
from app.utils.regex import extract_fields
from app.search import InContextSearcher

logger = get_logger(__name__)

searcher = InContextSearcher()

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

def agent_generator(context:dict, max_tokens=512):
    prompts = load_prompt_by_name("generator", variables=context)
    response = call_groq_api(prompts, model="qwen/qwen3-32b",max_tokens=max_tokens)
    results = {}
    if response:
        try:
            parsed = json.loads(response)
            results = {
                "filename": context.get("filename", "unknown"),
                "hate_speech": context.get("hate_speech", ""),
                "counters": parsed
            }
        except Exception as json_err:
            logger.info("Returned content was:")
            logger.info(repr(response))
    else:
        logger.info(f"[!] No response for: {context.get('hate_speech', 'unknown')}")
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
    hate_speech = context['hate_speech']
    counter_sets = context['counters']
    evaluations = {}

    for key, value in counter_sets.items():
        prompt = load_prompt_by_name("evaluator", {
            "hate_speech": hate_speech,
            "counter_narrative": value
        })
        response = call_gpt_api(
            prompt, 
            max_tokens=max_tokens,
            model="gpt-4o", 
            temperature=0.5
        )
        if not response:
            evaluations[key] = _default_evaluation("API call failed")
            continue
        try:
            evaluations[key] = json.loads(response)
        except:
            evaluations[key] = _default_evaluation("Parsing failed")

    return {
        "filename": context.get('filename', 'unknown'),
        "hate_speech": hate_speech,
        "counters": counter_sets,
        "evaluations": evaluations
    }
    
    
def agent_filter(context):
    """
    Return the counter-narrative with the highest weighted-average evaluation score.
    Compatible with either a single dict or a list of dicts produced by `evaluators`.
    """
    if not context:
        return None
    if not isinstance(context, list):
        context = [context]

    top_counter, max_score = None, -1

    for item in context:
        hate_speech = item.get("hate_speech", "")
        eval_dict   = item.get("evaluations", {})   # ← 评估结果（五维指标）
        cn_texts    = item.get("counters", {})      # ← 原始反叙事文本

        for ctype, metrics in eval_dict.items():
            spec = metrics.get("Specificity", {}).get("score", 0)
            opp  = metrics.get("Opposition",  {}).get("score", 0)
            rel  = metrics.get("Relatedness", {}).get("score", 0)
            tox  = metrics.get("Toxicity",    {}).get("score", 0)
            flu  = metrics.get("Fluency",     {}).get("score", 0)

            weighted = [
                spec * 1.1,
                opp  * 1.0,
                rel  * 1.5,
                tox  * 0.8,
                flu  * 0.8
            ]
            avg_score = sum(weighted) / len(weighted)

            if avg_score > max_score:
                max_score = avg_score
                top_counter = {
                    "filename": item.get("filename", "unknown"),
                    "text": hate_speech,
                    "counter_type": ctype,
                    "counter_text": cn_texts.get(ctype, ""),
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

