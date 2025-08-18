from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from rouge_score import rouge_scorer


import jieba



def get_bleu_n(references, predictions):
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []

    smooth_fn = SmoothingFunction().method1

    for ref, pred in zip(references, predictions):
        ref_tokens = list(jieba.cut(ref))
        pred_tokens = list(jieba.cut(pred))

        bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
        bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
        bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
        bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)

        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu3_scores.append(bleu3)
        bleu4_scores.append(bleu4)

    # 返回平均值
    return {
        "bleu1": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
        "bleu2": sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0.0,
        "bleu3": sum(bleu3_scores) / len(bleu3_scores) if bleu3_scores else 0.0,
        "bleu4": sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0.0,
    }
    
def get_bert_score(predictions, references, lang="zh"):
    # 计算 BERTScore
    P, R, F1 = score(predictions, references, lang=lang, verbose=True)

    # 平均值
    avg_p = P.mean().item()
    avg_r = R.mean().item()
    avg_f1 = F1.mean().item()

    return {
        "Precision": avg_p,
        "Recall": avg_r,
        "F1": avg_f1
    }
    
def get_pairwise_jaccard_novelty(texts1, texts2):
    """
    成对计算 Jaccard 相似度与新奇性：
    - 每对 texts1[i] 与 texts2[i] 比较
    - 返回平均 Jaccard 相似度与新奇性
    """
    assert len(texts1) == len(texts2), "输入列表长度必须一致"

    def jaccard_sim(set1, set2):
        return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0

    all_scores = []
    for t1, t2 in zip(texts1, texts2):
        tokens1 = set(jieba.lcut(t1))
        tokens2 = set(jieba.lcut(t2))
        score = jaccard_sim(tokens1, tokens2)
        all_scores.append(score)

    avg_sim = sum(all_scores) / len(all_scores)
    novelty = 1 - avg_sim
    return avg_sim, novelty



def get_rouge_l(predictions, references):
    """
    Pairwise 计算 ROUGE-L 分数（Precision, Recall, F1）
    :param predictions: List[str]，系统生成的句子
    :param references:  List[str]，人工参考句子
    :return: dict 包含平均 Precision、Recall、F1
    """
    assert len(predictions) == len(references), "预测和参考数量不一致"

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    p_list, r_list, f1_list = [], [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)["rougeL"]
        p_list.append(scores.precision)
        r_list.append(scores.recall)
        f1_list.append(scores.fmeasure)

    avg_p = sum(p_list) / len(p_list)
    avg_r = sum(r_list) / len(r_list)
    avg_f1 = sum(f1_list) / len(f1_list)

    return {
        "Precision": avg_p,
        "Recall": avg_r,
        "F1": avg_f1
    }


from collections import Counter
import jieba
import re

def calc_rr(predictions, n=2, lang="zh"):
    """
    计算预测句子列表的 Repetition Rate (RR)
    
    参数:
        predictions: list[str]  预测生成的句子列表
        n: int                  n-gram 大小，默认 2
        lang: str               'zh' 中文（用jieba分词）或 'en' 英文（用简单分词）
        
    返回:
        avg_rr: float           平均 RR
        rr_list: list[float]    每条句子的 RR
    """
    def repetition_rate(text):
        # 分词
        if lang == "zh":
            tokens = list(jieba.cut(text))
        else:
            tokens = re.findall(r"\w+", text.lower())

        if len(tokens) < n:
            return 0.0

        # 提取 n-grams
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        counts = Counter(ngrams)
        total = len(ngrams)
        repeated = sum(count for count in counts.values() if count > 1)

        return repeated / total

    rr_list = [repetition_rate(sent) for sent in predictions]
    avg_rr = sum(rr_list) / len(rr_list) if rr_list else 0.0
    return avg_rr, rr_list