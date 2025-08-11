import json
import pandas as pd

from app.agents import agent_sample, agent_generator, evaluators, agent_filter, default_gen_agent, default_gpt_gen_agent, default_evaluators
from app.utils.logger import get_logger

def gen_pipeline(context):
    info = f"[HS]{context['hate_speech']}⟪SEP⟫ [KNOWLEDGE]{context['background']}"
    samples = agent_sample({"query": info})   
    session_context = {
        "filename": context.get("filename", "unknown"),
        "hate_speech": context["hate_speech"],
        "background": context["background"],
        "samples": samples
    }
    results = agent_generator(session_context)    
    evaluation = evaluators(results) 
    top_c = agent_filter(evaluation)
    return top_c
    

def test_pipeline(context, model, type):
    response = default_gen_agent(context, model_name=model, prompt_type=type)
    return response


def gpt_test_pipeline(context, model="gpt-4o", type="knowledge"):
    response = default_gpt_gen_agent(context, model_name=model, prompt_type=type)
    return response

def evaluate_pipeline(context, model="gpt-4o"):
    evaluation = default_evaluators(context)
    return evaluation