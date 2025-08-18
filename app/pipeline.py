import json
import pandas as pd

from app.agents import (agent_sample, 
                        agent_generator, 
                        evaluators, 
                        agent_filter, 
                        default_gen_agent, 
                        default_gpt_gen_agent, 
                        default_evaluators,
                        agent_caption_generation,
                        agent_knowledge_extraction,
                        agent_prediction,
                        agent_span_extraction)
from app.utils.logger import get_logger

def predict_pipeline(context):
    caption_context = agent_caption_generation(context)
    knowledge_context = agent_knowledge_extraction(context)
    span_context = agent_span_extraction(context)
    
    combined_context = {
        **context,
        **caption_context,
        **knowledge_context,
        **span_context
    }
    
    results = agent_prediction(combined_context)
    
    return results,combined_context

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