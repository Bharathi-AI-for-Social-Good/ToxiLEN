import pandas as pd
import json
from tqdm import tqdm


from app.pipeline import gen_pipeline,test_pipeline

data = pd.read_csv("data/counter_narrative/data/test.csv", encoding="utf-8").to_dict(orient="records")

outputs = []
for row in tqdm(data, desc="Generating responses"):
# Prepare context for testing
    context = {
        'filename': row["filename"],
        "hate_speech": row["text"],
        "background": row["background"],
    }
    
    # Generate response using the pipeline
    res = gen_pipeline(context)
    # res = test_pipeline(context, model="llama-3.3-70b-versatile", type="knowledge")
    results= {
        "filename": context['filename'],
        "hate_speech": context['hate_speech'],
        "response": res
    }
    outputs.append(results)


with open('data/counter_narrative/results/in_context_without_llm.json', 'w', encoding='utf-8') as f:
    json.dump(outputs, f, ensure_ascii=False, indent=4)
    
    
    