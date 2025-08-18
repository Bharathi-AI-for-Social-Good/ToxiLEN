import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from app.main import main

def toxilen_interface(image, text):
    top_c, results = main(image, text)

    # Counter outputs
    counter_text = top_c["counter_text"]
    evaluation = top_c["evaluation"]
    avg_score = top_c["average_score"]
    counter_type = top_c["counter_type"]

    # Prediction outputs
    caption = results.get("caption", "")
    explanation = results.get("explanation", "")
    prompt = results.get("prompt", "")
    prediction = results.get("prediction", "")
    log_probs = results.get("log_probs")
    log_probs_str = str(log_probs.tolist()) if log_probs is not None else ""

    # Score table as DataFrame
    counter_metadata_df = pd.DataFrame([
        {"Aspect": k, "Score": v["score"]} for k, v in evaluation.items()
    ])
    counter_metadata_df.loc[len(counter_metadata_df)] = {
        "Aspect": "Counter Type",
        "Score": counter_type
    }
    counter_metadata_df.loc[len(counter_metadata_df)] = {
        "Aspect": "Average Score",
        "Score": round(avg_score, 3)
    }

    # Explanation per metric
    evaluation_explanations = {
        k: v["explanation"] for k, v in evaluation.items()
    }

    # Score plot
    fig, ax = plt.subplots()
    aspects = list(evaluation.keys())
    scores = [evaluation[a]["score"] for a in aspects]
    ax.bar(aspects, scores, color='skyblue')
    ax.set_ylim([0, 5])
    ax.set_ylabel("Score")
    ax.set_title("Counter-Narrative Evaluation")
    plt.xticks(rotation=30)
    plt.tight_layout()

    return (
        counter_text,
        f"\n### Prediction: {'🟥 Misogynistic' if prediction == 1 else '🟩 Non-Misogynistic'}",
        counter_metadata_df,
        evaluation_explanations,
        fig,
        caption,
        explanation,
        prompt,
        log_probs_str
    )


demo = gr.Interface(
    fn=toxilen_interface,
    inputs=[
        gr.Image(type="filepath", label="🖼️ Meme Image"),
        gr.Textbox(lines=2, placeholder="Enter meme text here...", label="📝 Meme Text")
    ],
    outputs=[
        gr.Textbox(label="💡 Top Counter-Narrative"),
        gr.Markdown(label="🎯 Prediction Result"),
        gr.Dataframe(label="📊 Counter Quality Scores", type="pandas"),
        gr.JSON(label="🧠 Metric Explanations"),
        gr.Plot(label="📈 Evaluation Chart"),
        gr.Textbox(label="🖋️ Generated Caption"),
        gr.Textbox(label="🔍 Explanation for Prediction"),
        gr.Textbox(label="📎 Prompt with Span"),
        gr.Textbox(label="📉 Log-Probabilities")
    ],
    title="🔥 ToxiLEN: Meme Misogyny Detector & Counter Narrative Generator",
    theme=gr.themes.Soft(),
    description="""
    <div style='font-size: 16px;'>
        🚀 Upload a meme image and provide its text. <br>
        🧠 ToxiLEN will detect misogynistic content, explain the reasoning, and generate a counter-narrative.
    </div>
    """,
    allow_flagging="never",
    live=False,
)

if __name__ == "__main__":
    demo.launch()
