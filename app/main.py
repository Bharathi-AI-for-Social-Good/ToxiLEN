from app.pipeline import predict_pipeline, gen_pipeline


def main(image, text = ""):
    
    context = {
        "image_path": image,
        "text_inputs": text
    }

    results, combined_context = predict_pipeline(context)
    
    result = results["prediction"]
    if result == 1:
        context = {
            "filename": image,
            "hate_speech": text,
            "background": combined_context['explanation'],
        }

        top_c = gen_pipeline(context)
    else:
        top_c = "This is a non-hate speech comment."

    return top_c, results


