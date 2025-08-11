import openai
import time
from groq import Groq

client = openai.OpenAI(api_key="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")  # ← 替换为你自己的 key
groq_client = Groq(api_key="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") 


def call_gpt_api(messages, max_tokens=512, model="gpt-4o", temperature=0.8, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        
        except openai.RateLimitError as e:
            wait_time = 10 + attempt * 5  # Gradually increase wait time
            print(f"[RateLimit] Waiting {wait_time}s... ({attempt+1}/{max_retries})")
            time.sleep(wait_time)

        except Exception as e:
            print(f"[API Error] {e}")
            break  # If an error occurs, exit the loop and return None

    return None


def call_groq_api(
    messages,
    model="llama3-70b-8192",
    temperature=0.8,
    max_completion_tokens=512,
    top_p=1,
    stream=False,
    stop=None,
    max_retries=5,
):
    for attempt in range(max_retries):
        try:
            # 如果是 Qwen 系列模型，附加 reasoning_effort 参数
            extra_args = {}
            if "qwen" in model.lower():
                extra_args["reasoning_effort"] = "none"

            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                top_p=top_p,
                stream=stream,
                stop=stop,
                **extra_args           # 注入可选参数
            )

            if stream:
                result = ""
                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    print(delta, end="", flush=True)
                    result += delta
                print()  # 换行
                return result.strip()
            else:
                return response.choices[0].message.content.strip()

        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = 10 + attempt * 5
                print(f"[RateLimit] Waiting {wait_time}s... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"[API Error] {e}")
                break

    return None

