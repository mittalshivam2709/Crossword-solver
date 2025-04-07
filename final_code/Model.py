import openai 
from openai import OpenAI

# openai.api_key = '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'
import os
os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI()  # Make sure your API key is set in env or passed to OpenAI()

with open('../3.5_fine_tuning/fine_tuned_model_name.txt', 'r') as f:
    fine_tuned_model = f.read().strip()


def generate_unique_completions(prompt, model, num_completions=5, max_tokens=50):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        n=num_completions,
        stop=None,
        temperature=0.9,
        top_p=0.9
    )
    
    completions = set()
    for choice in response.choices:
        completions.add(choice.message.content.strip())
        if len(completions) >= num_completions:
            break

    return list(completions)[:num_completions]
