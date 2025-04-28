# model_config.py
from sentence_transformers import SentenceTransformer
import requests
import os
import json
import re
from dotenv import load_dotenv
import openai, os, json, re

load_dotenv()

# Load embedding model
embedding_model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L12-cos-v5')

# Hugging Face API config
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
# API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen1.5-1.8B-Chat"
HF_TOKEN = os.getenv("HF_TOKEN", "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")


openai.api_key = os.getenv("OPENAI_API_KEY")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def text_to_embedding(text):
    return embedding_model.encode([text])[0]

def generate_answer_from_gpt(query, top_k_docs=None, model="gpt-4o-mini"):
    """Return a ReAct-style JSON answer without using retrieved context."""
    system_prompt = (
        "You are a professional AI assistant specializing in finance. "
        "Answer the question using Reason and Act (ReAct) and output ONLY the JSON structure below."
    )

    user_prompt = f"""
Respond in JSON ONLY:
{{
  "question": "<restate the question>",
  "reasoning_steps": [
    {{
      "question": "<thought process>",
      "act": "<action taken>",
      "observe": "<observation>",
      "answer": "<partial answer>"
    }}
  ],
  "final_answer": "<summarized answer>"
}}
---
Question:
{query}
"""

    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=1024,
    )

    raw = resp.choices[0].message.content.strip()
    print("Raw response:", raw)
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"error": "Invalid JSON generated", "final_answer": raw}

def generate_answer_from_mistral(query, top_k_docs, max_new_tokens=8192):

    prompt = f"""You are a professional AI assistant specializing in finance. Use the following retrieved context to answer the user's question.

    Respond strictly in the following JSON format:
    {{
    "question": "<restate the question>",
    "reasoning_steps": [
        {{
        "question": "<thought process>",
        "act": "<action taken>",
        "observe": "<observation after action>",
        "answer": "<intermediate or partial answer if any>"
        }},
        ...
    ],
    "final_answer": "<summarized answer>"
    }}
    ---

x

    Question:
    {query}

    Answer:
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    generated = result[0]["generated_text"].replace("### END", "").strip()

    
    match = re.search(r"\{[\s\S]*\}", generated)

    if match:
        try:
            parsed_json = json.loads(match.group(0))
            return parsed_json  
        except json.JSONDecodeError:
            print("[ERROR] JSON decoding failed.")
            print(generated)
            return {"error": "Invalid JSON generated", "final_answer": generated}
    else:
        print("[ERROR] No JSON found.")
        print(generated)
        return {"error": "No valid JSON object detected", "final_answer": generated}
