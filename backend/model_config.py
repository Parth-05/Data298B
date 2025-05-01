# model_config.py
from sentence_transformers import SentenceTransformer
import requests
import os
import json
import re
from dotenv import load_dotenv
import openai, os, json, re
import google.generativeai as genai 

load_dotenv()

# Load embedding model
embedding_model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L12-cos-v5')

# Hugging Face API config
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
# API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen1.5-1.8B-Chat"
HF_TOKEN = os.getenv("HF_TOKEN", "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")


openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def text_to_embedding(text):
    return embedding_model.encode([text])[0]

# Answer in Markdown format
def _gpt_markdown_answer(query: str, model="gpt-4o-mini") -> str:
    # system = ("You are a professional AI assistant specialising in finance. "
    #           "Answer the question using Reason and Act (ReAct) and answer in Markdown using:\n"
    #           "### Reasoning\n• Thought / Act / Observe lines\n"
    #           "### Final answer\n")
    system = (
    "You are a professional AI assistant specializing in finance. "
    "Think step by step using the Reason and Act (ReAct) pattern and reply **in Markdown only**.\n\n"
    "Use exactly this template:\n"
    "### Question\n"
    "<restate the question>\n\n"
    "### Reasoning\n"
    "1. **Thought:** …  \n"
    "   **Act:** …  \n"
    "   **Observe:** …  \n"
    "   **Answer:** …\n"
    "2. *(repeat Thought/Act/Observe/Answer as many steps as needed)*\n\n"
    "### Final answer\n"
    "<concise answer>"
)
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": query}],
        temperature=0.7,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()        # markdown string

# Convert markdown to JSON
def _markdown_to_json(md_text: str, fixer_model="gpt-3.5-turbo") -> dict:
    resp = openai.chat.completions.create(
        model=fixer_model,
        messages=[
            {"role": "system",
             "content": ("Convert the following Reason and Act (ReAct) markdown response into a valid JSON "
                         "object with keys: question, reasoning_steps, final_answer. "
                         "Do NOT change the content; return only JSON.")},
            {"role": "user", "content": md_text},
        ],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# Generate answer from GPT model using JSON format
def generate_answer_from_gpt(query: str, retrieved_docs=None, model="gpt-4o-mini"):
    """
    • retrieved_docs is ignored here (but kept for interface parity).
    • Returns a dict (JSON) if conversion succeeds, otherwise the raw markdown string.
    """
    markdown = _gpt_markdown_answer(query, model=model)

    # try fast regex-based extraction first  (cheap)
    m = re.search(r"\{[\s\S]*\}", markdown)
    print("Regex match:", m)
    if m:
        try:
            return json.loads(m.group(0))          # success → dict
        except json.JSONDecodeError:
            pass                                   # fall through to fixer

    # fallback: fixer second-call (cheap)
    try:
        return _markdown_to_json(markdown)
    except Exception:
        return markdown   

# # Generate answer from Mistral model using JSON format
# def generate_answer_from_mistral(query, top_k_docs, max_new_tokens=8192):

#     prompt = f"""You are a professional AI assistant specializing in finance. Use the following retrieved context to answer the user's question.

#     Respond strictly in the following JSON format:
#     {{
#     "question": "<restate the question>",
#     "reasoning_steps": [
#         {{
#         "question": "<thought process>",
#         "act": "<action taken>",
#         "observe": "<observation after action>",
#         "answer": "<intermediate or partial answer if any>"
#         }},
#         ...
#     ],
#     "final_answer": "<summarized answer>"
#     }}
#     ---

# x

#     Question:
#     {query}

#     Answer:
#     """

#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "max_new_tokens": max_new_tokens,
#             "temperature": 0.7,
#             "do_sample": True,
#             "return_full_text": False
#         }
#     }

#     response = requests.post(API_URL, headers=headers, json=payload)
#     response.raise_for_status()
#     result = response.json()

#     generated = result[0]["generated_text"].replace("### END", "").strip()

    
#     match = re.search(r"\{[\s\S]*\}", generated)

#     if match:
#         try:
#             parsed_json = json.loads(match.group(0))
#             return parsed_json  
#         except json.JSONDecodeError:
#             print("[ERROR] JSON decoding failed.")
#             print(generated)
#             return {"error": "Invalid JSON generated", "final_answer": generated}
#     else:
#         print("[ERROR] No JSON found.")
#         print(generated)
#         return {"error": "No valid JSON object detected", "final_answer": generated}

# Answer in Markdown format
def _mistral_markdown_answer(query: str, model=None) -> str:
    if model is None:
        model = os.getenv("MISTRAL_MODEL")
    system = (
    "You are a professional AI assistant specializing in finance. "
    "Think step by step using the Reason and Act (ReAct) pattern and reply **in Markdown only**.\n\n"
    "Use exactly this template:\n"
    "### Question\n"
    "<restate the question>\n\n"
    "### Reasoning\n"
    "1. **Thought:** …  \n"
    "   **Act:** …  \n"
    "   **Observe:** …  \n"
    "   **Answer:** …\n"
    "2. *(repeat Thought/Act/Observe/Answer as many steps as needed)*\n\n"
    "### Final answer\n"
    "<concise answer>"
)
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": query}],
        temperature=0.7,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()        # markdown string

# Convert markdown to JSON
def _mistral_markdown_to_json(md_text: str, fixer_model=None) -> dict:
    if model is None:
        model = os.getenv("MISTRAL_MODEL_FIXER")
    resp = openai.chat.completions.create(
        model=fixer_model,
        messages=[
            {"role": "system",
             "content": ("Convert the following Reason and Act (ReAct) markdown response into a valid JSON "
                         "object with keys: question, reasoning_steps, final_answer. "
                         "Do NOT change the content; return only JSON.")},
            {"role": "user", "content": md_text},
        ],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# Generate answer from Mistral model using JSON format
def generate_answer_from_mistral(query: str, retrieved_docs=None, model=None):
    """
    • retrieved_docs is ignored here (but kept for interface parity).
    • Returns a dict (JSON) if conversion succeeds, otherwise the raw markdown string.
    """

    if model is None:
        model = os.getenv("MISTRAL_MODEL")
    print(model)

    markdown = _mistral_markdown_answer(query, model=model)

    # try fast regex-based extraction first  (cheap)
    m = re.search(r"\{[\s\S]*\}", markdown)
    print("Regex match:", m)
    if m:
        try:
            return json.loads(m.group(0))          # success → dict
        except json.JSONDecodeError:
            pass                                   # fall through to fixer

    # fallback: fixer second-call (cheap)
    try:
        return _mistral_markdown_to_json(markdown)
    except Exception:
        return markdown    

# ─── Gemini: markdown answer ───────────────────────────────────────────────
def _gemini_markdown_answer(query: str, model="gemini-1.5-flash") -> str:
    prompt = (
        "You are a professional AI assistant specializing in finance.\n"
        "Think step-by-step using the Reason and Act (ReAct) pattern and reply **in Markdown only**.\n\n"
        "Use exactly this template:\n"
        "### Question\n"
        "<restate the question>\n\n"
        "### Reasoning\n"
        "1. **Thought:** …  \n"
        "   **Act:** …  \n"
        "   **Observe:** …  \n"
        "   **Answer:** …\n"
        "2. *(repeat Thought/Act/Observe/Answer as many steps as needed)*\n\n"
        "### Final answer\n"
        "<concise answer>"
    )
    model = genai.GenerativeModel(model)
    resp  = model.generate_content(
        [{"role": "user", "parts": [prompt + "\n\nQuestion:\n" + query]}]
    )
    print("Gemini response:", resp.candidates[0].content.parts[0].text.strip())
    return resp.candidates[0].content.parts[0].text.strip()

# ─── Public helper ---------------------------------------------------------
def generate_answer_from_gemini(query: str, retrieved_docs,
                                model="gemini-1.5-flash"):
    md = _gemini_markdown_answer(query, model=model)
    try:
        # quick regex path
        m = re.search(r"\{[\s\S]*\}", md)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    # fallback to fixer (same OpenAI-based fixer)
    try:
        return _markdown_to_json(md)
    except Exception:
        return md

# ─── Llama markdown answer ───────────────────────────────────────────────
def _llama_markdown_answer(query: str, model=None) -> str:
    if model is None:
        model = os.getenv("LLAMA_MODEL")
    prompt = (
        "You are a professional AI assistant specializing in finance.\n"
        "Think step-by-step using the Reason and Act (ReAct) pattern and reply **in Markdown only**.\n\n"
        "Use exactly this template:\n"
        "### Question\n"
        "<restate the question>\n\n"
        "### Reasoning\n"
        "1. **Thought:** …  \n"
        "   **Act:** …  \n"
        "   **Observe:** …  \n"
        "   **Answer:** …\n"
        "2. *(repeat Thought/Act/Observe/Answer as many steps as needed)*\n\n"
        "### Final answer\n"
        "<concise answer>"
    )
    model = genai.GenerativeModel(model)
    resp  = model.generate_content(
        [{"role": "user", "parts": [prompt + "\n\nQuestion:\n" + query]}]
    )
    print("Llama response:", resp.candidates[0].content.parts[0].text.strip())
    return resp.candidates[0].content.parts[0].text.strip()

# ─── Public helper ---------------------------------------------------------
def generate_answer_from_llama(query: str, retrieved_docs,
                                model=None):
    
    if model is None:
        model = os.getenv("LLAMA_MODEL")
    print(model)
    md = _llama_markdown_answer(query, model=model)
    try:
        # quick regex path
        m = re.search(r"\{[\s\S]*\}", md)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    # fallback to fixer (same OpenAI-based fixer)
    try:
        return _markdown_to_json(md)
    except Exception:
        return md
    
# Qwern Markdown answer
def _qwen_markdown_answer(query: str, model=None) -> str:
    if model is None:
        model = os.getenv("QWEN_MODEL")
    system = (
    "You are a professional AI assistant specializing in finance. "
    "Think step by step using the Reason and Act (ReAct) pattern and reply **in Markdown only**.\n\n"
    "Use exactly this template:\n"
    "### Question\n"
    "<restate the question>\n\n"
    "### Reasoning\n"
    "1. **Thought:** …  \n"
    "   **Act:** …  \n"
    "   **Observe:** …  \n"
    "   **Answer:** …\n"
    "2. *(repeat Thought/Act/Observe/Answer as many steps as needed)*\n\n"
    "### Final answer\n"
    "<concise answer>"
)
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": query}],
        temperature=0.7,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()        # markdown string

# Convert markdown to JSON
def _qwen_markdown_to_json(md_text: str, fixer_model=None) -> dict:
    if model is None:
        model = os.getenv("QWEN_MODEL_FIXER")
    resp = openai.chat.completions.create(
        model=fixer_model,
        messages=[
            {"role": "system",
             "content": ("Convert the following Reason and Act (ReAct) markdown response into a valid JSON "
                         "object with keys: question, reasoning_steps, final_answer. "
                         "Do NOT change the content; return only JSON.")},
            {"role": "user", "content": md_text},
        ],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

# Generate answer from Qwen model using JSON format
def generate_answer_from_qwen(query: str, retrieved_docs=None, model=None):
    """
    • retrieved_docs is ignored here (but kept for interface parity).
    • Returns a dict (JSON) if conversion succeeds, otherwise the raw markdown string.
    """

    if model is None:
        model = os.getenv("QWEN_MODEL")
    print(model)

    markdown = _qwen_markdown_answer(query, model=model)

    # try fast regex-based extraction first  (cheap)
    m = re.search(r"\{[\s\S]*\}", markdown)
    print("Regex match:", m)
    if m:
        try:
            return json.loads(m.group(0))          # success → dict
        except json.JSONDecodeError:
            pass                                   # fall through to fixer

    # fallback: fixer second-call (cheap)
    try:
        return _qwen_markdown_to_json(markdown)
    except Exception:
        return markdown    
