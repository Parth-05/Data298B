# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model_config import generate_answer_from_mistral, generate_answer_from_gpt, generate_answer_from_gemini
from vector_store import search_similar_texts

app = FastAPI()

# Enable CORS for frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/query")
def process_query(req: QueryRequest):
    top_docs = search_similar_texts(req.query, top_k=req.top_k)
    print(top_docs)

    # Extract only document texts for generation
    doc_texts = [item["document"] for item in top_docs]

    answer = generate_answer_from_mistral(req.query, doc_texts)
    return {
        "question": req.query,
        "retrieved_documents": top_docs,  # Keep similarity here
        "answer": answer
    }

@app.post("/query_gpt")
def process_query_gpt(req: QueryRequest):
    # top_docs  = search_similar_texts(req.query, top_k=req.top_k)
    # we pass doc list for future use but generate_answer_from_gpt ignores it for now
    # answer    = generate_answer_from_gpt(req.query, [d["document"] for d in top_docs])
    answer    = generate_answer_from_gpt(req.query, None)

    return {
        "question": req.query,
        # "retrieved_documents": top_docs,
        "answer": answer,                 # could be dict *or* markdown str
        "model_used": "gpt",
    }

@app.post("/query_gemini")              # NEW
def process_query_gemini(req: QueryRequest):
    # top_docs = search_similar_texts(req.query, top_k=req.top_k)
    # answer   = generate_answer_from_gemini(
    #     req.query, [d["document"] for d in top_docs]
    # )
    answer   = generate_answer_from_gemini(
        req.query, None
    )
    return {
        "question": req.query,
        # "retrieved_documents": top_docs,
        "answer": answer,
        "model_used": "gemini",
    }

@app.get("/")
def read_root():
    return {"message": "RAG application with Mistral and TiDB is running."}
