# vector_store.py
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from model_config import text_to_embedding

# TiDB Cloud Connection
ssl_path = 'ca.pem'  # Ensure this file exists in your working directory
engine = create_engine(
    'mysql+pymysql://DZCzEuBaWGAmRrr.root:pE7nKWUqFoTnRwHM@gateway01.us-west-2.prod.aws.tidbcloud.com:4000/data298A',
    connect_args={"ssl": {"ca": ssl_path}}
)

def string_to_array(s):
    s = s.strip('[]')
    return np.array([float(x) for x in s.split(',')])

def search_similar_texts(query, top_k=5):
    query_embedding = text_to_embedding(query)

    sql_query = 'SELECT document, embedding FROM embedded_documents'
    connection = engine.raw_connection()
    try:
        df = pd.read_sql_query(sql_query, con=connection)
    finally:
        connection.close()

    if df.empty:
        return ["No documents found."]

    df['embedding'] = df['embedding'].apply(string_to_array)
    similarities = cosine_similarity([query_embedding], np.vstack(df['embedding'].tolist()))[0]
    df['similarity'] = similarities

    df = df[df['similarity'] >= 0.3]
    top_k_docs = df.nlargest(top_k, 'similarity')
    # return top_k_docs['document'].tolist()
    return [
        {"document": row["document"], "similarity": round(row["similarity"], 4)}
        for _, row in top_k_docs.iterrows()
        ]

def insert_text(text):
    embedding = text_to_embedding(text)
    embedding_str = ','.join(map(str, embedding))
    insert_query = f"""
    INSERT INTO embedded_documents (document, embedding)
    VALUES (%s, %s)
    """
    with engine.begin() as conn:
        conn.execute(insert_query, (text, f"[{embedding_str}]"))
    return "Inserted successfully"
