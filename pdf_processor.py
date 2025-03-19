import pdfplumber
import tiktoken
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import uuid

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() or "" for page in pdf.pages)
    return text

def split_text_into_chunks(text, chunk_size=500):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def generate_embeddings(chunks, openai_client):
    response = openai_client.embeddings.create(model="text-embedding-ada-002", input=chunks)
    return [emb.embedding for emb in response.data]

def store_in_qdrant(chunks, embeddings, pdf_id, user_id,filename,description, qdrant_client):
    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "pdf_id": pdf_id,
                "user_id": user_id,
                "filename": filename,
                "description": description if description else "",  # Add filename to metadata
                "chunk_index": i,
                "text": chunk
            }
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    qdrant_client.upsert(collection_name="pdf_chunks", points=points)