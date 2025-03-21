import pdfplumber
import tiktoken
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import uuid
import asyncio


async def extract_text_from_pdf(pdf_path):
    # Add timeout parameters and chunking for large files
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        # Process pages in smaller batches to avoid memory issues
        for i in range(len(pdf.pages)):
            try:
                page_text = pdf.pages[i].extract_text() or ""
                text += page_text + "\n"
                # Periodically yield control back to the event loop
                if i % 10 == 0:
                    await asyncio.sleep(0)
            except Exception as e:
                print(f"Error extracting text from page {i}: {str(e)}")
    return text

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    # Add overlap between chunks for better context preservation
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    
    # Use smaller chunks for very large documents
    if len(tokens) > 50000:  # Adjust threshold as needed
        chunk_size = 300
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

async def generate_embeddings(chunks, openai_client):
    # Process in batches to avoid API limits and reduce memory usage
    batch_size = 20  # Adjust based on your needs
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        response = openai_client.embeddings.create(model="text-embedding-ada-002", input=batch)
        batch_embeddings = [emb.embedding for emb in response.data]
        all_embeddings.extend(batch_embeddings)
        # Add a small delay to avoid rate limits
        if i + batch_size < len(chunks):
            await asyncio.sleep(0.5)
            
    return all_embeddings

def store_in_qdrant(chunks, embeddings, pdf_id, user_id, filename, description, qdrant_client, collection_name="pdf_chunks"):
    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "pdf_id": pdf_id,
                "user_id": user_id,
                "filename": filename,
                "description": description if description else "",
                "chunk_index": i,
                "text": chunk
            }
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)