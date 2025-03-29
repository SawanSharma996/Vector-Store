import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import uuid
import asyncio


async def extract_text_from_pdf(pdf_path):
    """
    Asynchronously extracts text from a PDF using PyMuPDF.
    """
    return await asyncio.to_thread(_extract_text_from_pdf, pdf_path)
    # Add timeout parameters and chunking for large files
   
def _extract_text_from_pdf(pdf_path):
    """
    Synchronously extracts text from a PDF using PyMuPDF.
    """
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text   

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """
    Splits text into chunks with a specified chunk size and overlap for better context preservation.
    Uses tiktoken to tokenize the text.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    
    # Adjust chunk size for very large documents
    if len(tokens) > 50000:
        chunk_size = 300
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

async def generate_embeddings(chunks, openai_client):
    """
    Generates embeddings for a list of text chunks using OpenAI's embedding API.
    Processes chunks in batches asynchronously.
    """
    batch_size = 20  # Adjust batch size based on your needs
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        # Offload the API call to a thread to avoid blocking
        response = await asyncio.to_thread(
            openai_client.embeddings.create,
            model="text-embedding-ada-002",
            input=batch
        )
        batch_embeddings = [emb.embedding for emb in response.data]
        all_embeddings.extend(batch_embeddings)
        # Small delay to avoid hitting rate limits
        if i + batch_size < len(chunks):
            await asyncio.sleep(0.5)
            
    return all_embeddings

def store_in_qdrant(chunks, embeddings, pdf_id, user_id, filename, description, qdrant_client, collection_name="pdf_chunks"):
    """
    Stores text chunks and their corresponding embeddings in Qdrant with additional metadata.
    """
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