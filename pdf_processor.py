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
    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        pages.append(
            {
                "page_number": i,
                "text": page.get_text()
            }
        )
    doc.close()
    return pages   

def split_text_into_chunks(pages, chunk_size=900, overlap=50):
    """
    Splits text of each page into chunks with a specified chunk size and overlap.
    Uses tiktoken to tokenize the text.
    Returns a list of text strings with the page number embedded in each chunk.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = []
    
    for page in pages:
        page_number = page["page_number"]
        text = page["text"]
        tokens = tokenizer.encode(text)
        
        # Adjust chunk size for very large documents
        page_chunk_size = chunk_size
        
        # Create overlapping chunks per page and embed page number in the text itself
        for i in range(0, len(tokens), page_chunk_size - overlap):
            chunk_tokens = tokens[i:i + page_chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            # Embed page information directly into the text chunk
            combined_text = f"Page {page_number}:\n{chunk_text}"
            chunks.append(combined_text)
    return chunks

async def generate_embeddings(chunks, openai_client):
    """
    Generates embeddings for a list of text chunks using OpenAI's embedding API.
    Processes chunks in batches asynchronously.
    """
    batch_size = 20  # Adjust batch size based on your needs
    all_embeddings = []
    
    # Since chunks are now plain text strings with the page number included, use them directly
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        # Offload the API call to a thread to avoid blocking
        response = await asyncio.to_thread(
            openai_client.embeddings.create,
            model="text-embedding-3-large",
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
    The page number is included directly in the text field.
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
                "text": chunk  # This text already includes the page number information
            }
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)