from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks, Query, Form
from sqlalchemy.orm import Session
from database import get_db, engine, Base
from models import PDF, User
from auth import get_current_user, UserCreate, authenticate_user, create_access_token, get_password_hash
from pdf_processor import extract_text_from_file, split_text_into_chunks, generate_embeddings, store_in_qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import os
import config
from openai import OpenAI
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, JSONResponse
from datetime import datetime
from pydantic import BaseModel
import pandas as pd
import tempfile
import asyncio
from typing import List
from sqlalchemy import or_

app = FastAPI()
Base.metadata.create_all(bind=engine)

# Add this class definition
class UpdateDescription(BaseModel):
    description: str

# Add these new models
class CollectionCreate(BaseModel):
    name: str
    description: str = None

class CollectionUpdate(BaseModel):
    description: str

# Define supported file types
SUPPORTED_FILE_TYPES = {
    ".pdf": "pdf",
    ".xlsx": "xlsx",
    ".xls": "xlsx"
}

def get_file_type(filename):
    """Get the file type from filename extension"""
    extension = os.path.splitext(filename)[1].lower()
    return SUPPORTED_FILE_TYPES.get(extension, "unknown")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

def init_qdrant(qdrant_client):
    collections = qdrant_client.get_collections()
    if "pdf_chunks" not in [col.name for col in collections.collections]:
        qdrant_client.create_collection(
            collection_name="pdf_chunks",
            vectors_config=qmodels.VectorParams(size=3072, distance=qmodels.Distance.COSINE)
        )

@app.on_event("startup")
def startup_event():
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    init_qdrant(qdrant_client)

# Authentication Endpoints
@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, password_hash=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"message": "User registered successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}
    


# File Management Endpoints
@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    description: str = Form(None),
    collection: str = Form(None),  # Use Form to get the collection
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    print(f"Received upload request with collection: {collection}")  # Debug print
    
    # Validate file type
    file_type = get_file_type(file.filename)
    if file_type == "unknown":
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_FILE_TYPES.keys())}"
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
        temp_path = temp_file.name
        chunk_size = 1024 * 1024  # 1MB chunks
        while chunk := await file.read(chunk_size):
            temp_file.write(chunk)
    
    try:
        # Create file record with collection and file type
        document = PDF(
            user_id=current_user.id,
            filename=file.filename,
            description=description,
            status="pending",
            collection=collection,
            file_type=file_type
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Schedule processing
        background_tasks.add_task(
            process_file_in_background,
            temp_path,
            file.filename,
            description,
            current_user.id,
            document.id,
            collection,
            file_type
        )
        
        return JSONResponse(
            status_code=202,
            content={
                "message": f"{file_type.upper()} upload in progress",
                "pdf_id": document.id,
                "status": "pending",
                "collection": collection,
                "file_type": file_type
            }
        )
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/pdfs")
def list_files(
    limit: int = 40,
    offset: int = 0,
    collection: str = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # Base query
    query = db.query(PDF).filter(PDF.user_id == current_user.id)
    
    # Add collection filter if specified
    if collection:
        if collection == "default":
            # Handle default collection (either NULL or empty collection field)
            query = query.filter(
                or_(
                    PDF.collection == None,
                    PDF.collection == "",
                    PDF.collection == "default"
                )
            )
        else:
            query = query.filter(PDF.collection == collection)
    
    # Get total count for pagination
    total_count = query.count()
    
    # Get paginated results
    files = query.offset(offset).limit(limit).all()
    
    return {
        "pdfs": [{"id": file.id, "filename": file.filename, "description": file.description, 
                  "upload_date": file.upload_date.isoformat(), 
                  "status": file.status, 
                  "pages_total": file.pages_total,
                  "pages_indexed": file.pages_indexed,
                  "collection": file.collection,
                  "file_type": file.file_type if hasattr(file, 'file_type') else "pdf",
                  "error_message": file.error_message} for file in files],
        "total_count": total_count
    }

@app.put("/pdfs/{pdf_id}")
def update_pdf(
    pdf_id: int,
    update_data: UpdateDescription,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # Update description in the database
    pdf = db.query(PDF).filter(PDF.id == pdf_id, PDF.user_id == current_user.id).first()
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    pdf.description = update_data.description
    db.commit()

    # Initialize Qdrant client
    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    # Define the filter to find points matching pdf_id and user_id
    query_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(key="pdf_id", match=qmodels.MatchValue(value=pdf_id)),
            qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=current_user.id))
        ]
    )

    # Scroll for points that match the filter to get their IDs
    scroll_result = qdrant_client.scroll(
        collection_name="pdf_chunks",
        scroll_filter=query_filter,
        limit=1000  # Adjust this limit based on the expected number of chunks per PDF
    )

    # Extract point IDs from the scroll result
    point_ids = [point.id for point in scroll_result[0]]

    # Update the payload for the matching points (if any exist)
    if point_ids:
        qdrant_client.set_payload(
            collection_name="pdf_chunks",
            payload={"description": update_data.description},
            points=point_ids
        )

    return {"message": "PDF updated successfully"}

@app.delete("/pdfs/{pdf_id}")
def delete_pdf(pdf_id: int, db: Session = Depends(get_db), 
               current_user = Depends(get_current_user)):
    pdf = db.query(PDF).filter(PDF.id == pdf_id, PDF.user_id == current_user.id).first()
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    base_collection = pdf.collection
    collection_name = f"user_{current_user.id}_{base_collection}" if base_collection else "pdf_chunks"
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    qdrant_client.delete(
        collection_name=collection_name,
        points_selector=qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="pdf_id", match=qmodels.MatchValue(value=pdf_id)),
                qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=current_user.id)),
            ]
        )
    )
    
    db.delete(pdf)
    db.commit()
    return {"message": "PDF deleted successfully"}

# Update the process_file_in_background function to handle both PDFs and Excel files
async def process_file_in_background(
    temp_path: str,
    filename: str,
    description: str,
    user_id: int,
    file_id: int,
    collection: str = None,
    file_type: str = "pdf"
):
    db = None
    try:
        # Update status to "processing" to give more detailed feedback
        db = next(get_db())
        file_record = db.query(PDF).filter(PDF.id == file_id).first()
        if file_record:
            file_record.status = "processing"
            db.commit()
        
        # Extract text with progress tracking
        pages = await extract_text_from_file(temp_path)
        
        # Update status to show progress
        db = next(get_db())
        file_record = db.query(PDF).filter(PDF.id == file_id).first()
        if file_record:
            file_record.status = "chunking"
            db.commit()
        
        chunks = split_text_into_chunks(pages)
        
        # Update status again
        db = next(get_db())
        file_record = db.query(PDF).filter(PDF.id == file_id).first()
        if file_record:
            file_record.status = "embedding"
            db.commit()
            
        openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        embeddings = await generate_embeddings(chunks, openai_client)
        
        # Final embedding storage
        db = next(get_db())
        file_record = db.query(PDF).filter(PDF.id == file_id).first()
        if file_record:
            file_record.status = "storing"
            db.commit()
            
        qdrant_client = QdrantClient(url=config.QDRANT_URL)
        
        # Store in the specified collection if provided, otherwise use default
        collection_name = f"user_{user_id}_{collection}" if collection else "pdf_chunks"
        print(f"Storing in collection: {collection_name}")  # Debug print
        
        # Create collection if it doesn't exist
        collections = qdrant_client.get_collections()
        if collection_name not in [col.name for col in collections.collections]:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=3072, distance=qmodels.Distance.COSINE)
            )
        
        # Store in smaller batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            store_in_qdrant(
                batch_chunks, 
                batch_embeddings, 
                file_id, 
                user_id, 
                filename, 
                description, 
                qdrant_client,
                collection_name  # Pass the collection name
            )
            await asyncio.sleep(0.1)
        
        # Update status and collection in database
        db = next(get_db())
        file_record = db.query(PDF).filter(PDF.id == file_id).first()
        if file_record:
            file_record.status = "processed"
            file_record.pages_total = len(pages)
            
            # Calculate pages indexed based on file type
            if file_type == "pdf":
                file_record.pages_indexed = len({
                    int(c.split(":",1)[0].split()[1]) for c in chunks
                    if "Page" in c.split(":",1)[0]
                })
            else:  # Excel files
                file_record.pages_indexed = len({
                    int(c.split(":",1)[0].split()[1]) for c in chunks
                    if "Sheet" in c.split(":",1)[0]
                })
                
            file_record.collection = collection
            db.commit()
            
    except Exception as e:
        # Log the error
        print(f"Error processing file {filename}: {str(e)}")
        # Update status in database
        if db is None:
            db = next(get_db())
        file_record = db.query(PDF).filter(PDF.id == file_id).first()
        if file_record:
            file_record.status = "error"
            file_record.error_message = str(e)
            db.commit()
    finally:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Add a status endpoint to check processing status
@app.get("/pdfs/{pdf_id}/status")
async def get_pdf_status(
    pdf_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    pdf = db.query(PDF).filter(PDF.id == pdf_id, PDF.user_id == current_user.id).first()
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
        
    return {
        "pdf_id": pdf.id,
        "status": pdf.status,
        "pages_total": pdf.pages_total,
        "pages_indexed": pdf.pages_indexed,
        "error_message": pdf.error_message if hasattr(pdf, "error_message") else None
    }

# Add new collection management endpoints
@app.post("/collections")
async def create_collection(
    collection: CollectionCreate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # Check if collection already exists
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    collections = qdrant_client.get_collections()
    collection_name = f"user_{current_user.id}_{collection.name}"
    
    if collection_name in [col.name for col in collections.collections]:
        raise HTTPException(status_code=400, detail="Collection name already exists")
    
    # Create collection in Qdrant
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=3072, distance=qmodels.Distance.COSINE)
        )
        return {"message": f"Collection '{collection.name}' created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections(
    current_user = Depends(get_current_user)
):
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    collections = qdrant_client.get_collections()
    
    user_collections = []
    for col in collections.collections:
        if col.name.startswith(f"user_{current_user.id}_"):
            # Get collection info to get the vectors count
            collection_info = qdrant_client.get_collection(col.name)
            user_collections.append({
                "name": col.name.replace(f"user_{current_user.id}_", ""),
                "vectors_count": collection_info.points_count  # Use points_count instead of vectors_count
            })
    
    return user_collections

@app.delete("/collections/{collection_name}")
async def delete_collection(
    collection_name: str,
    current_user = Depends(get_current_user)
):
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    full_collection_name = f"user_{current_user.id}_{collection_name}"
    try:
        qdrant_client.delete_collection(collection_name=full_collection_name)
        return {"message": f"Collection '{collection_name}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{collection_name}/pdfs/{pdf_id}")
async def add_pdf_to_collection(
    collection_name: str,
    pdf_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # Verify PDF exists and belongs to user
    pdf = db.query(PDF).filter(PDF.id == pdf_id, PDF.user_id == current_user.id).first()
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    # Get existing vectors from pdf_chunks
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    points = qdrant_client.scroll(
        collection_name="pdf_chunks",
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="pdf_id", match=qmodels.MatchValue(value=pdf_id)),
                qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=current_user.id))
            ]
        )
    )[0]
    
    # Add vectors to the specified collection
    full_collection_name = f"user_{current_user.id}_{collection_name}"
    try:
        qdrant_client.upsert(
            collection_name=full_collection_name,
            points=[point for point in points]
        )
        return {"message": f"PDF added to collection '{collection_name}' successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{collection_name}/pdfs/{pdf_id}")
async def remove_pdf_from_collection(
    collection_name: str,
    pdf_id: int,
    current_user = Depends(get_current_user)
):
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    full_collection_name = f"user_{current_user.id}_{collection_name}"
    
    try:
        qdrant_client.delete(
            collection_name=full_collection_name,
            points_selector=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="pdf_id", match=qmodels.MatchValue(value=pdf_id)),
                    qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=current_user.id))
                ]
            )
        )
        return {"message": f"PDF removed from collection '{collection_name}' successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))