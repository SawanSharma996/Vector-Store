from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks, Query, Form
from sqlalchemy.orm import Session
from database import get_db, engine, Base
from models import PDF, User
from auth import get_current_user, UserCreate, authenticate_user, create_access_token, get_password_hash
from pdf_processor import extract_text_from_pdf, split_text_into_chunks, generate_embeddings, store_in_qdrant
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

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

def init_qdrant(qdrant_client):
    collections = qdrant_client.get_collections()
    if "pdf_chunks" not in [col.name for col in collections.collections]:
        qdrant_client.create_collection(
            collection_name="pdf_chunks",
            vectors_config=qmodels.VectorParams(size=1536, distance=qmodels.Distance.COSINE)
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
    


# PDF Management Endpoints
@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    description: str = Form(None),
    collection: str = Form(None),  # Use Form to get the collection
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    print(f"Received upload request with collection: {collection}")  # Debug print
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
        temp_path = temp_file.name
        chunk_size = 1024 * 1024  # 1MB chunks
        while chunk := await file.read(chunk_size):
            temp_file.write(chunk)
    
    try:
        # Create PDF record with collection
        pdf = PDF(
            user_id=current_user.id,
            filename=file.filename,
            description=description,
            status="pending",
            collection=collection  # Set the collection
        )
        db.add(pdf)
        db.commit()
        db.refresh(pdf)
        
        # Schedule processing
        background_tasks.add_task(
            process_pdf_in_background,
            temp_path,
            file.filename,
            description,
            current_user.id,
            pdf.id,
            collection
        )
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "PDF upload in progress",
                "pdf_id": pdf.id,
                "status": "pending",
                "collection": collection
            }
        )
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/pdfs")
def list_pdfs(
    limit: int = 10,
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
    pdfs = query.offset(offset).limit(limit).all()
    
    return {
        "pdfs": [{"id": pdf.id, "filename": pdf.filename, "description": pdf.description, 
                  "upload_date": pdf.upload_date.isoformat(), 
                  "status": pdf.status, 
                  "collection": pdf.collection,
                  "error_message": pdf.error_message} for pdf in pdfs],
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
    
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    qdrant_client.delete(
        collection_name="pdf_chunks",
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

# Update the process_pdf_in_background function to be asynchronous and handle large files better
async def process_pdf_in_background(
    temp_path: str,
    filename: str,
    description: str,
    user_id: int,
    pdf_id: int,
    collection: str = None
):
    db = None
    try:
        # Update status to "processing" to give more detailed feedback
        db = next(get_db())
        pdf = db.query(PDF).filter(PDF.id == pdf_id).first()
        if pdf:
            pdf.status = "processing"
            db.commit()
        
        # Extract text with progress tracking
        text = await extract_text_from_pdf(temp_path)
        
        # Update status to show progress
        db = next(get_db())
        pdf = db.query(PDF).filter(PDF.id == pdf_id).first()
        if pdf:
            pdf.status = "chunking"
            db.commit()
            
        chunks = split_text_into_chunks(text)
        
        # Update status again
        db = next(get_db())
        pdf = db.query(PDF).filter(PDF.id == pdf_id).first()
        if pdf:
            pdf.status = "embedding"
            db.commit()
            
        openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        embeddings = await generate_embeddings(chunks, openai_client)
        
        # Final embedding storage
        db = next(get_db())
        pdf = db.query(PDF).filter(PDF.id == pdf_id).first()
        if pdf:
            pdf.status = "storing"
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
                vectors_config=qmodels.VectorParams(size=1536, distance=qmodels.Distance.COSINE)
            )
        
        # Store in smaller batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            store_in_qdrant(
                batch_chunks, 
                batch_embeddings, 
                pdf_id, 
                user_id, 
                filename, 
                description, 
                qdrant_client,
                collection_name  # Pass the collection name
            )
            await asyncio.sleep(0.1)
        
        # Update status and collection in database
        db = next(get_db())
        pdf = db.query(PDF).filter(PDF.id == pdf_id).first()
        if pdf:
            pdf.status = "processed"
            pdf.collection = collection  # Make sure to update the collection in the database
            db.commit()
            
    except Exception as e:
        # Log the error
        print(f"Error processing PDF {filename}: {str(e)}")
        # Update status in database
        if db is None:
            db = next(get_db())
        pdf = db.query(PDF).filter(PDF.id == pdf_id).first()
        if pdf:
            pdf.status = "error"
            pdf.error_message = str(e)
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
            vectors_config=qmodels.VectorParams(size=1536, distance=qmodels.Distance.COSINE)
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