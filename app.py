from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
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

app = FastAPI()
Base.metadata.create_all(bind=engine)

# Add this class definition
class UpdateDescription(BaseModel):
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
    description: str = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # Create a unique temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
        temp_path = temp_file.name
        # Read file in chunks to avoid memory issues with large files
        chunk_size = 1024 * 1024  # 1MB chunks
        while chunk := await file.read(chunk_size):
            temp_file.write(chunk)
    
    try:
        # First create the PDF record with pending status
        pdf = PDF(
            user_id=current_user.id,
            filename=file.filename,
            description=description,
            status="pending"  # Add this status field to your PDF model
        )
        db.add(pdf)
        db.commit()
        db.refresh(pdf)
        
        # Schedule the processing to happen in the background
        background_tasks.add_task(
            process_pdf_in_background,
            temp_path,
            file.filename,
            description,
            current_user.id,
            pdf.id
        )
        
        return JSONResponse(
            status_code=202,  # Accepted
            content={
                "message": "PDF upload in progress",
                "pdf_id": pdf.id,
                "status": "pending"
            }
        )
    except Exception as e:
        # If something goes wrong before the background task starts
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/pdfs")
def list_pdfs(db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    pdfs = db.query(PDF).filter(PDF.user_id == current_user.id).all()
    return [{"id": pdf.id, "filename": pdf.filename, "description": pdf.description, 
             "upload_date": pdf.upload_date.isoformat(), 
             "status": pdf.status, 
             "error_message": pdf.error_message} for pdf in pdfs]

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
    pdf_id: int
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
                qdrant_client
            )
            await asyncio.sleep(0.1)  # Give the event loop a chance to process other tasks
        
        # Update status in database
        db = next(get_db())
        pdf = db.query(PDF).filter(PDF.id == pdf_id).first()
        if pdf:
            pdf.status = "processed"
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