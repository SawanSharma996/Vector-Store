from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
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
from fastapi.responses import FileResponse
from datetime import datetime
from pydantic import BaseModel
import pandas as pd

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
    file: UploadFile = File(...),
    description: str = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # Check if user has already uploaded 10 PDFs in this batch
    recent_uploads = db.query(PDF).filter(
        PDF.user_id == current_user.id,
        PDF.upload_date == datetime.today().date()
    ).count()
    
    

    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(file.file.read())
        
        text = extract_text_from_pdf(temp_path)
        chunks = split_text_into_chunks(text)
        openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        embeddings = generate_embeddings(chunks, openai_client)
        
        pdf = PDF(
            user_id=current_user.id,
            filename=file.filename,
            description=description
        )
        db.add(pdf)
        db.commit()
        db.refresh(pdf)
        
        qdrant_client = QdrantClient(url=config.QDRANT_URL)
        store_in_qdrant(chunks, embeddings, pdf.id, current_user.id, file.filename, description, qdrant_client)
        
        return {"message": "PDF uploaded successfully", "pdf_id": pdf.id}
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/pdfs")
def list_pdfs(db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    pdfs = db.query(PDF).filter(PDF.user_id == current_user.id).all()
    return [{"id": pdf.id, "filename": pdf.filename, "description": pdf.description, 
             "upload_date": pdf.upload_date.isoformat()} for pdf in pdfs]

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