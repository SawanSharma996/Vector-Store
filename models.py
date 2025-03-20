from sqlalchemy import Column, Integer, String, ForeignKey, Date
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime, date
from database import engine
from sqlalchemy import text

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    pdfs = relationship("PDF", back_populates="user")

class PDF(Base):
    __tablename__ = "pdfs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String, index=True)
    description = Column(String, nullable=True)
    upload_date = Column(Date, default=datetime.now().date())
    status = Column(String, default="processed")  # New status field: pending, processed, error
    error_message = Column(String, nullable=True)  # To store error details
    user = relationship("User", back_populates="pdfs")

