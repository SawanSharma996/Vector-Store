from sqlalchemy import Column, Integer, String, ForeignKey, Date
from sqlalchemy.orm import relationship
from database import Base
import datetime
from datetime import timezone  # Import timezone module

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
    upload_date = Column(Date, default=lambda: datetime.date.today(), index=True)  # Set default value to today's date
    user = relationship("User", back_populates="pdfs") 