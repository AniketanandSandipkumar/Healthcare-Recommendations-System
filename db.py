from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="user")
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    preferences = Column(String, nullable=True)
    logs = relationship("PredictionLog", back_populates="user")
    activities = relationship("ActivityLog", back_populates="user")

class PredictionLog(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    disease = Column(String)
    drug = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="logs")

class ActivityLog(Base):
    __tablename__ = "activities"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action_type = Column(String)  # prediction, search, feedback, click
    details = Column(Text)  # JSON or plain text
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="activities")

class FeedbackLog(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    prediction_id = Column(Integer, ForeignKey("logs.id"))  # feedback linked to a prediction
    text = Column(Text)  # raw feedback
    sentiment = Column(String)  # positive, negative, neutral
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")
    prediction = relationship("PredictionLog")

Base.metadata.create_all(bind=engine)
