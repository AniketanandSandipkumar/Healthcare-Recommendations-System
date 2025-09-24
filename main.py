from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import jwt, JWTError
from db import SessionLocal, User
from pydantic import BaseModel

# Config
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI()

# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Auth helpers
def get_password_hash(password): return pwd_context.hash(password)
def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user(db: Session, username: str): 
    return db.query(User).filter(User.username==username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password): 
        return False
    return user

# Schemas
class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"

# Routes
@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    if get_user(db, user.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    new_user = User(username=user.username, hashed_password=get_password_hash(user.password), role=user.role)
    db.add(new_user)
    db.commit()
    return {"msg":"User created successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user: raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type":"bearer"}
# ----------------- DEBUG ROUTE -----------------
from sqlalchemy.orm import Session

@app.get("/debug/users")
def debug_list_users(admin_key: str, db: Session = Depends(get_db)):
    """
    Temporary debug route to list all registered users.
    Provide the query parameter admin_key for safety.
    Example: /debug/users?admin_key=my-debug-secret
    """
    SECRET_ADMIN_KEY = "my-debug-secret"  # Change to a strong secret
    if admin_key != SECRET_ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    users = db.query(User).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "hashed_password": u.hashed_password,
            "role": u.role
        } for u in users
    ]

# main.py
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from db import UserActivity, get_db, User
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

@app.post("/activity/log")
def log_activity(user_id: int, event_type: str, item: str, db: Session = Depends(get_db)):
    activity = UserActivity(user_id=user_id, event_type=event_type, item=item)
    db.add(activity)
    db.commit()
    return {"msg": "Activity logged"}


