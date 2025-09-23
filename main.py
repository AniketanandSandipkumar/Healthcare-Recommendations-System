from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from pydantic import BaseModel
from db import SessionLocal, User, PredictionLog,ActivityLog
from sqlalchemy import func
from textblob import TextBlob
from db import FeedbackLog
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors


SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI()

# ------------------ DB ------------------
# ------------------ HEART DISEASE MODEL ------------------
heart_model = joblib.load("heart_model.pkl")
heart_scaler = joblib.load("heart_scaler.pkl")   # if you scaled features

heart_features = [
    "age", "sex", "chest_pain", "blood_pressure", "cholestrol",
    "fbs", "restecg", "max_heart_rate", "exang", 
    "oldpeak", "slope", "major_vessels", "thal"
]

class HeartRequest(BaseModel):
    age: int
    sex: int
    chest_pain: int
    blood_pressure: float
    cholestrol: float
    fbs: int
    restecg: int
    max_heart_rate: int
    exang: int
    oldpeak: float
    slope: int
    major_vessels: int
    thal: int

@app.post("/predict_heart")
def predict_heart(req: HeartRequest):
    # Convert request to dataframe
    X = pd.DataFrame([req.dict().values()], columns=heart_features)
    
    # Scale input
    X_scaled = heart_scaler.transform(X)

    # Predict
    prediction = heart_model.predict(X_scaled)[0]
    proba = heart_model.predict_proba(X_scaled)[0].tolist()

    return {
        "prediction": int(prediction),   # 0 = No Disease, 1 = Heart Disease
        "probabilities": {"No Disease": proba[0], "Disease": proba[1]}
    }

# ------------------ KNN MODEL LOADING ------------------
knn_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
df_knn = pd.read_csv("disease_drug_mapping.csv")

symptom_cols = [c for c in df_knn.columns if "Symptom" in c]
feature_matrix = scaler.transform(df_knn[symptom_cols])

def get_knn_recommendations(disease_name, df, model, feature_matrix, num_recs=5):
    df['Disease'] = df['Disease'].str.strip().str.lower()
    disease_name = disease_name.strip().lower()
    
    matches = df[df['Disease'].str.contains(disease_name, case=False, na=False)]['Disease'].unique()
    if len(matches) == 0:
        return None
    
    closest_match = matches[0]
    idx = df[df['Disease'] == closest_match].index[0]
    distances, indices = model.kneighbors([feature_matrix[idx]])
    
    recommended_diseases = df.iloc[indices[0][1:num_recs+1]]
    return recommended_diseases[['Disease', 'Drug']].to_dict(orient="records")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# AUTH HELPERS 
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = get_user(db, username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ------------------ SCHEMAS ------------------
class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"

class UserProfileUpdate(BaseModel):
    age: int | None = None
    gender: str | None = None
    preferences: str | None = None

class PredictionRequest(BaseModel):
    symptoms: list[str]

class PredictionResponse(BaseModel):
    disease: str
    drug: str

# ------------------ ROUTES ------------------
@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    if get_user(db, user.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    new_user = User(
        username=user.username,
        hashed_password=get_password_hash(user.password),
        role=user.role
    )
    db.add(new_user)
    db.commit()
    return {"msg": "User created successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}

@app.put("/profile")
def update_profile(profile: UserProfileUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if profile.age: current_user.age = profile.age
    if profile.gender: current_user.gender = profile.gender
    if profile.preferences: current_user.preferences = profile.preferences
    db.commit()
    return {"msg": "Profile updated successfully"}

@app.get("/profile")
def get_profile(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "role": current_user.role,
        "age": current_user.age,
        "gender": current_user.gender,
        "preferences": current_user.preferences
    }

@app.get("/logs")
def get_logs(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role == "admin":
        logs = db.query(PredictionLog).all()
    else:
        logs = db.query(PredictionLog).filter(PredictionLog.user_id == current_user.id).all()
    return [{"disease": l.disease, "drug": l.drug, "timestamp": l.timestamp} for l in logs]

class ActivityCreate(BaseModel):
    action_type: str  
    details: str      

@app.post("/activity")
def log_activity(activity: ActivityCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    new_activity = ActivityLog(
        user_id=current_user.id,
        action_type=activity.action_type,
        details=activity.details
    )
    db.add(new_activity)
    db.commit()
    return {"msg": "Activity logged"}

@app.get("/activity")
def get_activity(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role == "admin":
        activities = db.query(ActivityLog).all()
    else:
        activities = db.query(ActivityLog).filter(ActivityLog.user_id == current_user.id).all()
    return [{"type": a.action_type, "details": a.details, "timestamp": a.timestamp} for a in activities]

from sqlalchemy import func

@app.get("/analytics/user")
def user_analytics(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Count predictions per disease for this user
    disease_counts = db.query(PredictionLog.disease, func.count(PredictionLog.disease))\
        .filter(PredictionLog.user_id == current_user.id)\
        .group_by(PredictionLog.disease).all()

    return {
        "most_common_disease": max(disease_counts, key=lambda x: x[1])[0] if disease_counts else None,
        "disease_distribution": dict(disease_counts)
    }

@app.get("/analytics/global")
def global_trends(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role not in ["admin", "analyst"]:
        raise HTTPException(status_code=403, detail="Access denied")

    # Top 5 diseases overall
    top_diseases = db.query(PredictionLog.disease, func.count(PredictionLog.disease))\
        .group_by(PredictionLog.disease)\
        .order_by(func.count(PredictionLog.disease).desc())\
        .limit(5).all()

    # Top 5 drugs overall
    top_drugs = db.query(PredictionLog.drug, func.count(PredictionLog.drug))\
        .group_by(PredictionLog.drug)\
        .order_by(func.count(PredictionLog.drug).desc())\
        .limit(5).all()

    return {
        "top_diseases": dict(top_diseases),
        "top_drugs": dict(top_drugs)
    }

class FeedbackRequest(BaseModel):
    prediction_id: int
    text: str

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Run sentiment analysis
    blob = TextBlob(req.text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    feedback = FeedbackLog(
        user_id=current_user.id,
        prediction_id=req.prediction_id,
        text=req.text,
        sentiment=sentiment
    )
    db.add(feedback)
    db.commit()
    return {"msg": "Feedback submitted", "sentiment": sentiment}

@app.get("/feedback/user")
def get_user_feedback(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    feedbacks = db.query(FeedbackLog).filter(FeedbackLog.user_id == current_user.id).all()
    return [{"text": f.text, "sentiment": f.sentiment, "timestamp": f.timestamp} for f in feedbacks]

@app.get("/feedback/global")
def get_global_feedback(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role not in ["admin", "analyst"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    sentiments = db.query(FeedbackLog.sentiment, func.count(FeedbackLog.sentiment))\
        .group_by(FeedbackLog.sentiment).all()
    return {"sentiment_distribution": dict(sentiments)}

@app.get("/analytics/feedback_trends")
def feedback_trends(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.role not in ["admin", "analyst"]:
        raise HTTPException(status_code=403, detail="Access denied")

    # Top diseases with positive vs negative feedback
    results = db.query(PredictionLog.disease, FeedbackLog.sentiment, func.count(FeedbackLog.sentiment))\
        .join(FeedbackLog, FeedbackLog.prediction_id == PredictionLog.id)\
        .group_by(PredictionLog.disease, FeedbackLog.sentiment).all()

    trends = {}
    for disease, sentiment, count in results:
        if disease not in trends:
            trends[disease] = {}
        trends[disease][sentiment] = count

    return {"feedback_trends": trends}

@app.get("/recommend_knn/{disease_name}")
def recommend_knn(disease_name: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    recs = get_knn_recommendations(disease_name, df_knn, knn_model, feature_matrix)
    if recs is None:
        raise HTTPException(status_code=404, detail=f"No disease found for '{disease_name}'")
    return {"input_disease": disease_name, "recommendations": recs}


