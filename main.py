from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from pydantic import BaseModel
from db import SessionLocal, User, PredictionLog, ActivityLog, FeedbackLog
from passlib.context import CryptContext
from jose import jwt, JWTError
import pandas as pd
import joblib
from textblob import TextBlob

# ================= CONFIG =================
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI(title="Healthcare Recommendation Backend")

# ================= DB =================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ================= MODELS =================
# Heart Disease Model
heart_model = joblib.load("models/heart_model.pkl")
heart_scaler = joblib.load("models/heart_scaler.pkl")
heart_features = [
    "age","sex","chest_pain","blood_pressure","cholestrol",
    "fbs","restecg","max_heart_rate","exang",
    "oldpeak","slope","major_vessels","thal"
]

# KNN Disease-Drug Model
knn_model = joblib.load("models/knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
df_knn = pd.read_csv("disease_drug_mapping.csv")
symptom_cols = [c for c in df_knn.columns if "Symptom" in c]
feature_matrix = scaler.transform(df_knn[symptom_cols])

# ================= HELPERS =================
def get_knn_recommendations(disease_name, df, model, feature_matrix, num_recs=5):
    df['Disease'] = df['Disease'].str.strip().str.lower()
    disease_name = disease_name.strip().lower()
    matches = df[df['Disease'].str.contains(disease_name, case=False, na=False)]['Disease'].unique()
    if len(matches) == 0: return None
    closest_match = matches[0]
    idx = df[df['Disease'] == closest_match].index[0]
    distances, indices = model.kneighbors([feature_matrix[idx]])
    recommended_diseases = df.iloc[indices[0][1:num_recs+1]]
    return recommended_diseases[['Disease','Drug']].to_dict(orient="records")

def get_password_hash(password): return pwd_context.hash(password)
def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user(db: Session, username: str): return db.query(User).filter(User.username==username).first()
def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password): return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = get_user(db, username)
    if user is None: raise HTTPException(status_code=401, detail="User not found")
    return user

# ================= SCHEMAS =================
class HeartRequest(BaseModel):
    age: int; sex: int; chest_pain: int
    blood_pressure: float; cholestrol: float
    fbs: int; restecg: int; max_heart_rate: int
    exang: int; oldpeak: float; slope: int
    major_vessels: int; thal: int

class UserCreate(BaseModel):
    username: str; password: str; role: str = "user"

class UserProfileUpdate(BaseModel):
    age: int | None = None
    gender: str | None = None
    preferences: str | None = None

class ActivityCreate(BaseModel):
    action_type: str
    details: str

class FeedbackRequest(BaseModel):
    prediction_id: int
    text: str

# ================= ROUTES =================
@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    if get_user(db, user.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    new_user = User(username=user.username, hashed_password=get_password_hash(user.password), role=user.role)
    db.add(new_user); db.commit()
    return {"msg":"User created successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user: raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type":"bearer"}

@app.put("/profile")
def update_profile(profile: UserProfileUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if profile.age: current_user.age = profile.age
    if profile.gender: current_user.gender = profile.gender
    if profile.preferences: current_user.preferences = profile.preferences
    db.commit(); return {"msg": "Profile updated successfully"}

@app.get("/profile")
def get_profile(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username,"role": current_user.role,
            "age": current_user.age,"gender": current_user.gender,"preferences": current_user.preferences}

@app.post("/predict_heart")
def predict_heart(req: HeartRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    X = pd.DataFrame([req.dict().values()], columns=heart_features)
    X_scaled = heart_scaler.transform(X)
    prediction = heart_model.predict(X_scaled)[0]
    proba = heart_model.predict_proba(X_scaled)[0].tolist()
    log = PredictionLog(user_id=current_user.id, disease="Heart Disease", drug="N/A")
    db.add(log); db.commit()
    return {"prediction": int(prediction), "probabilities": {"No Disease": proba[0], "Disease": proba[1]}}

@app.get("/recommend_knn/{disease_name}")
def recommend_knn(disease_name: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    recs = get_knn_recommendations(disease_name, df_knn, knn_model, feature_matrix)
    if recs is None: raise HTTPException(status_code=404, detail=f"No disease found for '{disease_name}'")
    for r in recs:
        log = PredictionLog(user_id=current_user.id, disease=r['Disease'], drug=r['Drug'])
        db.add(log)
    db.commit()
    return {"input_disease": disease_name, "recommendations": recs}

@app.post("/activity")
def log_activity(activity: ActivityCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    new_activity = ActivityLog(user_id=current_user.id, action_type=activity.action_type, details=activity.details)
    db.add(new_activity); db.commit(); return {"msg": "Activity logged"}

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    blob = TextBlob(req.text); polarity = blob.sentiment.polarity
    if polarity>0.1: sentiment="positive"
    elif polarity<-0.1: sentiment="negative"
    else: sentiment="neutral"
    feedback = FeedbackLog(user_id=current_user.id, prediction_id=req.prediction_id, text=req.text, sentiment=sentiment)
    db.add(feedback); db.commit()
    return {"msg":"Feedback submitted", "sentiment":sentiment}
