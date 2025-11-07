from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import joblib
import shap
import secrets
import smtplib
from email.mime.text import MIMEText
import os

# -----------------------------
# 🌍 ENVIRONMENT CONFIG
# -----------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
RESET_KEY = os.getenv("RESET_KEY")
RESET_TOKEN_EXPIRE_MINUTES = int(os.getenv("RESET_TOKEN_EXPIRE_MINUTES", 30))
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM") or (SMTP_USER or "")
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:5173")

# -----------------------------
# ⚙️ DATABASE SETUP
# -----------------------------
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# -----------------------------
# 🧠 MODELS
# -----------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    disabled = Column(Boolean, default=False)
    transactions = relationship("TransactionLog", back_populates="user")

class TransactionLog(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    data = Column(JSON)
    result = Column(JSON)
    user = relationship("User", back_populates="transactions")

class PasswordReset(Base):
    __tablename__ = "password_resets"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)

Base.metadata.create_all(bind=engine)

# -----------------------------
# 🔐 AUTH CONFIG
# -----------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)
def get_password_hash(password): return pwd_context.hash(password)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user(db, username: str):
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db, email: str):
    return db.query(User).filter(User.email == email).first()

def _email_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASS and MAIL_FROM)

def send_reset_email(to_email: str, token: str) -> bool:
    reset_link = f"{FRONTEND_BASE_URL}/reset-password?token={token}"
    if not _email_configured():
        # Fallback: log reset link to console for development instead of raising
        print(f"[PASSWORD RESET LINK] {to_email}: {reset_link}")
        return False
    body = f"Click the link to reset your password (valid for {RESET_TOKEN_EXPIRE_MINUTES} minutes):\n{reset_link}"
    msg = MIMEText(body)
    msg["Subject"] = "Password Reset"
    msg["From"] = MAIL_FROM
    msg["To"] = to_email
    with smtplib.SMTP(host=SMTP_HOST, port=SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(MAIL_FROM, [to_email], msg.as_string())
    return True

def authenticate_user(db, identifier, password):
    user = get_user_by_email(db, identifier) or get_user(db, identifier)
    return user if user and verify_password(password, user.hashed_password) else None

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)):
    cred_error = HTTPException(status_code=401, detail="Invalid or expired token")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        subject = payload.get("sub")
        if not subject:
            raise cred_error
    except JWTError:
        raise cred_error
    user = get_user_by_email(db, subject) or get_user(db, subject)
    if not user:
        raise cred_error
    return user

# -----------------------------
# 🧠 ML MODEL LOADING
# -----------------------------
model = joblib.load("fraud_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

try:
    explainer = shap.TreeExplainer(model)
    print("✅ SHAP explainer ready.")
except Exception:
    explainer = None
    print("⚠️ SHAP not available for this model.")

# -----------------------------
# 🧩 Pydantic Models
# -----------------------------
class Token(BaseModel):
    access_token: str
    token_type: str

class Transaction(BaseModel):
    Customer_ID: str
    Gender: str
    Age: float
    State: str
    City: str
    Bank_Branch: str
    Account_Type: str
    Transaction_Time: float
    Transaction_Amount: float
    Merchant_ID: str
    Transaction_Type: str
    Merchant_Category: str
    Account_Balance: float
    Transaction_Device: str
    Transaction_Location: str
    Device_Type: str
    Transaction_Currency: str
    time_diff: float
    amount_mean: float
    amount_ratio: float
    tx_hour: float
    tx_day: float

# -----------------------------
# ⚙️ APP SETUP
# -----------------------------
app = FastAPI(title="💳 Fraud Detection API (Neon + JWT Secure)")

# -----------------------------
# 🔑 AUTH ENDPOINTS
# -----------------------------
@app.post("/register")
async def register(username: str, email: str, password: str, full_name: str | None = None, db=Depends(get_db)):
    if get_user(db, username):
        raise HTTPException(status_code=400, detail="Username already registered")
    if get_user_by_email(db, email):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = get_password_hash(password)
    user = User(username=username, email=email, hashed_password=hashed_pw, full_name=full_name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "✅ User registered successfully"}

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    subject = user.email or user.username
    access_token = create_access_token(data={"sub": subject})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/whoami")
async def whoami(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username, "email": current_user.email, "full_name": current_user.full_name}

# -----------------------------
# 🧮 HELPERS
# -----------------------------
def preprocess_input(data: Transaction):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    for col, le in encoders.items():
        if col in input_df.columns:
            val = str(input_df[col].iloc[0])
            try:
                input_df[col] = le.transform([val])
            except ValueError:
                if "<UNK>" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "<UNK>")
                input_df[col] = le.transform(["<UNK>"])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]
    input_df = input_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return input_df, input_dict

# -----------------------------
# 🔮 PREDICTION ENDPOINTS
# -----------------------------
@app.post("/predict")
async def predict(data: Transaction, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    input_df, input_dict = preprocess_input(data)
    prob = float(model.predict_proba(input_df)[:, 1][0])
    pred = int(prob > 0.5)

    result = {
        "fraud_probability": prob,
        "prediction": pred,
        "message": "⚠️ Fraudulent transaction detected!" if pred else "✅ Transaction looks normal"
    }

    log = TransactionLog(user_id=current_user.id, data=input_dict, result=result)
    db.add(log)
    db.commit()
    return result

@app.get("/logs")
async def get_logs(current_user: User = Depends(get_current_user), db=Depends(get_db)):
    logs = db.query(TransactionLog).filter(TransactionLog.user_id == current_user.id).order_by(TransactionLog.timestamp.desc()).limit(50).all()
    return [{"timestamp": log.timestamp, "data": log.data, "result": log.result} for log in logs]

# -----------------------------
# 🔑 PASSWORD RESET (admin-only via env RESET_KEY)
# -----------------------------
@app.post("/reset_password")
async def reset_password(username: str, new_password: str, reset_key: str, db=Depends(get_db)):
    if not RESET_KEY or reset_key != RESET_KEY:
        raise HTTPException(status_code=403, detail="Invalid reset key")
    user = get_user(db, username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.hashed_password = get_password_hash(new_password)
    db.add(user)
    db.commit()
    return {"message": "✅ Password updated"}

# -----------------------------
# ✉️ Forgot Password (email) and Reset Confirm
# -----------------------------
@app.post("/forgot_password")
async def forgot_password(email: str, background_tasks: BackgroundTasks, db=Depends(get_db)):
    user = get_user_by_email(db, email)
    if not user:
        # Do not reveal user existence
        return {"message": "If that email exists, a reset link has been sent"}
    token = secrets.token_urlsafe(32)
    expires = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
    pr = PasswordReset(user_id=user.id, token=token, expires_at=expires, used=False)
    db.add(pr)
    db.commit()
    # If email is configured, send in background; otherwise log link in send_reset_email
    background_tasks.add_task(send_reset_email, email, token)
    return {"message": "If that email exists, a reset link has been sent"}

@app.post("/reset_password_confirm")
async def reset_password_confirm(token: str, new_password: str, db=Depends(get_db)):
    pr: PasswordReset | None = db.query(PasswordReset).filter(PasswordReset.token == token).first()
    if not pr or pr.used or pr.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    user = db.query(User).filter(User.id == pr.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.hashed_password = get_password_hash(new_password)
    pr.used = True
    db.add(user)
    db.add(pr)
    db.commit()
    return {"message": "✅ Password updated"}

# -----------------------------
# 🏁 ROOT
# -----------------------------
@app.get("/")
def root():
    return {"message": "🚀 Fraud Detection API running with Neon PostgreSQL & JWT Security"}

