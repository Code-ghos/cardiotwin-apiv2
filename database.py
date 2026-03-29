"""
CardioTwin - FastAPI backend
----------------------------
Endpoints cover:
  - /health
  - Patients CRUD
  - Sessions and recent history
  - Readings + risk records
  - /ecg/ingest (raw ADC samples) for the ESP32
  - /ecg/beat (classified beat push) for the worker
  - WebSocket broadcast for live dashboard updates

Defaults to SQLite when DATABASE_URL is unset.
"""

import os
import json
from datetime import datetime
from typing import List, Optional

from fastapi import (
    FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean, DateTime,
    Text, ForeignKey, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship


# ─── Database setup ──────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./cardiotwin.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── Models ──────────────────────────────────────────────────────────────────
class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(120), nullable=False, unique=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    sleep_hours = Column(Integer, default=7)
    exercise_days = Column(Integer, default=3)
    stress_level = Column(Integer, default=5)
    water_glasses = Column(Integer, default=6)
    smoking = Column(Boolean, default=False)
    diabetes = Column(Boolean, default=False)
    hypertension = Column(Boolean, default=False)
    family_history = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("ECGSession", back_populates="patient", cascade="all, delete")


class ECGSession(Base):
    __tablename__ = "ecg_sessions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    source = Column(String(30), default="esp32")
    notes = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="sessions")


class ECGReading(Base):
    __tablename__ = "ecg_readings"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("ecg_sessions.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    heart_rate = Column(Integer)
    hrv = Column(Float)
    rr_interval = Column(Float)
    qrs_duration = Column(Float)
    ecg_class = Column(String(30))
    confidence = Column(Float, default=0.0)
    leads_off = Column(Boolean, default=False)
    beat_json = Column(Text, nullable=True)


class RiskRecord(Base):
    __tablename__ = "risk_records"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("ecg_sessions.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    risk_score = Column(Float, default=0.0)
    risk_label = Column(String(30), default="unknown")
    hrv_contrib = Column(Float, default=0.0)
    sleep_contrib = Column(Float, default=0.0)
    stress_contrib = Column(Float, default=0.0)
    exercise_contrib = Column(Float, default=0.0)
    age_contrib = Column(Float, default=0.0)
    smoking_contrib = Column(Float, default=0.0)
    lifestyle_json = Column(Text, nullable=True)
    symptoms_json = Column(Text, nullable=True)


class RawECGSample(Base):
    """
    Written by /ecg/ingest, read by ecg_processor.py worker.
    """
    __tablename__ = "raw_ecg_samples"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(50), index=True, nullable=False, default="default")
    patient_id = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    value = Column(Integer, nullable=False)
    leads_off = Column(Boolean, default=False)
    processed = Column(Boolean, default=False, index=True)


Base.metadata.create_all(bind=engine)


# ─── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="CardioTwin API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── WebSocket manager ───────────────────────────────────────────────────────
class WSManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = WSManager()


# ─── Dependencies ────────────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Schemas ─────────────────────────────────────────────────────────────────
class PatientIn(BaseModel):
    name: str
    email: str
    age: Optional[int] = None
    gender: Optional[str] = None
    sleep_hours: int = 7
    exercise_days: int = 3
    stress_level: int = 5
    water_glasses: int = 6
    smoking: bool = False
    diabetes: bool = False
    hypertension: bool = False
    family_history: bool = False


class PatientOut(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int]
    gender: Optional[str]

    class Config:
        orm_mode = True


class SessionCreate(BaseModel):
    patient_id: int
    source: str = "esp32"
    notes: Optional[str] = None


class ReadingCreate(BaseModel):
    session_id: int
    heart_rate: Optional[int] = None
    hrv: Optional[float] = None
    rr_interval: Optional[float] = None
    qrs_duration: Optional[float] = None
    ecg_class: Optional[str] = None
    confidence: Optional[float] = 0.0
    leads_off: bool = False
    beat_json: Optional[str] = None


class RiskCreate(BaseModel):
    session_id: int
    risk_score: float
    risk_label: str
    hrv_contrib: float = 0.0
    sleep_contrib: float = 0.0
    stress_contrib: float = 0.0
    exercise_contrib: float = 0.0
    age_contrib: float = 0.0
    smoking_contrib: float = 0.0
    lifestyle_json: Optional[str] = None
    symptoms_json: Optional[str] = None


class ECGSample(BaseModel):
    device_id: str = Field(default="default", max_length=50)
    patient_id: Optional[int] = None
    value: int
    leads_off: bool = False


class ECGSampleBatch(BaseModel):
    samples: List[ECGSample]


class BeatPush(BaseModel):
    beat: List[float]
    hr: int
    hrv: float
    rr: float
    leads_off: bool = False
    device_id: str = "db_processor"
    patient_id: Optional[int] = None
    class_name: Optional[str] = None
    ecg_type: Optional[str] = None
    confidence: Optional[float] = None


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "db": DATABASE_URL.split("://")[0]}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keepalive; client sends pings
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.post("/patients", response_model=PatientOut)
def create_patient(payload: PatientIn, db: Session = Depends(get_db)):
    existing = db.query(Patient).filter(Patient.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")
    p = Patient(**payload.dict())
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@app.get("/patients", response_model=List[PatientOut])
def list_patients(db: Session = Depends(get_db)):
    return db.query(Patient).order_by(Patient.created_at.desc()).all()


@app.get("/patients/{pid}", response_model=PatientOut)
def get_patient(pid: int, db: Session = Depends(get_db)):
    p = db.query(Patient).get(pid)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    return p


@app.put("/patients/{pid}", response_model=PatientOut)
def update_patient(pid: int, payload: PatientIn, db: Session = Depends(get_db)):
    p = db.query(Patient).get(pid)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    for k, v in payload.dict().items():
        setattr(p, k, v)
    db.commit()
    db.refresh(p)
    return p


@app.post("/sessions")
def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    if not db.query(Patient).get(payload.patient_id):
        raise HTTPException(status_code=404, detail="Patient missing")
    s = ECGSession(**payload.dict())
    db.add(s)
    db.commit()
    db.refresh(s)
    return {"id": s.id, "started_at": s.started_at}


@app.get("/sessions/{pid}/history")
def session_history(pid: int, limit: int = 10, db: Session = Depends(get_db)):
    return (
        db.query(ECGSession)
        .filter(ECGSession.patient_id == pid)
        .order_by(ECGSession.started_at.desc())
        .limit(limit)
        .all()
    )


@app.post("/readings")
def add_reading(payload: ReadingCreate, db: Session = Depends(get_db)):
    r = ECGReading(**payload.dict())
    db.add(r)
    db.commit()
    return {"id": r.id}


@app.post("/risk")
def add_risk(payload: RiskCreate, db: Session = Depends(get_db)):
    rr = RiskRecord(**payload.dict())
    db.add(rr)
    db.commit()
    return {"id": rr.id}


@app.get("/analytics/{pid}")
def analytics(pid: int, db: Session = Depends(get_db)):
    p = db.query(Patient).get(pid)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")

    total_sessions = db.query(func.count(ECGSession.id)).filter(ECGSession.patient_id == pid).scalar()
    total_readings = (
        db.query(func.count(ECGReading.id))
        .join(ECGSession, ECGSession.id == ECGReading.session_id)
        .filter(ECGSession.patient_id == pid)
        .scalar()
    )
    latest_risk = (
        db.query(RiskRecord)
        .join(ECGSession, ECGSession.id == RiskRecord.session_id)
        .filter(ECGSession.patient_id == pid)
        .order_by(RiskRecord.created_at.desc())
        .first()
    )
    last_reading = (
        db.query(ECGReading.timestamp)
        .join(ECGSession, ECGSession.id == ECGReading.session_id)
        .filter(ECGSession.patient_id == pid)
        .order_by(ECGReading.timestamp.desc())
        .first()
    )

    return {
        "patient_name": p.name,
        "patient_age": p.age,
        "total_sessions": total_sessions or 0,
        "total_readings": total_readings or 0,
        "latest_risk_score": latest_risk.risk_score if latest_risk else None,
        "last_monitored": last_reading[0] if last_reading else None,
    }


@app.post("/ecg/ingest")
async def ingest_ecg(sample: ECGSample, db: Session = Depends(get_db)):
    raw = RawECGSample(
        device_id=sample.device_id,
        patient_id=sample.patient_id,
        value=sample.value,
        leads_off=sample.leads_off,
        processed=False,
    )
    db.add(raw)
    db.commit()

    await ws_manager.broadcast(
        {
            "type": "raw",
            "value": sample.value,
            "device_id": sample.device_id,
            "leads_off": sample.leads_off,
            "timestamp": raw.timestamp.isoformat(),
        }
    )

    return {"status": "queued", "id": raw.id}


@app.post("/ecg/ingest_batch")
async def ingest_ecg_batch(payload: ECGSampleBatch, db: Session = Depends(get_db)):
    if not payload.samples:
        raise HTTPException(status_code=400, detail="No samples provided")

    raws = [
        RawECGSample(
            device_id=s.device_id,
            patient_id=s.patient_id,
            value=s.value,
            leads_off=s.leads_off,
            processed=False,
        )
        for s in payload.samples
    ]

    db.add_all(raws)
    db.commit()

    # Broadcast only the latest sample to avoid flooding sockets
    latest = payload.samples[-1]
    await ws_manager.broadcast(
        {
            "type": "raw",
            "value": latest.value,
            "device_id": latest.device_id,
            "leads_off": latest.leads_off,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    return {"status": "queued", "count": len(raws)}


@app.post("/ecg/beat")
async def receive_beat(payload: BeatPush):
    await ws_manager.broadcast(
        {
            "type": "beat",
            "hr": payload.hr,
            "hrv": payload.hrv,
            "rr": payload.rr,
            "beat": payload.beat,
            "device_id": payload.device_id,
            "patient_id": payload.patient_id,
            "leads_off": payload.leads_off,
            "class": payload.class_name,
            "ecg_type": payload.ecg_type,
            "confidence": payload.confidence,
        }
    )
    return {"status": "broadcast"}
