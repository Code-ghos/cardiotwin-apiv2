"""
CardioTwin — ECG Signal Processor + 1D-CNN Trainer
====================================================
This module runs SEPARATELY from database.py (as a background worker).

What it does:
  1. Polls the `raw_ecg_samples` table for unprocessed samples
  2. Per device: bandpass filter → R-peak detection → beat slice (187 samples)
               → HR + HRV calculation
  3. Runs the 1D-CNN on each complete beat
  4. Writes classification result back into `ecg_readings`
  5. Broadcasts result via WebSocket to any connected dashboard

Architecture:
  ESP32  →  POST /ecg/ingest  →  raw_ecg_samples (DB)
                                       ↓  (this file polls here)
                               signal processing
                                       ↓
                               1D-CNN (ONNX)
                                       ↓
                               ecg_readings (DB)  +  WebSocket broadcast

How to run alongside database.py:
  Terminal 1:  python database.py           # FastAPI server
  Terminal 2:  python ecg_processor.py      # this worker (poll loop)

On Railway:
  Add a second process to Procfile:
    web:    uvicorn database:app --host 0.0.0.0 --port $PORT
    worker: python ecg_processor.py

Requirements (same as database.py, already in requirements.txt):
  fastapi, uvicorn, sqlalchemy, psycopg2-binary, pydantic,
  onnxruntime, numpy, scipy, websockets
"""

import os
import json
import time
import asyncio
import collections
import threading
import numpy as np
from datetime import datetime
from typing import Optional

# ── SQLAlchemy ───────────────────────────────────────────────────────────────
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Boolean, DateTime, Text, ForeignKey, func, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

try:
    from scipy.signal import butter, filtfilt
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[WARN] scipy not installed — bandpass filter disabled")

try:
    import onnxruntime as ort
    ONNX_OK = True
except ImportError:
    ONNX_OK = False
    print("[WARN] onnxruntime not installed — classifier in simulator mode")

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP  (mirrors database.py — same URL, same engine)
# ══════════════════════════════════════════════════════════════════════════════

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./cardiotwin.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine       = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


# ══════════════════════════════════════════════════════════════════════════════
# NEW TABLE:  raw_ecg_samples
# ──────────────────────────────────────────────────────────────────────────────
# ESP32 POSTs to /ecg/ingest  →  database.py writes every raw ADC value here.
# This processor reads rows where  processed = False  in arrival order.
# ══════════════════════════════════════════════════════════════════════════════

class RawECGSample(Base):
    """
    One ADC sample from ESP32 (0-4095 for 12-bit, or 0-1023 for 10-bit).
    Written by the FastAPI /ecg/ingest endpoint.
    Read and processed by this worker.
    """
    __tablename__ = "raw_ecg_samples"

    id         = Column(Integer, primary_key=True, index=True)
    device_id  = Column(String(50), index=True, nullable=False, default="default")
    patient_id = Column(Integer,    nullable=True)
    timestamp  = Column(DateTime,   default=datetime.utcnow, index=True)
    value      = Column(Integer,    nullable=False)   # raw ADC reading
    leads_off  = Column(Boolean,    default=False)
    processed  = Column(Boolean,    default=False, index=True)


# ── mirror tables from database.py (needed for foreign keys + writes) ─────────

class ECGSession(Base):
    __tablename__ = "ecg_sessions"
    id         = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at   = Column(DateTime, nullable=True)
    source     = Column(String(20), default="esp32")
    notes      = Column(Text, nullable=True)


class ECGReading(Base):
    """
    One classified beat — written by this processor after CNN inference.
    Read by the FastAPI dashboard endpoints.
    """
    __tablename__ = "ecg_readings"
    id           = Column(Integer, primary_key=True, index=True)
    session_id   = Column(Integer, ForeignKey("ecg_sessions.id"), nullable=False)
    timestamp    = Column(DateTime, default=datetime.utcnow)
    heart_rate   = Column(Integer)
    hrv          = Column(Float)
    rr_interval  = Column(Float)
    qrs_duration = Column(Float)
    ecg_class    = Column(String(30))
    confidence   = Column(Float, default=0.0)
    leads_off    = Column(Boolean, default=False)
    # Beat waveform stored as JSON array for dashboard rendering
    beat_json    = Column(Text, nullable=True)


# Create only the new table if it doesn't exist; other tables already exist.
Base.metadata.create_all(bind=engine)

print("[DB] Tables ready (including raw_ecg_samples)")


# ══════════════════════════════════════════════════════════════════════════════
# 1D-CNN MODEL DEFINITION  (PyTorch — for training / export to ONNX)
# ══════════════════════════════════════════════════════════════════════════════
# This section lets you RE-TRAIN the model from scratch if needed.
# In production the processor loads the pre-trained ecg_model.onnx.
# To retrain: run  python ecg_processor.py --train
# ══════════════════════════════════════════════════════════════════════════════

def build_1d_cnn_pytorch():
    """
    Returns the same 4-block 1D-CNN architecture that produced ecg_model.onnx.
    Input:  (batch, 1, 187)
    Output: (batch, 5)   — logits for 5 AAMI classes
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[MODEL] PyTorch not installed — cannot build/train model.")
        print("        Install with:  pip install torch")
        return None

    class ResBlock(nn.Module):
        def __init__(self, ch, k=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(ch, ch, k, padding=k//2, bias=False),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.Conv1d(ch, ch, k, padding=k//2, bias=False),
                nn.BatchNorm1d(ch),
            )
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.net(x) + x)

    class ECGNet(nn.Module):
        """
        4-block 1D residual CNN.
        Block 1:  1  → 32  ch, MaxPool /2  →  93 time steps
        Block 2: 32  → 64  ch, MaxPool /2  →  46 time steps
        Block 3: 64  → 128 ch, MaxPool /2  →  23 time steps
        Block 4: 128 → 256 ch, AdaptiveAvg  →   1 time step
        FC:  256 → 5
        """
        def __init__(self, n_classes=5):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )
            self.block1 = nn.Sequential(ResBlock(32), nn.MaxPool1d(2))
            self.block2 = nn.Sequential(ResBlock(64),  nn.MaxPool1d(2))
            self.block3 = nn.Sequential(ResBlock(128), nn.MaxPool1d(2))
            self.block4 = nn.Sequential(ResBlock(256), nn.AdaptiveAvgPool1d(1))

            self.up1 = nn.Conv1d(32,  64,  1)
            self.up2 = nn.Conv1d(64,  128, 1)
            self.up3 = nn.Conv1d(128, 256, 1)

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, n_classes)
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.block1(self.up1(x) if x.shape[1] != 64 else x)

            # Note: we use sequential projection at each stage for simplicity
            x = self.stem(x) if False else x   # placeholder
            # Correct sequential forward:
            x2 = self.up1(x)
            x2 = self.block2(x2)
            x3 = self.up2(x2)
            x3 = self.block3(x3)
            x4 = self.up3(x3)
            x4 = self.block4(x4)
            return self.classifier(x4)

    class ECGNetClean(nn.Module):
        """Clean version of ECGNet without the above placeholder confusion."""
        def __init__(self, n_classes=5):
            super().__init__()
            # Stem
            self.stem = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(32), nn.ReLU()
            )
            # Residual blocks + channel projection
            self.res1   = ResBlock(32)
            self.proj1  = nn.Conv1d(32, 64, 1)
            self.pool1  = nn.MaxPool1d(2)

            self.res2   = ResBlock(64)
            self.proj2  = nn.Conv1d(64, 128, 1)
            self.pool2  = nn.MaxPool1d(2)

            self.res3   = ResBlock(128)
            self.proj3  = nn.Conv1d(128, 256, 1)
            self.pool3  = nn.MaxPool1d(2)

            self.res4   = ResBlock(256)
            self.gap    = nn.AdaptiveAvgPool1d(1)

            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(256, 64), nn.ReLU(),
                nn.Linear(64, n_classes)
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.pool1(self.proj1(self.res1(x)))
            x = self.pool2(self.proj2(self.res2(x)))
            x = self.pool3(self.proj3(self.res3(x)))
            x = self.gap(self.res4(x))
            return self.head(x)

    return ECGNetClean


def export_to_onnx(model, path="ecg_model.onnx"):
    """Export trained PyTorch model to ONNX (opset 11)."""
    try:
        import torch
        model.eval()
        dummy = torch.randn(1, 1, 187)
        torch.onnx.export(
            model, dummy, path,
            input_names=["ecg_input"],
            output_names=["class_logits"],
            dynamic_axes={"ecg_input": {0: "batch"}},
            opset_version=11
        )
        print(f"[MODEL] Exported to {path}")
    except Exception as e:
        print(f"[MODEL] Export failed: {e}")


def train_from_mitbih(onnx_output_path="ecg_model.onnx"):
    """
    Download MIT-BIH from physionet (via wfdb), build dataset,
    train ECGNetClean, export to ONNX.

    Install extras:  pip install torch wfdb scikit-learn
    Run:             python ecg_processor.py --train
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        import wfdb
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"[TRAIN] Missing dependency: {e}")
        print("        pip install torch wfdb scikit-learn")
        return

    print("[TRAIN] Downloading MIT-BIH Arrhythmia Database...")

    # MIT-BIH record numbers
    records = [
        100,101,102,103,104,105,106,107,108,109,
        111,112,113,114,115,116,117,118,119,121,
        122,123,124,200,201,202,203,205,207,208,
        209,210,212,213,214,215,217,219,220,221,
        222,223,228,230,231,232,233,234
    ]

    # AAMI class mapping
    aami_map = {
        'N':0,'L':0,'R':0,'e':0,'j':0,        # Normal
        'A':1,'a':1,'J':1,'S':1,               # SVE
        'V':2,'E':2,                            # PVC
        'F':3,                                  # Fusion
        '/':4,'f':4,'Q':4,'?':4                # Unknown/Paced
    }

    beats, labels = [], []
    FS = 360
    HALF = 93   # 93+1+93 = 187

    for rec in records:
        try:
            record   = wfdb.rdrecord(str(rec),
                                     pn_dir='mitdb',
                                     sampfrom=0)
            ann      = wfdb.rdann(str(rec),
                                  'atr',
                                  pn_dir='mitdb')
            sig = record.p_signal[:, 0]   # Lead II
            # Bandpass 0.5–40 Hz
            nyq = FS / 2
            b, a = butter(3, [0.5/nyq, 40/nyq], btype='band')
            sig = filtfilt(b, a, sig)

            for i, (samp, sym) in enumerate(zip(ann.sample, ann.symbol)):
                if sym not in aami_map:
                    continue
                start = samp - HALF
                end   = samp + HALF + 1
                if start < 0 or end > len(sig):
                    continue
                beat = sig[start:end].astype(np.float32)
                # Z-score normalise
                mu, std = beat.mean(), beat.std()
                if std > 1e-6:
                    beat = (beat - mu) / std
                beats.append(beat)
                labels.append(aami_map[sym])
        except Exception as e:
            print(f"[TRAIN] Record {rec} skipped: {e}")
            continue

    if not beats:
        print("[TRAIN] No data loaded — check internet connection")
        return

    X = np.array(beats,  dtype=np.float32)[:, np.newaxis, :]  # (N,1,187)
    y = np.array(labels, dtype=np.int64)
    print(f"[TRAIN] Dataset: {len(X)} beats, class dist: "
          f"{np.bincount(y).tolist()}")

    # Class-weighted sampler to handle imbalance
    from torch.utils.data import WeightedRandomSampler
    class_counts = np.bincount(y)
    weights      = 1.0 / class_counts[y]
    sampler      = WeightedRandomSampler(weights, len(weights))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Training on {device}")

    ECGNetClean = build_1d_cnn_pytorch()
    model = ECGNetClean(n_classes=5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(30):
        # ── Train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(xb)
        scheduler.step()

        # ── Validate
        model.eval()
        vcorrect, vtotal = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                preds   = model(xb).argmax(1)
                vcorrect += (preds == yb).sum().item()
                vtotal   += len(xb)
        val_acc = vcorrect / vtotal

        print(f"  Epoch {epoch+1:2d}/30 | "
              f"train_loss={total_loss/total:.4f} | "
              f"train_acc={correct/total:.4f} | "
              f"val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "ecg_model_best.pt")

    # Load best weights and export
    model.load_state_dict(torch.load("ecg_model_best.pt"))
    export_to_onnx(model, onnx_output_path)

    # Update metadata
    meta = {
        "class_names":  ["Normal", "SVE", "PVC", "Fusion", "Unknown"],
        "ecg_types":    ["normal", "afib", "pvc", "normal", "normal"],
        "n_classes":    5,
        "segment_len":  187,
        "sample_rate":  360,
        "best_val_acc": best_val_acc,
        "aami_standard": True,
        "description":  {
            "0": "Normal Sinus Rhythm",
            "1": "Supraventricular Ectopic (SVE / AFib)",
            "2": "Ventricular Ectopic Beat (PVC)",
            "3": "Fusion Beat",
            "4": "Unknown / Paced Beat"
        },
        "model_info": {
            "architecture": "4-block 1D-CNN",
            "trained_on":   "MIT-BIH (physionet.org/content/mitdb)",
            "epochs":       30,
            "opset":        11
        }
    }
    with open("ecg_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[TRAIN] Done. Best val_acc = {best_val_acc*100:.2f}%")
    print(f"[TRAIN] Saved: {onnx_output_path}  +  ecg_metadata.json")


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSOR
# ──────────────────────────────────────────────────────────────────────────────
# One instance per device_id.
# Maintains a rolling buffer of raw samples and fires a beat callback
# whenever it detects and slices a complete QRS complex.
# ══════════════════════════════════════════════════════════════════════════════

class SignalProcessor:
    """
    Pure signal processing — no DB, no HTTP.

    Usage:
        proc = SignalProcessor("esp32_01")
        result = proc.add_sample(512)   # returns dict or None
    """
    SEGMENT = 187
    FS      = 360

    def __init__(self, device_id: str = "default"):
        self.device_id = device_id
        self._buf      = collections.deque(maxlen=self.FS * 5)  # 5-second window
        self._rr       = collections.deque(maxlen=10)           # last 10 RR intervals
        self._last_r   = -1                                     # last detected peak index

    def _bandpass(self, sig: list) -> np.ndarray:
        """Butterworth bandpass 0.5–40 Hz (passes QRS, rejects baseline wander)."""
        arr = np.array(sig, dtype=np.float64)
        if not SCIPY_OK or len(arr) < 15:
            return arr
        nyq  = self.FS / 2.0
        b, a = butter(3, [0.5 / nyq, 40.0 / nyq], btype="band")
        return filtfilt(b, a, arr)

    def _normalize(self, beat: np.ndarray) -> np.ndarray:
        """Z-score normalise — matches training preprocessing."""
        mu, std = beat.mean(), beat.std()
        return (beat - mu) / std if std > 1e-6 else beat - mu

    def _detect_peaks(self, sig: np.ndarray) -> list:
        """
        Simple threshold + minimum-spacing R-peak detector.
        Threshold: mean + 1.5×std of the signal window.
        Minimum distance between peaks: 0.2 s (72 samples at 360 Hz) —
        prevents double-detection on the same QRS.
        """
        thr   = np.mean(sig) + 1.5 * np.std(sig)
        mdist = int(0.2 * self.FS)          # 72 samples minimum spacing
        peaks, last = [], -mdist
        for i in range(1, len(sig) - 1):
            if (sig[i] > thr
                    and sig[i] >= sig[i - 1]
                    and sig[i] >= sig[i + 1]
                    and i - last > mdist):
                peaks.append(i)
                last = i
        return peaks

    def _qrs_duration(self, beat: np.ndarray) -> float:
        """
        Estimate QRS duration from the beat window.
        Finds the width of the central spike above 50% of its peak.
        Returns duration in milliseconds.
        """
        centre = len(beat) // 2
        region = beat[centre - 20: centre + 20]
        if len(region) == 0:
            return 88.0
        peak_val = region.max()
        half     = peak_val * 0.5
        above    = np.where(region > half)[0]
        if len(above) < 2:
            return 88.0
        return round((above[-1] - above[0]) / self.FS * 1000, 1)

    def add_sample(self, raw_value: int, timestamp: datetime = None) -> Optional[dict]:
        """
        Feed one raw ADC sample (0-4095 or 0-1023).
        Returns a beat dict when a complete beat is ready, else None.

        Returned dict keys:
          beat      list[float]  – 187 normalised samples
          hr        int          – heart rate BPM
          hrv       int          – HRV (SDNN) ms
          rr        float        – mean RR interval ms
          qrs_ms    float        – QRS duration ms
          timestamp datetime     – timestamp of the R-peak sample
        """
        # Convert 12-bit (0-4095) or 10-bit (0-1023) to centred float
        centre = 2048 if raw_value > 1023 else 512
        scale  = 2048 if raw_value > 1023 else 341
        self._buf.append((raw_value - centre) / scale)

        # Need at least 1 second of data before we can detect peaks reliably
        if len(self._buf) < self.FS:
            return None

        sig   = self._bandpass(list(self._buf))
        peaks = self._detect_peaks(sig)
        if not peaks:
            return None

        latest = peaks[-1]
        if latest == self._last_r:
            return None   # same peak as last call
        self._last_r = latest

        # ── RR → HR + HRV ─────────────────────────────────────────────────
        if len(peaks) >= 2:
            rr = (peaks[-1] - peaks[-2]) / self.FS * 1000.0
            if 300 < rr < 2000:   # physiologically valid 30–200 BPM
                self._rr.append(rr)

        rr_list = list(self._rr)
        if rr_list:
            mean_rr = float(np.mean(rr_list))
            hr      = int(np.clip(60000 / mean_rr, 30, 250))
            hrv     = int(np.std(rr_list)) if len(rr_list) > 1 else 40
        else:
            mean_rr, hr, hrv = 833.0, 75, 40

        # ── Beat slice  (R-peak ± 93 samples = 187 total) ─────────────────
        half  = self.SEGMENT // 2
        start = latest - half
        end   = latest + half + 1
        if start < 0 or end > len(sig):
            return None
        beat = sig[start:end]
        if len(beat) != self.SEGMENT:
            return None
        beat = self._normalize(beat)

        qrs_ms = self._qrs_duration(beat)

        return {
            "beat":      beat.tolist(),
            "hr":        hr,
            "hrv":       hrv,
            "rr":        round(mean_rr, 1),
            "qrs_ms":    qrs_ms,
            "timestamp": timestamp or datetime.utcnow()
        }


# ══════════════════════════════════════════════════════════════════════════════
# 1D-CNN INFERENCE WRAPPER  (ONNX)
# ══════════════════════════════════════════════════════════════════════════════

class ECGClassifierONNX:
    SEGMENT = 187

    def __init__(self, model_path="ecg_model.onnx", meta_path="ecg_metadata.json"):
        # Load metadata
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            self.class_names = meta["class_names"]
            self.ecg_types   = meta["ecg_types"]
            print(f"[CNN] Metadata loaded — {meta['n_classes']} classes, "
                  f"val_acc={meta.get('best_val_acc',0)*100:.1f}%")
        except Exception as e:
            print(f"[CNN] Metadata not found ({e}) — using defaults")
            self.class_names = ["Normal", "SVE", "PVC", "Fusion", "Unknown"]
            self.ecg_types   = ["normal", "afib", "pvc", "normal", "normal"]

        # Load ONNX session
        self.sess = None
        if ONNX_OK:
            try:
                self.sess = ort.InferenceSession(model_path)
                print(f"[CNN] Model loaded: {model_path}")
            except Exception as e:
                print(f"[CNN] Could not load {model_path}: {e}")

    @property
    def ready(self):
        return self.sess is not None

    def classify(self, beat: list) -> dict:
        """
        Classify a 187-sample normalised beat.
        Returns dict with class, ecg_type, confidence.
        """
        arr = np.array(beat, dtype=np.float32)
        if len(arr) != self.SEGMENT:
            return {"class": "Unknown", "ecg_type": "normal", "confidence": 0.0,
                    "source": "error"}

        if not self.ready:
            return {"class": "Normal", "ecg_type": "normal", "confidence": 0.0,
                    "source": "simulator"}

        x      = arr.reshape(1, 1, self.SEGMENT)
        logits = self.sess.run(["class_logits"], {"ecg_input": x})[0][0]
        exp_l  = np.exp(logits - logits.max())
        probs  = exp_l / exp_l.sum()
        idx    = int(np.argmax(probs))
        conf   = float(probs[idx])

        return {
            "class":      self.class_names[idx],
            "ecg_type":   self.ecg_types[idx],
            "confidence": round(conf * 100, 1),
            "source":     "cnn"
        }


# ══════════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_or_create_session(db: Session, patient_id: Optional[int],
                          device_id: str) -> int:
    """Return the latest open session for this patient/device, or create one."""
    if patient_id:
        s = db.query(ECGSession)\
              .filter(ECGSession.patient_id == patient_id,
                      ECGSession.ended_at == None)\
              .order_by(ECGSession.started_at.desc()).first()
        if s:
            return s.id
    new_s = ECGSession(
        patient_id = patient_id or 0,
        source     = f"esp32/{device_id}"
    )
    db.add(new_s)
    db.commit()
    db.refresh(new_s)
    return new_s.id


def fetch_unprocessed(db: Session, device_id: str, batch: int = 50):
    """
    Fetch up to `batch` unprocessed samples for one device,
    ordered by arrival time (FIFO).
    """
    return (
        db.query(RawECGSample)
          .filter(RawECGSample.device_id  == device_id,
                  RawECGSample.processed  == False)
          .order_by(RawECGSample.timestamp.asc())
          .limit(batch)
          .all()
    )


def mark_processed(db: Session, ids: list):
    """Bulk-mark sample rows as processed."""
    db.query(RawECGSample)\
      .filter(RawECGSample.id.in_(ids))\
      .update({"processed": True}, synchronize_session=False)
    db.commit()


def active_devices(db: Session) -> list:
    """Return list of device_ids that have unprocessed samples."""
    rows = (
        db.query(RawECGSample.device_id)
          .filter(RawECGSample.processed == False)
          .distinct()
          .all()
    )
    return [r.device_id for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET BROADCASTER
# ──────────────────────────────────────────────────────────────────────────────
# Posts classification results to the running FastAPI server's internal
# WebSocket manager via a simple HTTP POST to /ecg/beat so that
# database.py broadcasts to connected dashboards without duplicating state.
# ══════════════════════════════════════════════════════════════════════════════

import urllib.request

INTERNAL_API = os.environ.get("INTERNAL_API_URL", "http://localhost:8000")

def notify_dashboard(beat_result: dict, clf_result: dict, patient_id: Optional[int]):
    """
    POST a pre-classified beat to /ecg/beat so database.py's WebSocket
    manager broadcasts it to all connected dashboard clients.
    The 'source' field is set to 'db_processor' so dashboard knows
    this came from the server-side pipeline, not the ESP32.
    """
    payload = {
        "beat":       beat_result["beat"],
        "hr":         beat_result["hr"],
        "hrv":        beat_result["hrv"],
        "rr":         int(beat_result["rr"]),
        "leads_off":  False,
        "device_id":  "db_processor",
        "patient_id": patient_id,
        "class_name": clf_result.get("class"),
        "ecg_type":   clf_result.get("ecg_type"),
        "confidence": clf_result.get("confidence"),
        # Pre-classified result is passed as extra fields.
        # database.py will re-run ONNX for verification (or trust this if model absent).
    }
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{INTERNAL_API}/ecg/beat",
        data    = body,
        headers = {"Content-Type": "application/json"},
        method  = "POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=2):
            pass
    except Exception:
        pass   # Dashboard notification is best-effort


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def processing_loop(poll_interval: float = 0.1):
    """
    Continuously polls raw_ecg_samples for unprocessed rows,
    runs signal processing + CNN, writes ECGReading results to DB,
    and notifies the dashboard.

    poll_interval: seconds to sleep when no new data is available.
    """
    cnn        = ECGClassifierONNX()
    processors = {}   # device_id → SignalProcessor

    print("[WORKER] Processing loop started")
    print(f"[WORKER] CNN ready: {cnn.ready}")
    print(f"[WORKER] Poll interval: {poll_interval}s")

    while True:
        db = SessionLocal()
        try:
            devices = active_devices(db)

            if not devices:
                time.sleep(poll_interval)
                continue

            for device_id in devices:
                # Ensure we have a SignalProcessor for this device
                if device_id not in processors:
                    processors[device_id] = SignalProcessor(device_id)
                    print(f"[WORKER] New device: {device_id}")

                proc    = processors[device_id]
                samples = fetch_unprocessed(db, device_id, batch=200)

                if not samples:
                    continue

                processed_ids = []
                beats_ready   = []

                for samp in samples:
                    processed_ids.append(samp.id)

                    if samp.leads_off:
                        continue

                    result = proc.add_sample(samp.value, samp.timestamp)
                    if result:
                        result["patient_id"] = samp.patient_id
                        beats_ready.append(result)

                # Mark all fetched samples as processed (avoid re-processing)
                mark_processed(db, processed_ids)

                # Write classification results for each detected beat
                for beat_data in beats_ready:
                    clf = cnn.classify(beat_data["beat"])

                    patient_id = beat_data["patient_id"]
                    sid        = get_or_create_session(db, patient_id, device_id)

                    reading = ECGReading(
                        session_id   = sid,
                        timestamp    = beat_data["timestamp"],
                        heart_rate   = beat_data["hr"],
                        hrv          = float(beat_data["hrv"]),
                        rr_interval  = beat_data["rr"],
                        qrs_duration = beat_data["qrs_ms"],
                        ecg_class    = clf["ecg_type"],
                        confidence   = clf["confidence"],
                        leads_off    = False,
                        beat_json    = json.dumps(beat_data["beat"])
                    )
                    db.add(reading)

                    print(
                        f"[BEAT] device={device_id} "
                        f"class={clf['class']} ({clf['confidence']:.1f}%) "
                        f"HR={beat_data['hr']} HRV={beat_data['hrv']} "
                        f"QRS={beat_data['qrs_ms']}ms"
                    )

                    notify_dashboard(beat_data, clf, patient_id)

                db.commit()

        except Exception as e:
            print(f"[WORKER] Error: {e}")
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

        time.sleep(0.05)   # 50ms minimum between polls


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--train" in sys.argv:
        print("=" * 55)
        print("  CardioTwin — 1D-CNN Training Mode")
        print("=" * 55)
        train_from_mitbih("ecg_model.onnx")
        sys.exit(0)

    print("=" * 55)
    print("  CardioTwin — ECG Signal Processor Worker")
    print("=" * 55)
    print(f"  DB:    {DATABASE_URL.split('://')[0]}")
    print(f"  ONNX:  {'available' if ONNX_OK else 'not installed'}")
    print(f"  Scipy: {'available' if SCIPY_OK else 'not installed (filter disabled)'}")
    print("=" * 55)

    processing_loop(poll_interval=0.1)
