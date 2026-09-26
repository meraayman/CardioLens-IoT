"""CardioLens-IoT cloud service.

Implements the three IoT-cloud modules of the framework:
  1. Data collection & storage  -> every received ECG is logged to records.jsonl
  2. ECG analysis               -> EfficientNetB0 classification (+ optional Grad-CAM)
  3. Disease alert              -> alert level per prediction, optional webhook notification

Run (from the repository root):
    uvicorn iot.cloud.api:app --host 0.0.0.0 --port 8000

Environment variables:
    CARDIOLENS_MODEL       path to the .keras model (default: models/ecg_efficientnetb0.keras)
    CARDIOLENS_RECORDS     path of the JSONL record store (default: records.jsonl)
    CARDIOLENS_WEBHOOK     optional URL that receives a JSON POST for every critical/warning alert
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
import uuid
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from cardiolens import __version__
from cardiolens.config import DEFAULT_MODEL_PATH
from cardiolens.inference import ECGClassifier, load_image

MODEL_PATH   = os.getenv("CARDIOLENS_MODEL", DEFAULT_MODEL_PATH)
RECORDS_PATH = os.getenv("CARDIOLENS_RECORDS", "records.jsonl")
WEBHOOK_URL  = os.getenv("CARDIOLENS_WEBHOOK")

app = FastAPI(title="CardioLens-IoT Cloud", version=__version__,
              description="ECG analysis and disease-alert service for IoT cardiac monitoring.")


@lru_cache(maxsize=1)
def classifier() -> ECGClassifier:
    return ECGClassifier(MODEL_PATH)


def store_record(record: dict) -> None:
    with open(RECORDS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def send_alert(record: dict) -> bool:
    """Disease-alert module: notify an external system (dashboard, SMS gateway, ...) via webhook."""
    if not WEBHOOK_URL or record["prediction"]["alert_level"] == "none":
        return False
    req = urllib.request.Request(WEBHOOK_URL, data=json.dumps(record).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as exc:  # the alert failure must never block the analysis response
        print("Alert delivery failed:", exc)
        return False


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "model": MODEL_PATH}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), patient_id: str = Form("unknown"),
                  device_id: str = Form("unknown"), explain: bool = Form(False)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        image = load_image(data)
    except Exception:
        raise HTTPException(415, "Could not decode the image. Send a JPG or PNG ECG image.")

    t0 = time.perf_counter()
    clf = classifier()
    pred = clf.predict(image)
    record = {
        "record_id": str(uuid.uuid4()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "patient_id": patient_id,
        "device_id": device_id,
        "prediction": pred.to_dict(),
        "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    store_record(record)
    record["alert_sent"] = send_alert(record)

    if explain:
        png = clf.overlay_png(image, clf.gradcam(image))
        record["gradcam_png_base64"] = base64.b64encode(png).decode()
    return record


@app.get("/records")
def records(patient_id: str | None = None, limit: int = 50) -> list:
    """Physician dashboard feed: most recent analyses, optionally for one patient."""
    if not os.path.exists(RECORDS_PATH):
        return []
    with open(RECORDS_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if patient_id:
        rows = [r for r in rows if r["patient_id"] == patient_id]
    return rows[-limit:][::-1]
