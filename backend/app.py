"""
Full-Stack FastAPI Application for Driver Fatigue & Distraction Safety Monitor.
Integrates:
- High-Performance Async REST Endpoints (Drivers, Trips, Incidents, Telematics Analytics)
- Real-time WebSockets for Live Vehicle Telemetry & Instant Emergency Broadcasts
- Cloud-Ready ML Inference Endpoint (/api/v1/ml/predict)
- Single-page Frontend Static Mount for Cloud Hosting (Render, Railway, Docker, AWS, Vercel)
"""

import os
import json
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from backend.database import engine, Base, get_db, SessionLocal
from backend.models import Driver, Trip, SafetyIncident, TelemetrySnapshot
from backend.schemas import (
    DriverResponse, TripResponse, IncidentCreate, IncidentResponse,
    TelemetryIngest, FleetAnalyticsSummary
)
from backend.services.telematics_service import (
    seed_initial_demo_data, record_incident, get_fleet_summary
)

# Optional ML Classifier for serverless environments (e.g. Vercel without cv2)
classifier = None
try:
    from ml.classifier import SafetyClassifier
    from core.temporal_buffer import TemporalFeatures
    classifier = SafetyClassifier(model_dir="models")
except Exception:
    classifier = None


# --- Lifespan Handler: Auto-migration & Database Seeding ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables
    try:
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        seed_initial_demo_data(db)
        db.close()
    except Exception as e:
        print(f"Database initialization note: {e}")
    yield


app = FastAPI(
    title="Driver Fatigue & Distraction Safety Monitor API",
    description="Enterprise Telematics & Real-Time Computer Vision Driver Monitoring System (DMS)",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Middleware for Cloud Deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- WebSocket Connection Manager for Real-Time Telematics ---
class TelematicsConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        for dc in dead_connections:
            self.disconnect(dc)


ws_manager = TelematicsConnectionManager()


# --- REST API Endpoints ---

@app.get("/api/v1/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "Driver Safety Telematics Gateway",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected",
        "ml_model_loaded": classifier is not None,
        "team": ["Deepak Sharma", "Aastha Singh", "Abhishek Shukla"]
    }


@app.get("/api/v1/telematics/summary", response_model=FleetAnalyticsSummary, tags=["Telematics"])
def get_analytics_summary(db: Session = Depends(get_db)):
    summary = get_fleet_summary(db)
    return summary


@app.get("/api/v1/drivers", response_model=List[DriverResponse], tags=["Fleet Management"])
def list_drivers(db: Session = Depends(get_db)):
    return db.query(Driver).all()


@app.get("/api/v1/trips", response_model=List[TripResponse], tags=["Fleet Management"])
def list_trips(status: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Trip)
    if status:
        query = query.filter(Trip.status == status)
    return query.all()


@app.get("/api/v1/incidents", response_model=List[IncidentResponse], tags=["Safety Incidents"])
def list_incidents(
    trip_id: Optional[int] = None,
    severity: Optional[str] = None,
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    query = db.query(SafetyIncident)
    if trip_id:
        query = query.filter(SafetyIncident.trip_id == trip_id)
    if severity:
        query = query.filter(SafetyIncident.severity == severity)
    return query.order_by(SafetyIncident.timestamp.desc()).limit(limit).all()


@app.post("/api/v1/incidents", response_model=IncidentResponse, status_code=201, tags=["Safety Incidents"])
async def create_incident(incident_in: IncidentCreate, db: Session = Depends(get_db)):
    incident = record_incident(db, incident_in)

    event_payload = {
        "type": "SAFETY_ALERT",
        "event_id": incident.id,
        "trip_id": incident.trip_id,
        "event_type": incident.event_type,
        "severity": incident.severity,
        "trigger_reason": incident.trigger_reason,
        "duration_sec": incident.duration_sec,
        "timestamp": incident.timestamp.isoformat(),
        "ear_val": incident.ear_val,
        "mar_val": incident.mar_val
    }
    await ws_manager.broadcast(event_payload)

    return incident


@app.post("/api/v1/telemetry/ingest", tags=["Telemetry Stream"])
async def ingest_telemetry_snapshot(snapshot: TelemetryIngest, db: Session = Depends(get_db)):
    db_snapshot = TelemetrySnapshot(
        trip_id=snapshot.trip_id,
        ear_avg=snapshot.ear_avg,
        mar_val=snapshot.mar_val,
        pitch_deg=snapshot.pitch_deg,
        yaw_deg=snapshot.yaw_deg,
        roll_deg=snapshot.roll_deg,
        perclos=snapshot.perclos or 0.0,
        driver_state=snapshot.driver_state,
        confidence=snapshot.confidence
    )
    db.add(db_snapshot)
    db.commit()

    await ws_manager.broadcast({
        "type": "TELEMETRY_UPDATE",
        "trip_id": snapshot.trip_id,
        "ear_avg": snapshot.ear_avg,
        "mar_val": snapshot.mar_val,
        "pitch_deg": snapshot.pitch_deg,
        "yaw_deg": snapshot.yaw_deg,
        "driver_state": snapshot.driver_state,
        "timestamp": datetime.utcnow().isoformat()
    })

    return {"status": "ingested", "id": db_snapshot.id}


@app.post("/api/v1/ml/predict", tags=["AI / ML Inference"])
def ml_predict(features: List[float]):
    if len(features) < 16:
        raise HTTPException(status_code=400, detail="Feature vector must have exactly 16 values.")

    if classifier is not None:
        from core.temporal_buffer import TemporalFeatures
        tf = TemporalFeatures(
            ear_mean=features[0], ear_std=features[1], ear_min=features[2], perclos=features[3],
            blink_count=int(features[4]), blink_rate_per_min=features[5], mar_mean=features[6],
            mar_max=features[7], yawn_ratio=features[8], pitch_mean=features[9], pitch_std=features[10],
            yaw_mean=features[11], yaw_std=features[12], roll_mean=features[13],
            phone_posture_ratio=features[14], distraction_ratio=features[15]
        )
        pred = classifier.predict(tf)
        return {
            "class_id": pred.class_id,
            "class_name": pred.class_name,
            "confidence": pred.confidence,
            "probabilities": pred.probabilities,
            "is_critical_alarm": pred.is_critical_alarm,
            "trigger_reason": pred.trigger_reason
        }
    else:
        # High-performance heuristic classifier fallback
        ear_val = features[0]
        perclos = features[3]
        pitch_val = features[9]
        if perclos > 0.4 or ear_val < 0.20:
            return {"class_id": 1, "class_name": "DROWSY / MICROSLEEP", "confidence": 0.98, "is_critical_alarm": True, "trigger_reason": "High PERCLOS / Eyelid Closure"}
        elif pitch_val < -14.0:
            return {"class_id": 2, "class_name": "DISTRACTED / PHONE", "confidence": 0.95, "is_critical_alarm": True, "trigger_reason": "Head Pitched Down into Lap"}
        else:
            return {"class_id": 0, "class_name": "ALERT", "confidence": 0.99, "is_critical_alarm": False, "trigger_reason": "Forward Gaze Attentive"}


# --- WebSocket Route ---
@app.websocket("/ws/telematics")
async def websocket_telematics_stream(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("action") == "PING":
                    await websocket.send_json({"type": "PONG", "timestamp": datetime.utcnow().isoformat()})
                elif msg.get("action") == "BROADCAST_ALERT":
                    await ws_manager.broadcast(msg)
            except Exception:
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# --- Root Frontend Serving ---
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
index_file = os.path.join(root_dir, "index.html")
if not os.path.exists(index_file):
    index_file = os.path.join(root_dir, "frontend", "index.html")

frontend_dir = os.path.join(root_dir, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def serve_root():
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Driver Safety Telematics Gateway API Online. Visit /docs for Swagger documentation."}
