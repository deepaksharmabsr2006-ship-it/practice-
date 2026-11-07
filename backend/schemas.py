"""
Pydantic Request & Response Data Schemas for REST API & Telematics Validation.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


# --- Telemetry Ingestion Schemas ---
class TelemetryIngest(BaseModel):
    trip_id: int = 1
    ear_avg: float = Field(..., ge=0.0, le=1.0, description="Eye Aspect Ratio")
    mar_val: float = Field(..., ge=0.0, le=2.0, description="Mouth Aspect Ratio")
    pitch_deg: float = Field(..., description="Head Pitch Angle (Down/Up)")
    yaw_deg: float = Field(..., description="Head Yaw Angle (Left/Right)")
    roll_deg: float = Field(..., description="Head Roll Angle (Tilt)")
    perclos: Optional[float] = 0.0
    driver_state: str = "ALERT"
    confidence: float = 0.95
    speed_kmh: Optional[float] = 85.0


# --- Safety Incident Schemas ---
class IncidentCreate(BaseModel):
    trip_id: int = 1
    event_type: str = Field(..., description="MICROSLEEP, PHONE_DISTRACTION, YAWN, DROWSY")
    severity: str = Field(..., description="CRITICAL, WARNING, INFO")
    duration_sec: float = Field(..., gt=0.0)
    ear_val: float
    mar_val: float
    pitch_deg: float
    yaw_deg: float
    speed_kmh: Optional[float] = 85.0
    trigger_reason: str
    audio_alert_played: bool = True


class IncidentResponse(IncidentCreate):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


# --- Driver Schemas ---
class DriverBase(BaseModel):
    driver_code: str
    full_name: str
    license_number: str
    phone_number: Optional[str] = None


class DriverCreate(DriverBase):
    pass


class DriverResponse(DriverBase):
    id: int
    current_safety_score: float
    total_trips: int
    created_at: datetime

    class Config:
        from_attributes = True


# --- Trip Schemas ---
class TripCreate(BaseModel):
    driver_id: int
    trip_code: str
    vehicle_vin: str = "TRUCK-VOLVO-FH16-01"
    origin: str = "Central Distribution Terminal"
    destination: str = "Highway Logistics Hub"


class TripResponse(BaseModel):
    id: int
    trip_code: str
    driver_id: int
    vehicle_vin: str
    origin: str
    destination: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    distance_km: float
    avg_speed_kmh: float
    trip_safety_score: float

    class Config:
        from_attributes = True


# --- Fleet Analytics Summary Schema ---
class FleetAnalyticsSummary(BaseModel):
    total_drivers: int
    active_trips: int
    total_incidents_today: int
    microsleep_count: int
    phone_distraction_count: int
    yawn_count: int
    fleet_safety_score_avg: float
    high_risk_drivers_count: int
