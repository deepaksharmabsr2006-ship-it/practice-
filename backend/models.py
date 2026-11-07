"""
SQLAlchemy ORM Data Models for Fleet Telematics & Driver Safety.
Defines schemas for:
- Driver: Fleet driver registry with CDL license, safety rating, avatar
- Trip: Operational commercial trips with start/end time, distance, vehicle ID
- SafetyIncident: Specific infractions (Microsleep, Phone Distraction, Yawn, Distraction)
- TelemetrySnapshot: Periodic high-frequency biometric readings (EAR, MAR, Head Pose, Speed)
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.database import Base


class Driver(Base):
    __tablename__ = "drivers"

    id = Column(Integer, primary_key=True, index=True)
    driver_code = Column(String(50), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    license_number = Column(String(50), nullable=False)
    phone_number = Column(String(20), nullable=True)
    current_safety_score = Column(Float, default=100.0)  # Dynamic score (0-100)
    total_trips = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    trips = relationship("Trip", back_populates="driver", cascade="all, delete-orphan")


class Trip(Base):
    __tablename__ = "trips"

    id = Column(Integer, primary_key=True, index=True)
    trip_code = Column(String(60), unique=True, index=True, nullable=False)
    driver_id = Column(Integer, ForeignKey("drivers.id"), nullable=False)
    vehicle_vin = Column(String(50), nullable=False, default="TRUCK-VOLVO-FH16-01")
    origin = Column(String(100), default="Distribution Hub Alpha")
    destination = Column(String(100), default="Highway Logistics Terminal B")
    status = Column(String(20), default="ACTIVE")  # ACTIVE, COMPLETED, SUSPENDED
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    distance_km = Column(Float, default=0.0)
    avg_speed_kmh = Column(Float, default=78.5)
    trip_safety_score = Column(Float, default=100.0)

    # Relationships
    driver = relationship("Driver", back_populates="trips")
    incidents = relationship("SafetyIncident", back_populates="trip", cascade="all, delete-orphan")
    snapshots = relationship("TelemetrySnapshot", back_populates="trip", cascade="all, delete-orphan")


class SafetyIncident(Base):
    __tablename__ = "safety_incidents"

    id = Column(Integer, primary_key=True, index=True)
    trip_id = Column(Integer, ForeignKey("trips.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    event_type = Column(String(40), nullable=False)  # MICROSLEEP, PHONE_DISTRACTION, YAWN, DROWSY
    severity = Column(String(20), nullable=False)    # CRITICAL, WARNING, INFO
    duration_sec = Column(Float, nullable=False)
    ear_val = Column(Float, nullable=False)
    mar_val = Column(Float, nullable=False)
    pitch_deg = Column(Float, nullable=False)
    yaw_deg = Column(Float, nullable=False)
    speed_kmh = Column(Float, default=85.0)
    trigger_reason = Column(String(255), nullable=False)
    audio_alert_played = Column(Boolean, default=True)

    # Relationships
    trip = relationship("Trip", back_populates="incidents")


class TelemetrySnapshot(Base):
    __tablename__ = "telemetry_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    trip_id = Column(Integer, ForeignKey("trips.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    ear_avg = Column(Float, nullable=False)
    mar_val = Column(Float, nullable=False)
    pitch_deg = Column(Float, nullable=False)
    yaw_deg = Column(Float, nullable=False)
    roll_deg = Column(Float, nullable=False)
    perclos = Column(Float, default=0.0)
    driver_state = Column(String(30), default="ALERT")  # ALERT, DROWSY, DISTRACTED
    confidence = Column(Float, default=0.95)

    # Relationships
    trip = relationship("Trip", back_populates="snapshots")
