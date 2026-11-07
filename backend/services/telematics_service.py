"""
Telematics & Fleet Safety Analytics Business Logic Service.
Handles safety score deductions, seed initialization for demonstration,
and aggregate analytics generation.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from backend.models import Driver, Trip, SafetyIncident, TelemetrySnapshot
from backend.schemas import IncidentCreate, TelemetryIngest


def seed_initial_demo_data(db: Session) -> None:
    """Pre-populates demonstration fleet drivers and trips if database is empty."""
    if db.query(Driver).count() > 0:
        return

    # Seed 3 Commercial Fleet Drivers mapped to the Project Team Members
    drivers = [
        Driver(driver_code="DRV-901", full_name="Deepak Sharma (Lead / AIML)", license_number="DL-04-2023-8941", current_safety_score=96.5, total_trips=48),
        Driver(driver_code="DRV-902", full_name="Aastha Singh (Backend / Telematics)", license_number="DL-07-2022-1104", current_safety_score=94.0, total_trips=42),
        Driver(driver_code="DRV-903", full_name="Abhishek Shukla (Frontend / DevOps)", license_number="DL-12-2024-5582", current_safety_score=91.5, total_trips=36),
    ]
    db.add_all(drivers)
    db.commit()

    # Seed an active highway trip for Deepak Sharma
    d1 = db.query(Driver).filter_by(driver_code="DRV-901").first()
    trip = Trip(
        trip_code="TRIP-NH44-DELHI-MUMBAI",
        driver_id=d1.id,
        vehicle_vin="SCANIA-R500-HEAVY-TRUCK",
        origin="Delhi Industrial Hub Sector 18",
        destination="Mumbai Nhava Sheva Freight Corridor",
        status="ACTIVE",
        distance_km=420.8,
        avg_speed_kmh=82.4,
        trip_safety_score=95.0
    )
    db.add(trip)
    db.commit()

    # Seed initial telematics events for demonstration
    incidents = [
        SafetyIncident(
            trip_id=trip.id,
            timestamp=datetime.utcnow() - timedelta(minutes=45),
            event_type="YAWN",
            severity="WARNING",
            duration_sec=2.1,
            ear_val=0.24,
            mar_val=0.78,
            pitch_deg=-2.5,
            yaw_deg=1.0,
            speed_kmh=84.0,
            trigger_reason="Extended yawn aperture (MAR: 0.78)"
        ),
        SafetyIncident(
            trip_id=trip.id,
            timestamp=datetime.utcnow() - timedelta(minutes=18),
            event_type="PHONE_DISTRACTION",
            severity="WARNING",
            duration_sec=1.8,
            ear_val=0.29,
            mar_val=0.21,
            pitch_deg=-24.0,
            yaw_deg=8.0,
            speed_kmh=80.2,
            trigger_reason="Driver looking down towards console/lap (Pitch -24 deg)"
        )
    ]
    db.add_all(incidents)
    db.commit()


def calculate_trip_safety_score(incidents: List[SafetyIncident]) -> float:
    score = 100.0
    for inc in incidents:
        if inc.event_type == "MICROSLEEP":
            score -= 15.0
        elif inc.event_type == "PHONE_DISTRACTION":
            score -= 8.0
        elif inc.event_type == "YAWN":
            score -= 3.0
        elif inc.event_type == "DROWSY":
            score -= 5.0
    return max(0.0, min(100.0, score))


def record_incident(db: Session, inc_in: IncidentCreate) -> SafetyIncident:
    incident = SafetyIncident(
        trip_id=inc_in.trip_id,
        timestamp=datetime.utcnow(),
        event_type=inc_in.event_type,
        severity=inc_in.severity,
        duration_sec=inc_in.duration_sec,
        ear_val=inc_in.ear_val,
        mar_val=inc_in.mar_val,
        pitch_deg=inc_in.pitch_deg,
        yaw_deg=inc_in.yaw_deg,
        speed_kmh=inc_in.speed_kmh or 85.0,
        trigger_reason=inc_in.trigger_reason,
        audio_alert_played=inc_in.audio_alert_played
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)

    all_incidents = db.query(SafetyIncident).filter_by(trip_id=inc_in.trip_id).all()
    new_score = calculate_trip_safety_score(all_incidents)

    trip = db.query(Trip).filter_by(id=inc_in.trip_id).first()
    if trip:
        trip.trip_safety_score = new_score
        driver = db.query(Driver).filter_by(id=trip.driver_id).first()
        if driver:
            driver.current_safety_score = round((driver.current_safety_score * 0.8) + (new_score * 0.2), 1)
        db.commit()

    return incident


def get_fleet_summary(db: Session) -> Dict[str, Any]:
    total_drivers = db.query(Driver).count()
    active_trips = db.query(Trip).filter_by(status="ACTIVE").count()

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    incidents_today = db.query(SafetyIncident).filter(SafetyIncident.timestamp >= today_start).all()

    microsleep_count = sum(1 for i in incidents_today if i.event_type == "MICROSLEEP")
    phone_count = sum(1 for i in incidents_today if i.event_type == "PHONE_DISTRACTION")
    yawn_count = sum(1 for i in incidents_today if i.event_type == "YAWN")

    drivers = db.query(Driver).all()
    avg_score = sum(d.current_safety_score for d in drivers) / len(drivers) if drivers else 100.0
    high_risk = sum(1 for d in drivers if d.current_safety_score < 75.0)

    return {
        "total_drivers": total_drivers,
        "active_trips": active_trips,
        "total_incidents_today": len(incidents_today),
        "microsleep_count": microsleep_count,
        "phone_distraction_count": phone_count,
        "yawn_count": yawn_count,
        "fleet_safety_score_avg": round(avg_score, 1),
        "high_risk_drivers_count": high_risk
    }
