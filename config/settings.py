# Driver Fatigue & Distraction Safety Monitor - Global Configuration

from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class MonitorConfig:
    # --- Camera & Edge Video Stream ---
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    TARGET_FPS: int = 30

    # --- Geometric Biometric Thresholds ---
    # Eye Aspect Ratio (EAR)
    EAR_THRESHOLD: float = 0.22             # Below this value, eye is considered closed
    EAR_CONSEC_FRAMES_DROWSY: int = 20      # ~0.66s at 30 FPS: Prolonged eyelid closure / drowsiness
    EAR_CONSEC_FRAMES_MICROSLEEP: int = 45  # ~1.50s at 30 FPS: Dangerous microsleep event!

    # Mouth Aspect Ratio (MAR)
    MAR_THRESHOLD: float = 0.65             # Above this value, mouth is considered wide open (yawning)
    MAR_CONSEC_FRAMES_YAWN: int = 35        # ~1.16s at 30 FPS: Sustained yawning

    # Head Pose (Angles in degrees)
    PITCH_DOWN_THRESHOLD: float = -14.0     # Nodding off or looking down at phone
    PITCH_UP_THRESHOLD: float = 20.0        # Looking up / uncentered
    YAW_THRESHOLD: float = 22.0             # Looking left/right out window or distraction
    ROLL_THRESHOLD: float = 20.0            # Head tilt / sleeping against window
    POSE_CONSEC_FRAMES_DISTRACTED: int = 30 # ~1.0s at 30 FPS: looking away from road

    # --- Temporal Sliding Window (Features for ML) ---
    WINDOW_SIZE_SECONDS: float = 3.0        # Time window duration for temporal feature extraction
    SLIDING_STEP_FRAMES: int = 5            # Shift step for feature extraction

    # --- Alert & Hysteresis Settings ---
    ALERT_COOLDOWN_SECONDS: float = 2.5     # Minimum time between repeated audio alarms
    ENABLE_AUDIO: bool = True               # Toggle audio buzzer
    AUDIO_VOLUME: float = 1.0

    # --- Telematics & Telemetry Logging ---
    LOG_TELEMETRY: bool = True
    LOG_DIR: str = "logs"
    TRIP_LOG_FILENAME: str = "trip_telematics_events.jsonl"

    # --- Machine Learning Model Paths ---
    MODEL_DIR: str = "models"
    MODEL_PATH: str = "models/driver_safety_svm.joblib"
    SCALER_PATH: str = "models/scaler.joblib"

    # Class Label Mappings
    LABEL_ALERT: int = 0
    LABEL_DROWSY: int = 1
    LABEL_DISTRACTED: int = 2
    LABEL_NAMES: Tuple[str, ...] = ("ALERT", "DROWSY / MICROSLEEP", "DISTRACTED / PHONE")

config = MonitorConfig()
