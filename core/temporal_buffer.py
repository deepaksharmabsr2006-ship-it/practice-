"""
Temporal Sliding Window Buffer & Statistical Feature Aggregator.
Aggregates time-series facial biometrics over a 3-5 second sliding window
to extract rich features for Scikit-Learn SVM/Random Forest classification.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import time

from core.geometry_features import FrameBiometrics


@dataclass
class TemporalFeatures:
    ear_mean: float
    ear_std: float
    ear_min: float
    perclos: float               # Percentage of time eye closure >= threshold
    blink_count: int             # Total blinks detected in the window
    blink_rate_per_min: float    # Blinks/min extrapolation
    mar_mean: float
    mar_max: float
    yawn_ratio: float            # Fraction of window spent yawning
    pitch_mean: float
    pitch_std: float
    yaw_mean: float
    yaw_std: float
    roll_mean: float
    phone_posture_ratio: float   # Fraction of window head is pitched down (lap/phone view)
    distraction_ratio: float     # Fraction of window head turned away from roadway

    def to_vector(self) -> np.ndarray:
        """Converts to a 1D vector of shape (16,) for ML inference."""
        return np.array([
            self.ear_mean,
            self.ear_std,
            self.ear_min,
            self.perclos,
            float(self.blink_count),
            self.blink_rate_per_min,
            self.mar_mean,
            self.mar_max,
            self.yawn_ratio,
            self.pitch_mean,
            self.pitch_std,
            self.yaw_mean,
            self.yaw_std,
            self.roll_mean,
            self.phone_posture_ratio,
            self.distraction_ratio
        ], dtype=np.float32)

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
            "ear_mean", "ear_std", "ear_min", "perclos", "blink_count", "blink_rate_per_min",
            "mar_mean", "mar_max", "yawn_ratio", "pitch_mean", "pitch_std", "yaw_mean",
            "yaw_std", "roll_mean", "phone_posture_ratio", "distraction_ratio"
        ]


class TemporalBuffer:
    def __init__(self, window_size_seconds: float = 3.0, target_fps: int = 30):
        self.window_size_seconds = window_size_seconds
        self.max_frames = int(window_size_seconds * target_fps)
        self.timestamps: deque = deque(maxlen=self.max_frames)
        self.history: deque = deque(maxlen=self.max_frames)

        # Internal blink state machine
        self.is_blinking = False
        self.blink_start_time = 0.0
        self.consecutive_closed_frames = 0
        self.consecutive_yawn_frames = 0
        self.consecutive_phone_frames = 0

    def add_frame(self, biometrics: FrameBiometrics, timestamp: Optional[float] = None) -> None:
        """Appends a new frame measurement to the rolling temporal deque."""
        if timestamp is None:
            timestamp = time.time()

        self.timestamps.append(timestamp)
        self.history.append(biometrics)

        # Update consecutive frame counters for real-time edge triggers
        if biometrics.is_eyes_closed:
            self.consecutive_closed_frames += 1
        else:
            self.consecutive_closed_frames = 0

        if biometrics.is_yawning:
            self.consecutive_yawn_frames += 1
        else:
            self.consecutive_yawn_frames = 0

        if biometrics.is_looking_down_phone:
            self.consecutive_phone_frames += 1
        else:
            self.consecutive_phone_frames = 0

    def is_ready(self) -> bool:
        """Returns True once the window has accumulated enough samples (~1.0s minimum)."""
        return len(self.history) >= min(15, self.max_frames // 3)

    def extract_features(self) -> TemporalFeatures:
        """Computes statistical aggregations and time-series dynamics over the window."""
        if not self.history:
            return TemporalFeatures(
                ear_mean=0.30, ear_std=0.0, ear_min=0.30, perclos=0.0, blink_count=0,
                blink_rate_per_min=0.0, mar_mean=0.20, mar_max=0.20, yawn_ratio=0.0,
                pitch_mean=0.0, pitch_std=0.0, yaw_mean=0.0, yaw_std=0.0, roll_mean=0.0,
                phone_posture_ratio=0.0, distraction_ratio=0.0
            )

        n = len(self.history)
        ears = np.array([b.ear_avg for b in self.history], dtype=np.float32)
        mars = np.array([b.mar for b in self.history], dtype=np.float32)
        pitches = np.array([b.pitch for b in self.history], dtype=np.float32)
        yaws = np.array([b.yaw for b in self.history], dtype=np.float32)
        rolls = np.array([b.roll for b in self.history], dtype=np.float32)

        # 1. PERCLOS: fraction of frames where eyes are closed
        closed_flags = np.array([b.is_eyes_closed for b in self.history], dtype=np.float32)
        perclos = float(np.mean(closed_flags))

        # 2. Blink detection over time window (state transition edge detection)
        blink_count = 0
        in_blink = False
        for is_closed in closed_flags:
            if is_closed and not in_blink:
                in_blink = True
            elif not is_closed and in_blink:
                in_blink = False
                blink_count += 1

        # Extrapolate to blinks per minute
        window_duration = max(0.5, self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else self.window_size_seconds
        blink_rate_per_min = float((blink_count / window_duration) * 60.0)

        # 3. Yawning and MAR statistics
        yawn_flags = np.array([b.is_yawning for b in self.history], dtype=np.float32)
        yawn_ratio = float(np.mean(yawn_flags))

        # 4. Head pose and distraction statistics
        phone_flags = np.array([b.is_looking_down_phone for b in self.history], dtype=np.float32)
        phone_posture_ratio = float(np.mean(phone_flags))

        distract_flags = np.array([b.is_head_distracted for b in self.history], dtype=np.float32)
        distraction_ratio = float(np.mean(distract_flags))

        return TemporalFeatures(
            ear_mean=float(np.mean(ears)),
            ear_std=float(np.std(ears)),
            ear_min=float(np.min(ears)),
            perclos=perclos,
            blink_count=blink_count,
            blink_rate_per_min=blink_rate_per_min,
            mar_mean=float(np.mean(mars)),
            mar_max=float(np.max(mars)),
            yawn_ratio=yawn_ratio,
            pitch_mean=float(np.mean(pitches)),
            pitch_std=float(np.std(pitches)),
            yaw_mean=float(np.mean(yaws)),
            yaw_std=float(np.std(yaws)),
            roll_mean=float(np.mean(rolls)),
            phone_posture_ratio=phone_posture_ratio,
            distraction_ratio=distraction_ratio
        )
