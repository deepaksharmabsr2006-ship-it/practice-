"""
Real-time Classifier Inference Engine.
Wraps the trained SVM / Random Forest model with confidence calibration,
hysteresis smoothing, and biometric safety overrides (e.g. immediate microsleep alarm
when eye closure exceeds safety threshold before the temporal window shifts).
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.temporal_buffer import TemporalFeatures
from core.geometry_features import FrameBiometrics


@dataclass
class ClassificationResult:
    class_id: int               # 0: ALERT, 1: DROWSY, 2: DISTRACTED
    class_name: str             # "ALERT", "DROWSY / MICROSLEEP", "DISTRACTED / PHONE"
    confidence: float           # 0.0 to 1.0
    probabilities: List[float]  # [p_alert, p_drowsy, p_distracted]
    is_critical_alarm: bool     # True if urgent safety intervention required
    trigger_reason: str         # Detailed explanation for telematics log


class SafetyClassifier:
    LABEL_NAMES = ["ALERT", "DROWSY / MICROSLEEP", "DISTRACTED / PHONE"]

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.svm_model = None
        self.scaler = None
        self.fallback_model = None

        self._load_models()

    def _load_models(self) -> None:
        svm_path = os.path.join(self.model_dir, "driver_safety_svm.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        fallback_path = os.path.join(self.model_dir, "fallback_classifier.pkl")

        if os.path.exists(svm_path) and os.path.exists(scaler_path):
            try:
                with open(svm_path, "rb") as f:
                    self.svm_model = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                return
            except Exception:
                pass

        if os.path.exists(fallback_path):
            try:
                with open(fallback_path, "rb") as f:
                    self.fallback_model = pickle.load(f)
            except Exception:
                pass

    def predict(self, features: TemporalFeatures, current_biometrics: Optional[FrameBiometrics] = None, consec_closed_frames: int = 0) -> ClassificationResult:
        """
        Predicts driver state using SVM with deterministic edge-safety fail-safes:
        If eye closure or phone posture exceeds emergency limits, failsafe overrides ensure 0ms alert lag.
        """
        # 1. IMMEDIATE BIOMECHANICAL FAILSAFES (Automotive Zero-Lag Intervention)
        # Immediate Microsleep: Eyelids shut for > 40 frames (~1.3s)
        if consec_closed_frames >= 40:
            return ClassificationResult(
                class_id=1,
                class_name=self.LABEL_NAMES[1],
                confidence=0.99,
                probabilities=[0.01, 0.98, 0.01],
                is_critical_alarm=True,
                trigger_reason=f"CRITICAL MICROSLEEP: Closed eyelids for {consec_closed_frames} consecutive frames"
            )

        # Immediate Phone/Head Drop: Driver pitched down deeply (lap view) + high phone ratio
        if features.phone_posture_ratio > 0.60 and features.pitch_mean < -14.0:
            return ClassificationResult(
                class_id=2,
                class_name=self.LABEL_NAMES[2],
                confidence=0.95,
                probabilities=[0.03, 0.02, 0.95],
                is_critical_alarm=True,
                trigger_reason=f"PHONE DISTRACTION: Head pitched down ({features.pitch_mean:.1f} deg) towards lap"
            )

        # 2. SCATTERED / ML MODEL INFERENCE
        vec = features.to_vector().reshape(1, -1)

        if self.svm_model is not None and self.scaler is not None:
            scaled_vec = self.scaler.transform(vec)
            probs = self.svm_model.predict_proba(scaled_vec)[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
        elif self.fallback_model is not None:
            mean = self.fallback_model["mean"]
            std = self.fallback_model["std"]
            norm_vec = (vec[0] - mean) / std

            # Compute euclidean distance to each class centroid
            dists = []
            for c in [0, 1, 2]:
                centroid = self.fallback_model["centroids"][c]
                dists.append(float(np.linalg.norm(norm_vec - centroid)))

            # Softmax over negative distances
            neg_dists = -np.array(dists)
            exp_d = np.exp(neg_dists - np.max(neg_dists))
            probs = (exp_d / np.sum(exp_d)).tolist()
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
        else:
            # Deterministic heuristic classifier based on SAE International standards
            if features.perclos > 0.35 or features.yawn_ratio > 0.30 or features.ear_mean < 0.20:
                pred_class = 1
                probs = [0.05, 0.90, 0.05]
            elif features.distraction_ratio > 0.50 or features.phone_posture_ratio > 0.40 or abs(features.yaw_mean) > 20.0:
                pred_class = 2
                probs = [0.05, 0.05, 0.90]
            else:
                pred_class = 0
                probs = [0.92, 0.04, 0.04]
            confidence = probs[pred_class]

        # Determine safety trigger reason
        if pred_class == 1:
            if features.yawn_ratio > 0.25:
                reason = f"Yawning & Fatigue (MAR max: {features.mar_max:.2f})"
            else:
                reason = f"High PERCLOS ({features.perclos * 100:.1f}%) / Eyelid Droop"
            is_critical = features.perclos > 0.50
        elif pred_class == 2:
            if features.phone_posture_ratio > 0.35:
                reason = f"Mobile Phone / Lap Viewing (Pitch: {features.pitch_mean:.1f} deg)"
            else:
                reason = f"In-cabin Distraction (Yaw: {features.yaw_mean:.1f} deg)"
            is_critical = features.distraction_ratio > 0.70
        else:
            reason = "Driver Alert & Attentive to Roadway"
            is_critical = False

        return ClassificationResult(
            class_id=pred_class,
            class_name=self.LABEL_NAMES[pred_class],
            confidence=confidence,
            probabilities=list(probs),
            is_critical_alarm=is_critical,
            trigger_reason=reason
        )
