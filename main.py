"""
Driver Fatigue & Distraction Safety Monitor - Main Edge Application.
Real-time vision pipeline incorporating MediaPipe FaceMesh, Biomechanical Feature Extraction,
Sliding Temporal Aggregation, Scikit-Learn SVM Inference, Audio Warnings, and Telematics Logging.
"""

import os
import sys
import time
import argparse
from typing import Optional, Tuple
import numpy as np
import cv2

from config.settings import config
from core.facial_landmarks import LandmarkExtractor, MEDIAPIPE_AVAILABLE
from core.geometry_features import GeometryEngine, FrameBiometrics
from core.temporal_buffer import TemporalBuffer
from ml.model_trainer import train_and_export_models
from ml.classifier import SafetyClassifier, ClassificationResult
from audio.alert_manager import AlertManager, AlertType
from telematics.event_logger import TelematicsLogger
from ui.hud_overlay import HUDOverlay


class MockDriverSimulator:
    """
    Renders an interactive synthetic driver face to simulate in-cabin telematics
    when a physical webcam is unavailable or for automated testing.
    """
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.mode = "alert"  # "alert", "microsleep", "phone", "yawn"
        self.phase = 0.0

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.phase = 0.0

    def generate_frame_and_biometrics(self) -> Tuple[np.ndarray, FrameBiometrics, dict]:
        self.phase += 0.1
        canvas = np.full((self.height, self.width, 3), 35, dtype=np.uint8)

        # Draw vehicle windshield and dashboard horizon
        cv2.line(canvas, (0, int(self.height * 0.75)), (self.width, int(self.height * 0.75)), (60, 60, 60), 2)
        cv2.rectangle(canvas, (0, int(self.height * 0.75)), (self.width, self.height), (25, 25, 25), -1)

        cx, cy = self.width // 2, int(self.height * 0.45)

        # Apply posture offsets based on simulated state
        if self.mode == "microsleep":
            # Head sagging down, eyes tightly shut
            cy += 35
            ear_val = 0.08
            mar_val = 0.22
            pitch_val = -18.0
            yaw_val = 2.0
            roll_val = 5.0
            eye_h = 2
            mouth_h = 6
        elif self.mode == "yawn":
            ear_val = 0.18
            mar_val = 0.85
            pitch_val = 4.0
            yaw_val = -3.0
            roll_val = 0.0
            eye_h = 5
            mouth_h = 36
        elif self.mode == "phone":
            # Driver looking down into lap/phone
            cy += 45
            cx += 20
            ear_val = 0.28
            mar_val = 0.20
            pitch_val = -26.0
            yaw_val = 14.0
            roll_val = 8.0
            eye_h = 10
            mouth_h = 8
            # Render mock phone in hand
            cv2.rectangle(canvas, (cx + 60, cy + 90), (cx + 120, cy + 190), (50, 50, 50), -1)
            cv2.rectangle(canvas, (cx + 65, cy + 95), (cx + 115, cy + 185), (200, 240, 255), -1)
            cv2.putText(canvas, "PHONE", (cx + 68, cy + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1)
        else:  # Normal Alert
            ear_val = 0.32 + 0.04 * np.sin(self.phase * 0.5)
            mar_val = 0.22
            pitch_val = -1.0 + 2.0 * np.sin(self.phase * 0.3)
            yaw_val = 0.0 + 3.0 * np.cos(self.phase * 0.2)
            roll_val = 0.0
            eye_h = int(14 * (ear_val / 0.32))
            mouth_h = 10

        # Draw Driver Avatar Face
        # Head oval
        cv2.ellipse(canvas, (cx, cy), (85, 115), 0, 0, 360, (190, 210, 230), -1)
        cv2.ellipse(canvas, (cx, cy), (85, 115), 0, 0, 360, (100, 100, 100), 2)

        # Eyes
        eye_y = cy - 20
        left_eye_center = (cx - 38, eye_y)
        right_eye_center = (cx + 38, eye_y)
        cv2.ellipse(canvas, left_eye_center, (20, max(2, eye_h)), 0, 0, 360, (40, 40, 40), -1)
        cv2.ellipse(canvas, right_eye_center, (20, max(2, eye_h)), 0, 0, 360, (40, 40, 40), -1)

        # Pupils (if eyes open)
        if eye_h > 4:
            cv2.circle(canvas, left_eye_center, 4, (20, 20, 180), -1)
            cv2.circle(canvas, right_eye_center, 4, (20, 20, 180), -1)

        # Mouth
        mouth_y = cy + 45
        cv2.ellipse(canvas, (cx, mouth_y), (30, max(3, mouth_h)), 0, 0, 360, (50, 40, 130), -1)

        # Mock subsets for HUD mesh rendering
        subsets = {
            'left_eye': np.array([
                [cx - 55, eye_y], [cx - 45, eye_y - eye_h], [cx - 30, eye_y - eye_h],
                [cx - 20, eye_y], [cx - 30, eye_y + eye_h], [cx - 45, eye_y + eye_h]
            ], dtype=np.float32),
            'right_eye': np.array([
                [cx + 20, eye_y], [cx + 30, eye_y - eye_h], [cx + 45, eye_y - eye_h],
                [cx + 55, eye_y], [cx + 45, eye_y + eye_h], [cx + 30, eye_y + eye_h]
            ], dtype=np.float32),
            'mouth': np.array([
                [cx - 30, mouth_y], [cx - 15, mouth_y - mouth_h], [cx, mouth_y - mouth_h],
                [cx + 15, mouth_y - mouth_h], [cx + 30, mouth_y], [cx + 15, mouth_y + mouth_h],
                [cx, mouth_y + mouth_h], [cx - 15, mouth_y + mouth_h]
            ], dtype=np.float32),
            'pose_points_2d': np.array([
                [cx, cy], [cx, cy + 90], [cx - 38, eye_y], [cx + 38, eye_y],
                [cx - 30, mouth_y], [cx + 30, mouth_y]
            ], dtype=np.float64)
        }

        biometrics = FrameBiometrics(
            ear_left=ear_val,
            ear_right=ear_val,
            ear_avg=ear_val,
            mar=mar_val,
            pitch=pitch_val,
            yaw=yaw_val,
            roll=roll_val,
            is_eyes_closed=(ear_val < 0.22),
            is_yawning=(mar_val > 0.65),
            is_head_distracted=(abs(yaw_val) > 20.0 or pitch_val > 20.0),
            is_looking_down_phone=(pitch_val < -14.0)
        )

        cv2.putText(canvas, f"[SIMULATION MODE: {self.mode.upper()}] Press 1:Alert 2:Microsleep 3:Phone 4:Yawn",
                    (15, self.height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        return canvas, biometrics, subsets


def run_pipeline(source: str = "0", model_dir: str = "models", enable_audio: bool = True, log_telematics: bool = True):
    print("=" * 70)
    print("  PROJECT 119: DRIVER FATIGUE & DISTRACTION SAFETY MONITOR")
    print("  Edge Computer Vision, Biomechanical Telematics & Audio Alarms")
    print("=" * 70)

    # 1. Initialize Subsystems
    print("[1/6] Verifying ML model artifacts...")
    os.makedirs(model_dir, exist_ok=True)
    svm_path = os.path.join(model_dir, "driver_safety_svm.pkl")
    fallback_path = os.path.join(model_dir, "fallback_classifier.pkl")

    if not os.path.exists(svm_path) and not os.path.exists(fallback_path):
        print("      No pre-trained model found. Initializing and training ML models now...")
        train_res = train_and_export_models(save_dir=model_dir)
        print(f"      Model Training Result: {train_res.get('status')}")

    print("[2/6] Loading Safety Classifier...")
    classifier = SafetyClassifier(model_dir=model_dir)

    print("[3/6] Initializing Audio Safety Alert Manager...")
    alert_mgr = AlertManager(sounds_dir="audio/sounds", cooldown_seconds=config.ALERT_COOLDOWN_SECONDS, enabled=enable_audio)

    print("[4/6] Initializing Commercial Telematics Logger...")
    telematics = TelematicsLogger(log_dir=config.LOG_DIR) if log_telematics else None

    print("[5/6] Setting up Video Stream & Biomechanics Pipeline...")
    is_mock = False
    cap = None

    if source.lower() in ["mock", "demo", "sim"]:
        is_mock = True
        simulator = MockDriverSimulator(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)
        print("      Operating in INTERACTIVE SIMULATION MODE (Driver Avatar).")
    else:
        cam_idx = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            print(f"      [WARNING] Could not open camera source '{source}'. Falling back to interactive simulator.")
            is_mock = True
            simulator = MockDriverSimulator(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            print(f"      Connected successfully to video input: {source}")

    landmark_extractor = LandmarkExtractor() if MEDIAPIPE_AVAILABLE else None
    geometry_engine = GeometryEngine(ear_threshold=config.EAR_THRESHOLD, mar_threshold=config.MAR_THRESHOLD)
    temporal_buffer = TemporalBuffer(window_size_seconds=config.WINDOW_SIZE_SECONDS, target_fps=config.TARGET_FPS)
    hud = HUDOverlay()

    print("[6/6] Pipeline Initialized Successfully! Starting edge loop...")
    print("      Controls: [Q] Quit | [M] Mute Audio | [D] Mesh Wireframe | [S] Screenshot")
    if is_mock:
        print("      Simulator Keys: [1] Alert | [2] Microsleep | [3] Phone In Hand | [4] Yawn")

    prev_time = time.time()
    fps = 0.0
    active_event_type = None
    event_start_time = 0.0

    window_title = "Project 119: Driver Fatigue & Distraction Safety Monitor"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Capture frame
            if is_mock:
                frame, biometrics, subsets = simulator.generate_frame_and_biometrics()
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("End of video stream or failed capture.")
                    break

                h, w = frame.shape[:2]
                norm_landmarks = landmark_extractor.process_frame(frame) if landmark_extractor else None

                if norm_landmarks is not None:
                    subsets = landmark_extractor.extract_subsets(norm_landmarks, w, h)
                    biometrics = geometry_engine.compute_all(subsets, w, h)
                else:
                    subsets = None
                    biometrics = None

            # Temporal Aggregation & ML Classification
            if biometrics is not None:
                temporal_buffer.add_frame(biometrics, timestamp=curr_time)
                features = temporal_buffer.extract_features()
                classification = classifier.predict(
                    features,
                    current_biometrics=biometrics,
                    consec_closed_frames=temporal_buffer.consecutive_closed_frames
                )
            else:
                classification = ClassificationResult(
                    class_id=0,
                    class_name="NO FACE DETECTED",
                    confidence=0.0,
                    probabilities=[0.33, 0.33, 0.33],
                    is_critical_alarm=False,
                    trigger_reason="Searching for driver face in cabin..."
                )

            # Safety Interventions & Sound Alerts
            if classification.class_id == 1:  # Drowsy / Microsleep
                alert_type = AlertType.CRITICAL_MICROSLEEP if classification.is_critical_alarm else AlertType.DROWSY_WARNING
                alert_mgr.trigger(alert_type)

                if active_event_type != "DROWSY":
                    active_event_type = "DROWSY"
                    event_start_time = curr_time
            elif classification.class_id == 2:  # Distracted / Phone
                alert_mgr.trigger(AlertType.PHONE_DISTRACTION)

                if active_event_type != "PHONE_DISTRACTION":
                    active_event_type = "PHONE_DISTRACTION"
                    event_start_time = curr_time
            else:
                if active_event_type is not None:
                    # Event finished: Log completed infraction to fleet telematics
                    if telematics and biometrics is not None:
                        duration = curr_time - event_start_time
                        severity = "CRITICAL" if duration > 1.5 else "WARNING"
                        telematics.log_event(
                            event_type=active_event_type,
                            severity=severity,
                            duration_sec=duration,
                            ear_val=biometrics.ear_avg,
                            mar_val=biometrics.mar,
                            pitch_deg=biometrics.pitch,
                            yaw_deg=biometrics.yaw,
                            trigger_reason=f"Event duration: {duration:.2f}s"
                        )
                    active_event_type = None

            # Render HUD onto frame
            hud_frame = hud.draw_hud(
                frame=frame,
                biometrics=biometrics,
                classification=classification,
                subsets=subsets,
                fps=fps,
                is_muted=alert_mgr.muted
            )

            cv2.imshow(window_title, hud_frame)

            # Keyboard Interactions
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # ESC or 'q'
                print("\nShutdown signal received. Exiting...")
                break
            elif key == ord('m'):
                is_muted = alert_mgr.toggle_mute()
                print(f"[AUDIO] Mute toggled: {is_muted}")
            elif key == ord('d'):
                hud.show_landmarks = not hud.show_landmarks
                print(f"[HUD] Facial landmark wireframe toggled: {hud.show_landmarks}")
            elif key == ord('s'):
                snap_path = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_path, hud_frame)
                print(f"[SCREENSHOT] Saved HUD telemetry snapshot to {snap_path}")
            elif is_mock:
                if key == ord('1'):
                    simulator.set_mode("alert")
                    print("[SIM] Mode set to: ALERT")
                elif key == ord('2'):
                    simulator.set_mode("microsleep")
                    print("[SIM] Mode set to: MICROSLEEP (Eyes Closed)")
                elif key == ord('3'):
                    simulator.set_mode("phone")
                    print("[SIM] Mode set to: PHONE DISTRACTION (Looking into lap)")
                elif key == ord('4'):
                    simulator.set_mode("yawn")
                    print("[SIM] Mode set to: YAWNING")

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Telematics session ended cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Driver Fatigue & Distraction Safety Monitor")
    parser.add_argument("--source", type=str, default="mock", help="Video source: camera index (e.g. 0), path to video file, or 'mock' for simulator")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory for ML models")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio alarm tones")
    parser.add_argument("--no-telematics", action="store_true", help="Disable telematics logging")
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        model_dir=args.model_dir,
        enable_audio=not args.no_audio,
        log_telematics=not args.no_telematics
    )


if __name__ == "__main__":
    main()
