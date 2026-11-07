"""
Automotive Heads-Up Display (HUD) Visualizer.
Renders high-contrast, real-time safety telemetry, biometric gauge meters,
pose orientation compass, and warning borders directly onto OpenCV frames.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import cv2

from core.geometry_features import FrameBiometrics
from ml.classifier import ClassificationResult


class HUDOverlay:
    # Color palette (BGR format for OpenCV)
    COLOR_BG_DARK = (20, 20, 20)
    COLOR_TEXT_WHITE = (245, 245, 245)
    COLOR_TEXT_DIM = (160, 160, 160)
    COLOR_SAFE_GREEN = (60, 220, 100)
    COLOR_WARN_AMBER = (30, 170, 255)
    COLOR_ALERT_RED = (40, 40, 235)
    COLOR_CYAN = (230, 210, 50)
    COLOR_PURPLE = (200, 80, 210)

    def __init__(self):
        self.show_landmarks = True
        self.flash_state = False
        self.frame_count = 0

    def draw_bar(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        value: float,
        threshold: float,
        label: str,
        max_val: float = 1.0,
        invert_thresh: bool = False
    ) -> None:
        """Renders a telemetry gauge bar with warning threshold mark."""
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)

        # Fill length
        fill_pct = np.clip(value / max_val, 0.0, 1.0)
        fill_w = int(width * fill_pct)

        # Danger color decision
        if invert_thresh:  # Lower than threshold is dangerous (e.g. EAR)
            bar_color = self.COLOR_ALERT_RED if value < threshold else self.COLOR_SAFE_GREEN
        else:              # Higher than threshold is dangerous (e.g. MAR, PERCLOS)
            bar_color = self.COLOR_ALERT_RED if value > threshold else self.COLOR_SAFE_GREEN

        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), bar_color, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (120, 120, 120), 1)

        # Threshold tick line
        thresh_x = int(x + width * (threshold / max_val))
        cv2.line(frame, (thresh_x, y - 2), (thresh_x, y + height + 2), (0, 255, 255), 2)

        # Text label and numerical value
        text = f"{label}: {value:.2f}"
        cv2.putText(frame, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    def draw_pose_indicator(self, frame: np.ndarray, x: int, y: int, pitch: float, yaw: float, roll: float) -> None:
        """Renders pitch, yaw, roll compass values."""
        cv2.putText(frame, f"PITCH: {pitch:+5.1f} deg", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, f"YAW:   {yaw:+5.1f} deg", (x, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, f"ROLL:  {roll:+5.1f} deg", (x, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    def draw_hud(
        self,
        frame: np.ndarray,
        biometrics: Optional[FrameBiometrics],
        classification: ClassificationResult,
        subsets: Optional[Dict[str, np.ndarray]],
        fps: float,
        is_muted: bool
    ) -> np.ndarray:
        """Draws full HUD telemetry onto the frame."""
        h, w = frame.shape[:2]
        self.frame_count += 1
        if self.frame_count % 8 == 0:
            self.flash_state = not self.flash_state

        # 1. TOP TELEMETRY STATUS BAR (Semi-transparent overlay)
        top_bar = frame[0:46, 0:w].copy()
        cv2.rectangle(top_bar, (0, 0), (w, 46), self.COLOR_BG_DARK, -1)
        frame[0:46, 0:w] = cv2.addWeighted(top_bar, 0.75, frame[0:46, 0:w], 0.25, 0)

        # System title & FPS
        cv2.putText(frame, "TELEMATICS DMS v2.4", (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:4.1f}", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        # Audio Mute status
        audio_str = "[AUDIO: MUTED]" if is_muted else "[AUDIO: ACTIVE]"
        audio_color = (0, 165, 255) if is_muted else self.COLOR_SAFE_GREEN
        cv2.putText(frame, audio_str, (140, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, audio_color, 1, cv2.LINE_AA)

        # Driver State Banner in top bar
        class_colors = {
            0: self.COLOR_SAFE_GREEN,
            1: self.COLOR_ALERT_RED,
            2: self.COLOR_WARN_AMBER
        }
        status_color = class_colors.get(classification.class_id, self.COLOR_SAFE_GREEN)
        status_text = f"STATE: {classification.class_name} ({classification.confidence * 100:.0f}%)"
        cv2.putText(frame, status_text, (w - 340, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

        # 2. EMERGENCY FLASHING BORDER (when in danger state)
        if classification.class_id in [1, 2] and self.flash_state:
            border_col = self.COLOR_ALERT_RED if classification.class_id == 1 else self.COLOR_WARN_AMBER
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_col, 8)

        # 3. CRITICAL ALARM BANNER ACROSS LOWER SCREEN
        if classification.class_id in [1, 2]:
            banner_h = 44
            banner_y = h - 60
            banner_overlay = frame[banner_y:banner_y + banner_h, 0:w].copy()
            bg_col = (0, 0, 160) if classification.class_id == 1 else (0, 100, 180)
            cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), bg_col, -1)
            frame[banner_y:banner_y + banner_h, 0:w] = cv2.addWeighted(banner_overlay, 0.85, frame[banner_y:banner_y + banner_h, 0:w], 0.15, 0)

            alarm_icon = "(!)" if self.flash_state else "(X)"
            msg = f"{alarm_icon} SAFETY ALERT: {classification.trigger_reason}"
            cv2.putText(frame, msg, (15, banner_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. BIOMETRIC TELEMETRY SIDEBAR (LEFT SIDE)
        if biometrics is not None:
            panel_w = 170
            panel_h = 160
            py = 55
            sidebar_rect = frame[py:py + panel_h, 10:10 + panel_w].copy()
            cv2.rectangle(sidebar_rect, (0, 0), (panel_w, panel_h), self.COLOR_BG_DARK, -1)
            frame[py:py + panel_h, 10:10 + panel_w] = cv2.addWeighted(sidebar_rect, 0.70, frame[py:py + panel_h, 10:10 + panel_w], 0.30, 0)

            # EAR Bar (threshold 0.22)
            self.draw_bar(frame, 20, py + 22, 140, 12, biometrics.ear_avg, 0.22, "EAR (Eyelids)", max_val=0.45, invert_thresh=True)

            # MAR Bar (threshold 0.65)
            self.draw_bar(frame, 20, py + 56, 140, 12, biometrics.mar, 0.65, "MAR (Yawn)", max_val=1.0, invert_thresh=False)

            # Head Pose
            self.draw_pose_indicator(frame, 20, py + 92, biometrics.pitch, biometrics.yaw, biometrics.roll)

        # 5. FACIAL LANDMARK WIREFRAME (If enabled and face detected)
        if self.show_landmarks and subsets is not None:
            # Draw eyes
            for eye_key in ['left_eye', 'right_eye']:
                pts = subsets[eye_key].astype(np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=self.COLOR_CYAN, thickness=1, lineType=cv2.LINE_AA)

            # Draw lips
            mouth_pts = subsets['mouth'].astype(np.int32)
            cv2.polylines(frame, [mouth_pts], isClosed=True, color=self.COLOR_PURPLE, thickness=1, lineType=cv2.LINE_AA)

            # Draw nose tip point
            nose = subsets['pose_points_2d'][0].astype(np.int32)
            cv2.circle(frame, (nose[0], nose[1]), 3, (0, 255, 255), -1)

        # 6. BOTTOM COMMAND HELP STRIP
        cv2.putText(frame, "[Q] Quit  [M] Mute  [D] Toggle Mesh  [S] Save Snapshot  [T] Sim Alert", (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        return frame
