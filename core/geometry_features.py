"""
Biomechanical & Geometric Feature Calculations:
- Eye Aspect Ratio (EAR) for blink & microsleep detection
- Mouth Aspect Ratio (MAR) for yawning detection
- Head Pose Estimation (Pitch, Yaw, Roll) via Perspective-n-Point (cv2.solvePnP)
- Mobile-phone distraction heuristic detection
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import cv2


@dataclass
class FrameBiometrics:
    ear_left: float
    ear_right: float
    ear_avg: float
    mar: float
    pitch: float      # Degrees: -down / +up
    yaw: float        # Degrees: -left / +right
    roll: float       # Degrees: -tilt left / +tilt right
    is_eyes_closed: bool
    is_yawning: bool
    is_head_distracted: bool
    is_looking_down_phone: bool


class GeometryEngine:
    def __init__(self, ear_threshold: float = 0.22, mar_threshold: float = 0.65):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold

        # 3D Anthropometric Facial Model Keypoints (in mm relative to nose tip)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye outer corner
            (225.0, 170.0, -135.0),    # Right eye outer corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype=np.float64)

    @staticmethod
    def euclidean_dist(pt1: np.ndarray, pt2: np.ndarray) -> float:
        """Euclidean distance between two 2D points."""
        return float(np.linalg.norm(pt1 - pt2))

    def compute_ear(self, eye_pts: np.ndarray) -> float:
        """
        Eye Aspect Ratio (EAR) based on Soukupova & Cech (2016):
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        eye_pts shape: (6, 2)
        """
        if eye_pts.shape[0] < 6:
            return 0.0

        p1, p2, p3, p4, p5, p6 = eye_pts[:6]
        vertical_1 = self.euclidean_dist(p2, p6)
        vertical_2 = self.euclidean_dist(p3, p5)
        horizontal = self.euclidean_dist(p1, p4)

        if horizontal <= 1e-6:
            return 0.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return float(ear)

    def compute_mar(self, mouth_pts: np.ndarray) -> float:
        """
        Mouth Aspect Ratio (MAR) based on inner lips:
        MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (3 * ||p1 - p5||)
        mouth_pts shape: (8, 2)
        indices: 0: left corner, 1: top-left, 2: top-center, 3: top-right,
                 4: right corner, 5: bottom-right, 6: bottom-center, 7: bottom-left
        """
        if mouth_pts.shape[0] < 8:
            return 0.0

        p1, p2, p3, p4, p5, p6, p7, p8 = mouth_pts[:8]
        vert_1 = self.euclidean_dist(p2, p8)
        vert_2 = self.euclidean_dist(p3, p7)
        vert_3 = self.euclidean_dist(p4, p6)
        horiz = self.euclidean_dist(p1, p5)

        if horiz <= 1e-6:
            return 0.0

        mar = (vert_1 + vert_2 + vert_3) / (3.0 * horiz)
        return float(mar)

    def estimate_head_pose(self, image_points_2d: np.ndarray, frame_width: int, frame_height: int) -> Tuple[float, float, float]:
        """
        Solves Perspective-n-Point (cv2.solvePnP) to calculate 3D head rotation.
        Returns (Pitch, Yaw, Roll) in degrees.
        """
        # Approximate pinhole camera intrinsics
        focal_length = frame_width
        center = (frame_width / 2.0, frame_height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assume minimal radial distortion

        success, rvec, tvec = cv2.solvePnP(
            self.model_points_3d,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Convert rotation vector to 3x3 rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler Angles
        # Using cv2.RQDecomp3x3 for robust yaw/pitch/roll decomposition
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = float(angles[0])  # Nodding (down/up)
        yaw = float(angles[1])    # Turning head left/right
        roll = float(angles[2])   # Tilting head ear-to-shoulder

        return pitch, yaw, roll

    def compute_all(self, subsets: Dict[str, np.ndarray], frame_width: int, frame_height: int) -> FrameBiometrics:
        """Computes all frame-level geometric metrics and state flags."""
        ear_left = self.compute_ear(subsets['left_eye'])
        ear_right = self.compute_ear(subsets['right_eye'])
        ear_avg = (ear_left + ear_right) / 2.0

        mar = self.compute_mar(subsets['mouth'])

        pitch, yaw, roll = self.estimate_head_pose(subsets['pose_points_2d'], frame_width, frame_height)

        # Threshold checks
        is_eyes_closed = bool(ear_avg < self.ear_threshold)
        is_yawning = bool(mar > self.mar_threshold)

        # Distraction condition: Head pitched down or turned sideways
        is_head_distracted = bool(abs(yaw) > 22.0 or pitch > 20.0 or abs(roll) > 20.0)

        # Phone usage signature: Driver persistently looking downwards into lap/console (pitch < -14 degrees)
        is_looking_down_phone = bool(pitch < -14.0)

        return FrameBiometrics(
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            is_eyes_closed=is_eyes_closed,
            is_yawning=is_yawning,
            is_head_distracted=is_head_distracted,
            is_looking_down_phone=is_looking_down_phone
        )
