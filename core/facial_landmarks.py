"""
Facial Landmark Extractor using MediaPipe Face Mesh.
Extracts 468 3D facial landmarks and standard landmark subsets for:
- Left Eye & Right Eye (EAR)
- Inner Lips / Mouth (MAR)
- Canonical Keypoints for Head Pose Estimation (PnP)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class LandmarkExtractor:
    # --- MediaPipe Landmark Indices ---
    # Left Eye 6-point indices (compatible with Soukupova & Cech EAR formulation)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    # Right Eye 6-point indices
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

    # Mouth Inner Lip keypoints for MAR (vertical upper/lower pairs and horizontal corners)
    # 78: left corner, 308: right corner, 13: upper lip center, 14: lower lip center,
    # 81/178: left-mid vertical, 311/402: right-mid vertical
    MOUTH_INDICES = [78, 81, 13, 311, 308, 402, 14, 178]

    # 3D Head Pose Canonical Landmarks in FaceMesh
    # 1: Nose tip, 152: Chin, 33: Right eye corner, 263: Left eye corner, 61: Right mouth corner, 291: Left mouth corner
    HEAD_POSE_INDICES = {
        'nose_tip': 1,
        'chin': 152,
        'right_eye': 33,
        'left_eye': 263,
        'right_mouth': 61,
        'left_mouth': 291
    }

    def __init__(self, max_num_faces: int = 1, refine_landmarks: bool = True, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.refine_landmarks = refine_landmarks
        self.face_mesh = None

        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Processes BGR image and returns normalized (x, y, z) landmarks array of shape (N, 3).
        Returns None if no face is detected or if mediapipe is unavailable.
        """
        if self.face_mesh is None:
            return None

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        # Extract primary face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        points = np.array([
            [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
        ], dtype=np.float32)

        return points

    def get_pixel_landmarks(self, norm_landmarks: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
        """Converts normalized landmarks [0, 1] to pixel coordinates [0, W/H]."""
        pixel_points = norm_landmarks.copy()
        pixel_points[:, 0] *= frame_width
        pixel_points[:, 1] *= frame_height
        return pixel_points

    def extract_subsets(self, norm_landmarks: np.ndarray, frame_width: int, frame_height: int) -> Dict[str, np.ndarray]:
        """Returns pixel coordinates for key facial sub-structures."""
        pixel_pts = self.get_pixel_landmarks(norm_landmarks, frame_width, frame_height)

        return {
            'left_eye': pixel_pts[self.LEFT_EYE_INDICES, :2],
            'right_eye': pixel_pts[self.RIGHT_EYE_INDICES, :2],
            'mouth': pixel_pts[self.MOUTH_INDICES, :2],
            'pose_points_2d': np.array([
                pixel_pts[self.HEAD_POSE_INDICES['nose_tip'], :2],
                pixel_pts[self.HEAD_POSE_INDICES['chin'], :2],
                pixel_pts[self.HEAD_POSE_INDICES['left_eye'], :2],
                pixel_pts[self.HEAD_POSE_INDICES['right_eye'], :2],
                pixel_pts[self.HEAD_POSE_INDICES['left_mouth'], :2],
                pixel_pts[self.HEAD_POSE_INDICES['right_mouth'], :2]
            ], dtype=np.float64)
        }
