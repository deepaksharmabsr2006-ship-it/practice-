"""
Machine Learning Model Trainer & Benchmark Evaluator.
Trains lightweight Support Vector Machine (SVM) and Random Forest classifiers
on aggregated temporal features (PERCLOS, EAR, MAR, Head Pose dynamics)
to distinguish between:
  - Class 0: ALERT
  - Class 1: DROWSY / MICROSLEEP
  - Class 2: DISTRACTED / PHONE USAGE

Outputs serialized model and scaler to models/ directory.
Includes a fallback standalone model generator so the system works with zero external dependencies.
"""

import os
import pickle
import numpy as np
from typing import Dict, Tuple, Any

from core.temporal_buffer import TemporalFeatures


def generate_synthetic_telematics_dataset(num_samples_per_class: int = 1200, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates realistic physiological & telematics time-series feature distributions
    based on SAE International & NHTSA driver fatigue / distraction biomechanical studies:
      - Class 0 (Alert): High EAR (0.28-0.36), low PERCLOS (<0.10), regular blinks (12-24/min),
                         stable MAR (0.15-0.35), centered pitch/yaw near 0.
      - Class 1 (Drowsy/Microsleep): Low EAR (0.10-0.21), high PERCLOS (>0.40), sluggish or
                         sporadic blinks, elevated MAR (0.60-0.95 during yawns), nodding pitch.
      - Class 2 (Distracted/Phone): Low pitch (-15 to -35 deg looking into lap/console),
                         high yaw deviation (turning head to passenger/side), eyes often open
                         or gaze directed downward, high phone_posture_ratio (>0.50).
    """
    np.random.seed(random_state)
    X = []
    y = []

    # 1. ALERT SAMPLES
    for _ in range(num_samples_per_class):
        ear_mean = np.random.normal(0.31, 0.03)
        ear_std = np.random.exponential(0.02)
        ear_min = max(0.18, ear_mean - np.random.uniform(0.04, 0.08))
        perclos = np.random.beta(1.5, 25.0)  # very low (0.00 - 0.08)
        blink_count = int(np.random.poisson(1.0))
        blink_rate = np.random.normal(16.0, 4.0)
        mar_mean = np.random.normal(0.24, 0.04)
        mar_max = mar_mean + np.random.uniform(0.05, 0.15)
        yawn_ratio = 0.0 if np.random.rand() > 0.05 else np.random.uniform(0.0, 0.08)
        pitch_mean = np.random.normal(-1.0, 4.0)
        pitch_std = np.random.uniform(1.0, 3.5)
        yaw_mean = np.random.normal(0.0, 5.0)
        yaw_std = np.random.uniform(1.5, 4.0)
        roll_mean = np.random.normal(0.0, 3.0)
        phone_posture_ratio = np.random.beta(1.0, 30.0)
        distraction_ratio = np.random.beta(1.0, 20.0)

        vec = [
            ear_mean, ear_std, ear_min, perclos, blink_count, blink_rate,
            mar_mean, mar_max, yawn_ratio, pitch_mean, pitch_std,
            yaw_mean, yaw_std, roll_mean, phone_posture_ratio, distraction_ratio
        ]
        X.append(vec)
        y.append(0)

    # 2. DROWSY / MICROSLEEP SAMPLES
    for _ in range(num_samples_per_class):
        is_yawn_dominant = np.random.rand() > 0.5
        is_microsleep = np.random.rand() > 0.4

        if is_microsleep:
            # Complete or prolonged eye closure (microsleep)
            ear_mean = np.random.normal(0.14, 0.03)
            ear_min = np.random.uniform(0.05, 0.12)
            perclos = np.random.uniform(0.45, 0.95)
            blink_rate = np.random.uniform(2.0, 10.0)
            blink_count = int(np.random.poisson(0.5))
        else:
            # Heavy eyelids / drooping
            ear_mean = np.random.normal(0.19, 0.02)
            ear_min = np.random.uniform(0.10, 0.16)
            perclos = np.random.uniform(0.25, 0.55)
            blink_rate = np.random.uniform(24.0, 42.0)  # Frequent flutter blinks
            blink_count = int(np.random.poisson(2.5))

        ear_std = np.random.uniform(0.04, 0.08)

        if is_yawn_dominant:
            mar_mean = np.random.normal(0.55, 0.08)
            mar_max = np.random.uniform(0.70, 1.10)
            yawn_ratio = np.random.uniform(0.35, 0.85)
        else:
            mar_mean = np.random.normal(0.26, 0.05)
            mar_max = mar_mean + np.random.uniform(0.05, 0.20)
            yawn_ratio = np.random.uniform(0.0, 0.15)

        # Head nodding during drowsiness
        pitch_mean = np.random.normal(-8.0, 6.0)
        pitch_std = np.random.uniform(4.0, 9.0)
        yaw_mean = np.random.normal(0.0, 6.0)
        yaw_std = np.random.uniform(2.0, 5.0)
        roll_mean = np.random.normal(2.0, 6.0)
        phone_posture_ratio = np.random.beta(1.5, 10.0)
        distraction_ratio = np.random.uniform(0.1, 0.35)

        vec = [
            ear_mean, ear_std, ear_min, perclos, blink_count, blink_rate,
            mar_mean, mar_max, yawn_ratio, pitch_mean, pitch_std,
            yaw_mean, yaw_std, roll_mean, phone_posture_ratio, distraction_ratio
        ]
        X.append(vec)
        y.append(1)

    # 3. DISTRACTED / PHONE USAGE SAMPLES
    for _ in range(num_samples_per_class):
        is_phone_lap = np.random.rand() > 0.4
        ear_mean = np.random.normal(0.29, 0.03)  # Eyes are generally open
        ear_std = np.random.uniform(0.02, 0.04)
        ear_min = np.random.uniform(0.20, 0.26)
        perclos = np.random.beta(1.5, 18.0)
        blink_count = int(np.random.poisson(1.0))
        blink_rate = np.random.normal(14.0, 4.0)
        mar_mean = np.random.normal(0.23, 0.04)
        mar_max = mar_mean + np.random.uniform(0.05, 0.12)
        yawn_ratio = 0.0

        if is_phone_lap:
            # Looking downward into lap or lower center console
            pitch_mean = np.random.normal(-24.0, 5.0)
            pitch_std = np.random.uniform(3.0, 6.0)
            yaw_mean = np.random.normal(-6.0, 8.0)
            phone_posture_ratio = np.random.uniform(0.55, 0.98)
            distraction_ratio = np.random.uniform(0.60, 1.0)
        else:
            # Glancing sideways (phone at window, passenger, infotainment)
            pitch_mean = np.random.normal(-5.0, 6.0)
            pitch_std = np.random.uniform(2.0, 5.0)
            yaw_mean = np.random.choice([-1.0, 1.0]) * np.random.uniform(25.0, 48.0)
            phone_posture_ratio = np.random.uniform(0.15, 0.45)
            distraction_ratio = np.random.uniform(0.70, 1.0)

        yaw_std = np.random.uniform(4.0, 10.0)
        roll_mean = np.random.normal(0.0, 5.0)

        vec = [
            ear_mean, ear_std, ear_min, perclos, blink_count, blink_rate,
            mar_mean, mar_max, yawn_ratio, pitch_mean, pitch_std,
            yaw_mean, yaw_std, roll_mean, phone_posture_ratio, distraction_ratio
        ]
        X.append(vec)
        y.append(2)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int64)

    # Shuffle
    indices = np.random.permutation(len(y_arr))
    return X_arr[indices], y_arr[indices]


def train_and_export_models(save_dir: str = "models") -> Dict[str, Any]:
    """Trains SVM and Random Forest, performs evaluation, and saves artifacts."""
    os.makedirs(save_dir, exist_ok=True)
    X, y = generate_synthetic_telematics_dataset(num_samples_per_class=1200)

    # Train-test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Support Vector Machine with RBF kernel and probability calibration
        svm_model = SVC(kernel='rbf', C=1.5, gamma='scale', probability=True, random_state=42)
        svm_model.fit(X_train_scaled, y_train)

        # Random Forest for comparison
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X_train_scaled, y_train)

        y_pred_svm = svm_model.predict(X_test_scaled)
        y_pred_rf = rf_model.predict(X_test_scaled)

        svm_acc = float(accuracy_score(y_test, y_pred_svm))
        rf_acc = float(accuracy_score(y_test, y_pred_rf))

        report_svm = classification_report(y_test, y_pred_svm, target_names=["ALERT", "DROWSY", "DISTRACTED"])
        cm_svm = confusion_matrix(y_test, y_pred_svm)

        # Save artifacts
        with open(os.path.join(save_dir, "driver_safety_svm.pkl"), "wb") as f:
            pickle.dump(svm_model, f)
        with open(os.path.join(save_dir, "driver_safety_rf.pkl"), "wb") as f:
            pickle.dump(rf_model, f)
        with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        results = {
            "svm_accuracy": svm_acc,
            "rf_accuracy": rf_acc,
            "classification_report": report_svm,
            "confusion_matrix": cm_svm.tolist(),
            "status": "success",
            "model_type": "scikit-learn"
        }
        return results

    except ImportError:
        # Standalone NumPy-based Normalizer and Centroid Classifier fallback
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-6
        X_train_norm = (X_train - mean) / std

        # Compute class centroids
        centroids = {c: np.mean(X_train_norm[y_train == c], axis=0) for c in [0, 1, 2]}

        # Save fallback weights
        fallback_data = {
            "mean": mean,
            "std": std,
            "centroids": centroids,
            "feature_names": TemporalFeatures.feature_names()
        }
        with open(os.path.join(save_dir, "fallback_classifier.pkl"), "wb") as f:
            pickle.dump(fallback_data, f)

        return {
            "status": "fallback_created",
            "message": "Scikit-Learn not found in environment; created standalone vectorized centroid classifier.",
            "classes": ["ALERT", "DROWSY", "DISTRACTED"]
        }


if __name__ == "__main__":
    res = train_and_export_models()
    print("Model Training Output:", res)
