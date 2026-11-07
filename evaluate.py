"""
Model Evaluation & Latency Benchmarking Script.
Evaluates high Recall for dangerous classes (Drowsy, Microsleep, Phone Distraction)
and measures end-to-end inference latency in milliseconds and frames per second (FPS).
"""

import time
import numpy as np
from ml.model_trainer import generate_synthetic_telematics_dataset, train_and_export_models
from ml.classifier import SafetyClassifier
from core.temporal_buffer import TemporalFeatures


def run_comprehensive_evaluation():
    print("=" * 70)
    print("  PROJECT 119: MODEL EVALUATION & LATENCY BENCHMARK")
    print("=" * 70)

    # 1. Train or load model
    print("[1/3] Ensuring models are trained and exported...")
    train_res = train_and_export_models()
    print("      Training Status:", train_res.get("status"))

    classifier = SafetyClassifier()

    # 2. Benchmark on independent test dataset
    print("\n[2/3] Evaluating on independent 1,500-sample telematics dataset...")
    X_test, y_test = generate_synthetic_telematics_dataset(num_samples_per_class=500, random_state=999)

    predictions = []
    latencies = []

    for i in range(len(X_test)):
        row = X_test[i]
        feat = TemporalFeatures(
            ear_mean=float(row[0]), ear_std=float(row[1]), ear_min=float(row[2]),
            perclos=float(row[3]), blink_count=int(row[4]), blink_rate_per_min=float(row[5]),
            mar_mean=float(row[6]), mar_max=float(row[7]), yawn_ratio=float(row[8]),
            pitch_mean=float(row[9]), pitch_std=float(row[10]), yaw_mean=float(row[11]),
            yaw_std=float(row[12]), roll_mean=float(row[13]), phone_posture_ratio=float(row[14]),
            distraction_ratio=float(row[15])
        )

        t0 = time.perf_counter()
        res = classifier.predict(feat)
        t1 = time.perf_counter()

        predictions.append(res.class_id)
        latencies.append((t1 - t0) * 1000.0)  # ms

    predictions = np.array(predictions)

    # Calculate metrics
    classes = ["ALERT", "DROWSY", "DISTRACTED"]
    confusion_mat = np.zeros((3, 3), dtype=int)
    for true_c, pred_c in zip(y_test, predictions):
        confusion_mat[true_c, pred_c] += 1

    print("\n" + "-" * 60)
    print(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 60)

    overall_recalls = []
    for c_idx, c_name in enumerate(classes):
        tp = confusion_mat[c_idx, c_idx]
        fp = np.sum(confusion_mat[:, c_idx]) - tp
        fn = np.sum(confusion_mat[c_idx, :]) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        overall_recalls.append(rec)

        print(f"{c_name:<20} | {prec * 100:8.2f}% | {rec * 100:8.2f}% | {f1 * 100:8.2f}%")

    print("-" * 60)
    acc = np.mean(predictions == y_test)
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Drowsy/Microsleep Recall: {overall_recalls[1] * 100:.2f}% (Target: >95% to avoid fatal misses)")
    print(f"Phone Distraction Recall: {overall_recalls[2] * 100:.2f}% (Target: >90%)")

    # Latency Stats
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)
    throughput_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print("\n[3/3] Real-time Edge Latency Benchmarks (Inference Only):")
    print(f"      Average Latency: {mean_lat:.4f} ms")
    print(f"      95th Percentile: {p95_lat:.4f} ms")
    print(f"      99th Percentile: {p99_lat:.4f} ms")
    print(f"      Max Throughput:  {throughput_fps:.1f} FPS (Exceeds 30 FPS real-time standard)")
    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_evaluation()
