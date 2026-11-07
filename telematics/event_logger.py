"""
Commercial Telematics & Fleet Event Logger.
Logs timestamped safety infractions, microsleep, and phone-distraction events
to structured JSONL & CSV files for fleet manager telematics compliance.
"""

import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, Optional


class TelematicsLogger:
    def __init__(self, log_dir: str = "logs", trip_id: Optional[str] = None):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        if trip_id is None:
            trip_id = datetime.now().strftime("trip_%Y%m%d_%H%M%S")
        self.trip_id = trip_id

        self.jsonl_path = os.path.join(self.log_dir, f"{trip_id}_events.jsonl")
        self.csv_path = os.path.join(self.log_dir, f"{trip_id}_summary.csv")

        self.event_counter = 0
        self._init_csv()

    def _init_csv(self) -> None:
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "event_id", "timestamp_iso", "trip_id", "event_type",
                    "severity", "duration_sec", "ear_val", "mar_val",
                    "pitch_deg", "yaw_deg", "mock_speed_kmh", "trigger_reason"
                ])

    def log_event(
        self,
        event_type: str,
        severity: str,
        duration_sec: float,
        ear_val: float,
        mar_val: float,
        pitch_deg: float,
        yaw_deg: float,
        trigger_reason: str,
        mock_speed_kmh: float = 88.5
    ) -> Dict[str, Any]:
        """Records an event entry to both JSONL and CSV."""
        self.event_counter += 1
        now_iso = datetime.now().isoformat()

        record = {
            "event_id": self.event_counter,
            "timestamp_iso": now_iso,
            "timestamp_epoch": time.time(),
            "trip_id": self.trip_id,
            "event_type": event_type,
            "severity": severity,
            "duration_sec": round(duration_sec, 2),
            "ear_val": round(ear_val, 3),
            "mar_val": round(mar_val, 3),
            "pitch_deg": round(pitch_deg, 1),
            "yaw_deg": round(yaw_deg, 1),
            "mock_speed_kmh": mock_speed_kmh,
            "trigger_reason": trigger_reason
        }

        # Append to JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Append to CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                record["event_id"],
                record["timestamp_iso"],
                record["trip_id"],
                record["event_type"],
                record["severity"],
                record["duration_sec"],
                record["ear_val"],
                record["mar_val"],
                record["pitch_deg"],
                record["yaw_deg"],
                record["mock_speed_kmh"],
                record["trigger_reason"]
            ])

        return record
