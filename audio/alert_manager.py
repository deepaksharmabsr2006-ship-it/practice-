"""
Cabin Audio Alert Manager.
Generates and plays multi-tier sound alerts (Warning, Critical Microsleep, Phone Distraction)
in a dedicated non-blocking background thread to preserve zero-latency video processing.
Synthesizes pure-tone WAV files automatically using Python's standard `wave` & `math` modules.
"""

import os
import sys
import wave
import math
import struct
import threading
import time
from enum import Enum


class AlertType(Enum):
    NONE = 0
    DROWSY_WARNING = 1
    CRITICAL_MICROSLEEP = 2
    PHONE_DISTRACTION = 3


class AlertManager:
    def __init__(self, sounds_dir: str = "audio/sounds", cooldown_seconds: float = 2.0, enabled: bool = True):
        self.sounds_dir = sounds_dir
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        self.muted = False

        self.last_alert_time = 0.0
        self.last_alert_type = AlertType.NONE
        self._lock = threading.Lock()
        self._is_playing = False

        os.makedirs(self.sounds_dir, exist_ok=True)
        self._generate_default_sound_effects()

    def _generate_wav(self, file_path: str, frequencies: list, duration_sec: float = 0.5, sample_rate: int = 44100):
        """Generates a clean PCM 16-bit WAV audio tone file without external dependencies."""
        if os.path.exists(file_path):
            return

        total_samples = int(sample_rate * duration_sec)
        samples_per_tone = total_samples // len(frequencies)

        with wave.open(file_path, 'w') as wav_file:
            wav_file.setnchannels(1)       # Mono
            wav_file.setsampwidth(2)       # 16-bit
            wav_file.setframerate(sample_rate)

            data = bytearray()
            for tone_freq in frequencies:
                for i in range(samples_per_tone):
                    # Envelope window to avoid clicks
                    envelope = min(1.0, i / 500.0, (samples_per_tone - i) / 500.0)
                    value = int(32767.0 * 0.75 * envelope * math.sin(2.0 * math.pi * tone_freq * (i / sample_rate)))
                    data.extend(struct.pack('<h', value))

            wav_file.writeframes(data)

    def _generate_default_sound_effects(self) -> None:
        """Creates distinct tone signatures for each alert level."""
        # 1. Drowsy Warning: Double medium tone (880Hz, 880Hz)
        self._generate_wav(
            os.path.join(self.sounds_dir, "drowsy_warning.wav"),
            frequencies=[880, 880],
            duration_sec=0.4
        )
        # 2. Critical Microsleep: Urgent siren / rapid high frequencies (1800Hz, 2400Hz, 1800Hz, 2400Hz)
        self._generate_wav(
            os.path.join(self.sounds_dir, "critical_microsleep.wav"),
            frequencies=[1800, 2400, 1800, 2400],
            duration_sec=0.6
        )
        # 3. Phone Distraction: Distinctive downward chime (1200Hz, 700Hz)
        self._generate_wav(
            os.path.join(self.sounds_dir, "phone_distraction.wav"),
            frequencies=[1200, 700],
            duration_sec=0.5
        )

    def trigger(self, alert_type: AlertType, force: bool = False) -> bool:
        """
        Triggers an audio alert in a background thread if cooldown has elapsed.
        Returns True if alert was fired, False if throttled.
        """
        if not self.enabled or self.muted or alert_type == AlertType.NONE:
            return False

        current_time = time.time()
        # Elevate priority: Critical microsleep bypasses normal cooldown
        if not force and alert_type != AlertType.CRITICAL_MICROSLEEP:
            if current_time - self.last_alert_time < self.cooldown_seconds:
                return False

        with self._lock:
            if self._is_playing:
                return False
            self._is_playing = True

        self.last_alert_time = current_time
        self.last_alert_type = alert_type

        # Dispatch non-blocking playback thread
        t = threading.Thread(target=self._play_sound_worker, args=(alert_type,), daemon=True)
        t.start()
        return True

    def _play_sound_worker(self, alert_type: AlertType) -> None:
        try:
            sound_map = {
                AlertType.DROWSY_WARNING: "drowsy_warning.wav",
                AlertType.CRITICAL_MICROSLEEP: "critical_microsleep.wav",
                AlertType.PHONE_DISTRACTION: "phone_distraction.wav",
            }

            filename = sound_map.get(alert_type)
            if not filename:
                return

            wav_path = os.path.join(self.sounds_dir, filename)

            # Native Windows winsound playback
            if sys.platform == "win32":
                try:
                    import winsound
                    winsound.PlaySound(wav_path, winsound.SND_FILENAME)
                    return
                except Exception:
                    pass

            # Fallback to pygame if installed
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                sound = pygame.mixer.Sound(wav_path)
                sound.play()
                time.sleep(sound.get_length())
                return
            except Exception:
                pass

            # Console beep fallback
            sys.stdout.write('\a')
            sys.stdout.flush()

        finally:
            with self._lock:
                self._is_playing = False

    def toggle_mute(self) -> bool:
        """Toggles audio mute state."""
        self.muted = not self.muted
        return self.muted
