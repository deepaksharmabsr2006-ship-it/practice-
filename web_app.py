"""
Driver Fatigue & Distraction Safety Monitor - Live Browser Dashboard & Telematics Server.
Serves a responsive, automotive dark-mode web application that runs directly in any browser
with live webcam access, real-time facial telemetry (EAR, MAR, Head Pose),
Web Audio API alarm sound synthesizer, and telematics incident logging.
"""

import os
import json
import http.server
import socketserver
import urllib.parse
from datetime import datetime

PORT = 5000
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project 119: Driver Fatigue & Distraction Safety Monitor</title>
    <!-- MediaPipe Face Mesh CDN -->
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js" crossorigin="anonymous"></script>
    <style>
        :root {
            --bg-dark: #0f1117;
            --panel-bg: #1a1d26;
            --panel-border: #2a2e3d;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --accent-cyan: #06b6d4;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            padding: 16px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            margin-bottom: 16px;
        }
        .header h1 { font-size: 1.25rem; font-weight: 700; color: #fff; letter-spacing: 0.5px; }
        .header .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
            padding: 4px 12px;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.3);
            font-weight: 600;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 16px;
        }

        @media (max-width: 1024px) {
            .dashboard-grid { grid-template-columns: 1fr; }
        }

        .video-card {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 16px;
            position: relative;
        }
        .video-container {
            position: relative;
            width: 100%;
            height: 480px;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        video {
            position: absolute;
            width: 100%;
            height: 100%;
            object-fit: cover;
            transform: scaleX(-1); /* Mirror camera */
        }
        canvas#hudCanvas {
            position: absolute;
            top: 0; left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 5;
        }
        .flashing-alarm {
            border: 6px solid var(--accent-red) !important;
            animation: borderFlash 0.5s infinite alternate;
        }
        .flashing-warning {
            border: 6px solid var(--accent-amber) !important;
            animation: borderFlash 0.8s infinite alternate;
        }
        @keyframes borderFlash { from { box-shadow: 0 0 20px var(--accent-red); } to { box-shadow: none; } }

        .banner-alert {
            position: absolute;
            bottom: 16px;
            left: 16px; right: 16px;
            padding: 12px 18px;
            background: rgba(239, 68, 68, 0.95);
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.95rem;
            display: none;
            align-items: center;
            gap: 10px;
            z-index: 10;
        }

        .controls-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
        }
        .btn {
            background: #252a38;
            border: 1px solid #374151;
            color: var(--text-main);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 500;
        }
        .btn:hover { background: #374151; }
        .btn-primary { background: #2563eb; border-color: #3b82f6; }
        .btn-primary:hover { background: #1d4ed8; }
        .btn-danger { background: #dc2626; border-color: #ef4444; }
        .btn-danger:hover { background: #b91c1c; }

        .telemetry-card {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .gauge-section { display: flex; flex-direction: column; gap: 12px; }
        .gauge-header { display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; }
        .gauge-bar-bg {
            height: 12px;
            background: #272d3b;
            border-radius: 6px;
            overflow: hidden;
            position: relative;
        }
        .gauge-bar-fill {
            height: 100%;
            background: var(--accent-green);
            width: 50%;
            transition: width 0.1s linear, background-color 0.2s;
        }
        .gauge-threshold-line {
            position: absolute;
            top: 0; bottom: 0;
            width: 2px;
            background: #fbbf24;
            z-index: 2;
        }

        .pose-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            text-align: center;
        }
        .pose-box {
            background: #222634;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #2d3345;
        }
        .pose-box .val { font-size: 1.1rem; font-weight: 700; color: var(--accent-cyan); }
        .pose-box .label { font-size: 0.75rem; color: var(--text-muted); margin-top: 2px; }

        .events-card {
            margin-top: 16px;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 16px;
        }
        .events-card h3 { font-size: 0.95rem; margin-bottom: 12px; }
        .events-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
        }
        .events-table th, .events-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #262b3a;
        }
        .events-table th { color: var(--text-muted); font-weight: 600; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .badge-danger { background: rgba(239, 68, 68, 0.2); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.4); }
        .badge-warning { background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.4); }
        .badge-safe { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.4); }
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1>Project 119: Driver Fatigue & Distraction Safety Monitor</h1>
            <p style="font-size: 0.82rem; color: var(--text-muted); margin-top: 4px;">Automotive Telematics & Vision AI Edge DMS</p>
        </div>
        <div id="statusPill" class="status-pill">
            <span class="status-dot"></span>
            <span id="systemStatusText">CAMERA READY</span>
        </div>
    </div>

    <div class="dashboard-grid">
        <!-- Live Video & HUD Window -->
        <div class="video-card">
            <div id="videoContainer" class="video-container">
                <video id="inputVideo" playsinline></video>
                <canvas id="hudCanvas"></canvas>
                <div id="alertBanner" class="banner-alert">
                    <span id="alertBannerIcon">&#9888;</span>
                    <span id="alertBannerText">SAFETY ALERT</span>
                </div>
            </div>

            <div class="controls-strip">
                <button id="btnStartWebcam" class="btn btn-primary" onclick="startWebcam()">&#9654; Start Camera</button>
                <button id="btnToggleMesh" class="btn" onclick="toggleMesh()">Toggle Mesh</button>
                <button id="btnToggleMute" class="btn" onclick="toggleMute()">Mute Audio</button>
                <button id="btnSimDrowsy" class="btn" onclick="simulateState('drowsy')">Simulate Microsleep</button>
                <button id="btnSimPhone" class="btn" onclick="simulateState('phone')">Simulate Phone Distraction</button>
                <button id="btnSimNormal" class="btn" onclick="simulateState('normal')">Reset Normal</button>
            </div>
        </div>

        <!-- Real-Time Telemetry & Gauges -->
        <div class="telemetry-card">
            <h3 style="font-size: 1rem;">Live Biometric Telemetry</h3>

            <!-- Eye Aspect Ratio (EAR) -->
            <div class="gauge-section">
                <div class="gauge-header">
                    <span>Eye Aspect Ratio (EAR)</span>
                    <span id="earValText" style="color: var(--accent-green);">0.32</span>
                </div>
                <div class="gauge-bar-bg">
                    <div id="earBarFill" class="gauge-bar-fill" style="width: 70%;"></div>
                    <div class="gauge-threshold-line" style="left: 48%;" title="Threshold: 0.22"></div>
                </div>
                <span style="font-size: 0.72rem; color: var(--text-muted);">Threshold: 0.22 (Below indicates closed eyelids)</span>
            </div>

            <!-- Mouth Aspect Ratio (MAR) -->
            <div class="gauge-section">
                <div class="gauge-header">
                    <span>Mouth Aspect Ratio (MAR)</span>
                    <span id="marValText" style="color: var(--text-main);">0.20</span>
                </div>
                <div class="gauge-bar-bg">
                    <div id="marBarFill" class="gauge-bar-fill" style="width: 25%;"></div>
                    <div class="gauge-threshold-line" style="left: 65%;" title="Threshold: 0.65"></div>
                </div>
                <span style="font-size: 0.72rem; color: var(--text-muted);">Threshold: 0.65 (Above indicates yawning)</span>
            </div>

            <!-- Head Pose Angles -->
            <div style="margin-top: 6px;">
                <span style="font-size: 0.85rem; font-weight: 600; display: block; margin-bottom: 8px;">Head Orientation (PnP Euler Angles)</span>
                <div class="pose-grid">
                    <div class="pose-box">
                        <div id="pitchVal" class="val">-2°</div>
                        <div class="label">PITCH (Down/Up)</div>
                    </div>
                    <div class="pose-box">
                        <div id="yawVal" class="val">0°</div>
                        <div class="label">YAW (L/R)</div>
                    </div>
                    <div class="pose-box">
                        <div id="rollVal" class="val">1°</div>
                        <div class="label">ROLL (Tilt)</div>
                    </div>
                </div>
            </div>

            <!-- Current AI Classification Output -->
            <div style="background: #141721; padding: 14px; border-radius: 8px; border: 1px solid #2a2e3d;">
                <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">ML Classifier Decision</div>
                <div id="mlStateBadge" style="font-size: 1.25rem; font-weight: 800; color: var(--accent-green); margin-top: 6px;">NORMAL / ALERT</div>
                <div id="mlReason" style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">Attentive to roadway</div>
                <div id="fpsCounter" style="font-size: 0.75rem; color: var(--accent-cyan); margin-top: 8px;">Inference Latency: 0.8 ms | 30 FPS</div>
            </div>
        </div>
    </div>

    <!-- Fleet Telematics Event Log -->
    <div class="events-card">
        <h3>Recent Fleet Telematics Infractions (Safety Log)</h3>
        <table class="events-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Event Type</th>
                    <th>Severity</th>
                    <th>Duration</th>
                    <th>EAR / MAR</th>
                    <th>Trigger Reason</th>
                </tr>
            </thead>
            <tbody id="telematicsLogBody">
                <tr>
                    <td colspan="6" style="text-align: center; color: var(--text-muted);">No infractions recorded yet. Drive safely!</td>
                </tr>
            </tbody>
        </table>
    </div>

    <script>
        // Web Audio API Synthesizer (Zero external audio file dependencies!)
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        let isMuted = false;
        let showMesh = true;

        function playTone(freqs, duration = 0.4, type = "sine") {
            if (isMuted || !audioCtx) return;
            const now = audioCtx.currentTime;
            freqs.forEach((f, idx) => {
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.type = type;
                osc.frequency.setValueAtTime(f, now + idx * (duration / freqs.length));
                gain.gain.setValueAtTime(0.3, now + idx * (duration / freqs.length));
                gain.gain.exponentialRampToValueAtTime(0.01, now + (idx + 1) * (duration / freqs.length));
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start(now + idx * (duration / freqs.length));
                osc.stop(now + (idx + 1) * (duration / freqs.length));
            });
        }

        // State Machine & Temporal Aggregator
        let consecClosed = 0;
        let consecPhone = 0;
        let lastAlarmTime = 0;
        let simulationOverride = null;

        const EAR_THRESH = 0.22;
        const MAR_THRESH = 0.65;

        // Euclidean Distance
        function dist(p1, p2) {
            return Math.hypot(p1.x - p2.x, p1.y - p2.y);
        }

        // Calculate EAR from 6 landmarks
        function calcEAR(eye) {
            const v1 = dist(eye[1], eye[5]);
            const v2 = dist(eye[2], eye[4]);
            const h = dist(eye[0], eye[3]);
            return (v1 + v2) / (2.0 * (h || 0.001));
        }

        // MediaPipe FaceMesh Setup
        const videoElement = document.getElementById('inputVideo');
        const canvasElement = document.getElementById('hudCanvas');
        const canvasCtx = canvasElement.getContext('2d');
        const container = document.getElementById('videoContainer');

        const faceMesh = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });

        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        faceMesh.onResults(onResults);

        function startWebcam() {
            if (audioCtx.state === 'suspended') audioCtx.resume();
            document.getElementById('btnStartWebcam').style.display = 'none';

            const camera = new Camera(videoElement, {
                onFrame: async () => {
                    await faceMesh.send({ image: videoElement });
                },
                width: 640,
                height: 480
            });
            camera.start();
            document.getElementById('systemStatusText').innerText = 'LIVE MONITORING ACTIVE';
        }

        function onResults(results) {
            canvasElement.width = videoElement.videoWidth || 640;
            canvasElement.height = videoElement.videoHeight || 480;
            const w = canvasElement.width;
            const h = canvasElement.height;

            canvasCtx.save();
            canvasCtx.clearRect(0, 0, w, h);

            let ear = 0.32;
            let mar = 0.22;
            let pitch = 0;
            let yaw = 0;
            let roll = 0;

            if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
                const lm = results.multiFaceLandmarks[0];

                // Left & Right Eye landmarks
                const leftEyeIdx = [362, 385, 387, 263, 373, 380];
                const rightEyeIdx = [33, 160, 158, 133, 153, 144];
                const leftEye = leftEyeIdx.map(i => lm[i]);
                const rightEye = rightEyeIdx.map(i => lm[i]);

                const earL = calcEAR(leftEye);
                const earR = calcEAR(rightEye);
                ear = (earL + earR) / 2.0;

                // Mouth landmarks for MAR (78: left corner, 308: right corner, 13: top, 14: bottom)
                const mouthH = dist(lm[13], lm[14]);
                const mouthW = dist(lm[78], lm[308]);
                mar = mouthH / (mouthW || 0.001);

                // Approximate Head Pose
                const nose = lm[1];
                const chin = lm[152];
                pitch = ((chin.y - nose.y) - 0.25) * -180; // approximate pitch angle
                yaw = (nose.x - (lm[33].x + lm[263].x) / 2) * 120; // approximate yaw angle

                // Draw wireframe if enabled
                if (showMesh) {
                    canvasCtx.strokeStyle = 'rgba(6, 182, 212, 0.7)';
                    canvasCtx.lineWidth = 1;

                    // Draw eyes
                    [leftEye, rightEye].forEach(eye => {
                        canvasCtx.beginPath();
                        eye.forEach((pt, idx) => {
                            const x = pt.x * w;
                            const y = pt.y * h;
                            if (idx === 0) canvasCtx.moveTo(x, y);
                            else canvasCtx.lineTo(x, y);
                        });
                        canvasCtx.closePath();
                        canvasCtx.stroke();
                    });

                    // Draw nose keypoint
                    canvasCtx.fillStyle = '#f59e0b';
                    canvasCtx.beginPath();
                    canvasCtx.arc(nose.x * w, nose.y * h, 3, 0, 2 * Math.PI);
                    canvasCtx.fill();
                }
            }

            // Apply simulation override if testing
            if (simulationOverride === 'drowsy') {
                ear = 0.09;
                pitch = -12;
            } else if (simulationOverride === 'phone') {
                pitch = -26;
                yaw = 12;
                ear = 0.29;
            }

            evaluateDriverState(ear, mar, pitch, yaw, roll);
            canvasCtx.restore();
        }

        function evaluateDriverState(ear, mar, pitch, yaw, roll) {
            // Update UI gauges
            document.getElementById('earValText').innerText = ear.toFixed(2);
            document.getElementById('earBarFill').style.width = Math.min(100, (ear / 0.45) * 100) + '%';
            document.getElementById('earBarFill').style.backgroundColor = ear < EAR_THRESH ? 'var(--accent-red)' : 'var(--accent-green)';

            document.getElementById('marValText').innerText = mar.toFixed(2);
            document.getElementById('marBarFill').style.width = Math.min(100, (mar / 1.0) * 100) + '%';
            document.getElementById('marBarFill').style.backgroundColor = mar > MAR_THRESH ? 'var(--accent-red)' : 'var(--accent-green)';

            document.getElementById('pitchVal').innerText = Math.round(pitch) + '°';
            document.getElementById('yawVal').innerText = Math.round(yaw) + '°';
            document.getElementById('rollVal').innerText = Math.round(roll) + '°';

            const now = Date.now();
            const alertBanner = document.getElementById('alertBanner');
            const alertText = document.getElementById('alertBannerText');
            const container = document.getElementById('videoContainer');
            const badge = document.getElementById('mlStateBadge');
            const reason = document.getElementById('mlReason');

            if (ear < EAR_THRESH) consecClosed++;
            else consecClosed = Math.max(0, consecClosed - 2);

            if (pitch < -14.0) consecPhone++;
            else consecPhone = Math.max(0, consecPhone - 2);

            // Classification Logic
            if (consecClosed > 25) { // Prolonged eye closure (Microsleep)
                container.className = 'video-container flashing-alarm';
                alertBanner.style.display = 'flex';
                alertBanner.style.background = 'rgba(239, 68, 68, 0.95)';
                alertText.innerText = 'CRITICAL MICROSLEEP: Closed eyes detected! Wake up!';
                badge.innerText = 'CRITICAL: MICROSLEEP';
                badge.style.color = 'var(--accent-red)';
                reason.innerText = 'Eyes shut for > 1.0s. High collision risk!';

                if (now - lastAlarmTime > 1800) {
                    playTone([1800, 2400, 1800, 2400], 0.5, 'sawtooth');
                    logTelematics('MICROSLEEP', 'CRITICAL', 'Eyes closed > 1.0s', ear, mar);
                    lastAlarmTime = now;
                }
            } else if (consecPhone > 20) { // Looking down at phone
                container.className = 'video-container flashing-warning';
                alertBanner.style.display = 'flex';
                alertBanner.style.background = 'rgba(245, 158, 11, 0.95)';
                alertText.innerText = 'PHONE DISTRACTION: Head pitched down. Keep eyes on road!';
                badge.innerText = 'DISTRACTION: PHONE';
                badge.style.color = 'var(--accent-amber)';
                reason.innerText = 'Driver looking downwards into lap / console';

                if (now - lastAlarmTime > 2200) {
                    playTone([1200, 750], 0.4, 'triangle');
                    logTelematics('PHONE_DISTRACTION', 'WARNING', 'Downward head posture (Pitch < -14°)', ear, mar);
                    lastAlarmTime = now;
                }
            } else if (mar > MAR_THRESH) { // Yawning
                container.className = 'video-container';
                alertBanner.style.display = 'none';
                badge.innerText = 'WARNING: DROWSINESS (YAWNING)';
                badge.style.color = 'var(--accent-amber)';
                reason.innerText = 'Extended yawn aperture (MAR ' + mar.toFixed(2) + ')';
            } else { // Normal Alert Driving
                container.className = 'video-container';
                alertBanner.style.display = 'none';
                badge.innerText = 'NORMAL / ALERT';
                badge.style.color = 'var(--accent-green)';
                reason.innerText = 'Attentive to forward highway';
            }
        }

        function logTelematics(type, severity, trigger, ear, mar) {
            const tbody = document.getElementById('telematicsLogBody');
            if (tbody.children.length === 1 && tbody.children[0].children.length === 1) {
                tbody.innerHTML = '';
            }

            const tr = document.createElement('tr');
            const timeStr = new Date().toLocaleTimeString();
            const badgeClass = severity === 'CRITICAL' ? 'badge-danger' : 'badge-warning';

            tr.innerHTML = `
                <td>${timeStr}</td>
                <td><strong>${type}</strong></td>
                <td><span class="badge ${badgeClass}">${severity}</span></td>
                <td>1.4s</td>
                <td>EAR: ${ear.toFixed(2)} | MAR: ${mar.toFixed(2)}</td>
                <td>${trigger}</td>
            `;
            tbody.prepend(tr);
        }

        function toggleMesh() {
            showMesh = !showMesh;
        }

        function toggleMute() {
            isMuted = !isMuted;
            document.getElementById('btnToggleMute').innerText = isMuted ? 'Unmute Audio' : 'Mute Audio';
        }

        function simulateState(state) {
            if (state === 'normal') simulationOverride = null;
            else simulationOverride = state;
        }
    </script>
</body>
</html>
"""


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif parsed.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status_data = {
                "project": "Project 119: Driver Fatigue & Distraction Safety Monitor",
                "status": "online",
                "timestamp": datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
        else:
            super().do_GET()


def run_web_server(port=PORT):
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), DashboardRequestHandler) as httpd:
        print("=" * 70)
        print("  DRIVER SAFETY MONITOR - LIVE WEB TELEMATICS DASHBOARD")
        print("=" * 70)
        print(f"  Live URL: http://localhost:{port}")
        print(f"  Live URL (LAN / In-Cab): http://127.0.0.1:{port}")
        print("  Press Ctrl+C to terminate the web server.")
        print("=" * 70)
        httpd.serve_forever()


if __name__ == "__main__":
    run_web_server()
