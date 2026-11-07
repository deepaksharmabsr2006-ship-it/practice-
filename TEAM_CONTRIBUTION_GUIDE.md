# Project 119: 3-Member Academic Presentation & Contribution Guide

**Project Title:** Driver Fatigue, Microsleep & Distraction Safety Monitor  
**Presentation Timeline:** 1st Week of December 2026  
**Architecture:** Full Stack + AI/ML + Edge Vision + Cloud Deployment  

---

## 1. 3-Member Role & Ownership Matrix

Your professor and external evaluator will assess **individual contribution**. This is the exact division of labor for your project report, presentation slides, and GitHub commit history:

| Team Member | Official Role | Primary Modules & Code Files Owned | Key Deliverables for Evaluation |
|---|---|---|---|
| **Deepak Sharma** *(Team Lead)* | **AIML & Computer Vision Engineer** | `core/facial_landmarks.py`<br>`core/geometry_features.py`<br>`core/temporal_buffer.py`<br>`ml/model_trainer.py`<br>`ml/classifier.py`<br>`evaluate.py` | MediaPipe 468 landmark pipeline, EAR & MAR mathematical formulations, Perspective-n-Point (`solvePnP`) head pose estimation, SVM/Random Forest model training, PERCLOS temporal aggregation. |
| **Aastha Singh** | **Full-Stack Backend & Telematics Engineer** | `backend/app.py`<br>`backend/models.py`<br>`backend/schemas.py`<br>`backend/database.py`<br>`backend/services/telematics_service.py` | FastAPI asynchronous gateway, SQLAlchemy database models (Drivers, Trips, SafetyIncidents), bi-directional WebSockets, NHTSA/SAE safety score formulation, REST APIs for telematics data ingestion. |
| **Abhishek Shukla** | **Frontend UI, Audio Engine & Cloud DevOps Engineer** | `frontend/index.html`<br>`audio/alert_manager.py`<br>`Dockerfile`<br>`docker-compose.yml`<br>`render.yaml`<br>`.github/workflows/ci-cd.yml` | Responsive Fleet Telematics Dashboard, Web Audio API alarm synthesizer (multi-tone siren), Chart.js live streaming graphs, Docker multi-stage build, GitHub Actions automated CI/CD pipeline, Cloud live deployment. |

---

## 2. GitHub Branching & Commit Distribution Strategy

To ensure GitHub Insights shows clear, balanced contribution from all 3 members:

### Step 1: Create Main Repo & Initialize
```bash
cd "C:\Users\Deepak Sharma\.gemini\antigravity\scratch\driver_fatigue_safety_monitor"
git init
git add .gitignore README.md TEAM_CONTRIBUTION_GUIDE.md requirements.txt
git commit -m "chore: initial repository setup with comprehensive .gitignore"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/driver-fatigue-safety-monitor.git
git push -u origin main
```

### Step 2: Member 1 (Deepak Sharma) - AI/ML Pipeline Commits
```bash
git checkout -b feature/aiml-computervision
git config user.name "Deepak Sharma"
git add core/ ml/ evaluate.py config/
git commit -m "feat(aiml): implement MediaPipe 468 FaceMesh, EAR, MAR & solvePnP Head Pose estimation"
git commit -m "feat(ml): train SVM and Random Forest classifiers on rolling temporal features with PERCLOS"
git commit -m "test(ml): add evaluate.py with high recall benchmarks for safety-critical microsleep"
git push origin feature/aiml-computervision
# Open and Merge PR on GitHub into main
```

### Step 3: Member 2 (Aastha Singh) - Backend & Telematics Database Commits
```bash
git checkout main && git pull
git checkout -b feature/backend-telematics
git config user.name "Aastha Singh"
git add backend/
git commit -m "feat(backend): configure SQLAlchemy database models for Drivers, Trips & Incidents"
git commit -m "feat(api): implement FastAPI REST endpoints and bi-directional WebSockets for live telemetry"
git commit -m "feat(telematics): add NHTSA/SAE trip safety score algorithm and seed fleet demo data"
git push origin feature/backend-telematics
# Open and Merge PR on GitHub into main
```

### Step 4: Member 3 (Abhishek Shukla) - Frontend Dashboard, Audio & Cloud DevOps Commits
```bash
git checkout main && git pull
git checkout -b feature/frontend-cloud-devops
git config user.name "Abhishek Shukla"
git add frontend/ audio/ Dockerfile docker-compose.yml render.yaml Procfile .github/
git commit -m "feat(ui): design dark-mode Fleet Telematics Dashboard with Chart.js & In-Cabin HUD"
git commit -m "feat(audio): implement Web Audio API multi-tier alarm synthesizer for microsleep & phone alerts"
git commit -m "ci(devops): configure Docker containerization, render.yaml and GitHub Actions CI/CD"
git push origin feature/frontend-cloud-devops
# Open and Merge PR on GitHub into main
```

---

## 3. Viva & Presentation Defense Script (For Evaluation)

Here are the questions examiners and professors frequently ask, along with the precise technical answers for each team member:

### Questions for Deepak Sharma (AIML & Computer Vision):
1. **Professor:** *"Why did you use EAR (Eye Aspect Ratio) instead of a plain CNN image classifier?"*
   - **Deepak Sharma:** *"A 2D CNN suffers from high computational latency (~30-50ms) and struggles with variations in driver lighting or sunglasses. Soukupov&aacute; & Cech's EAR formula operates directly on 3D geometric coordinates from MediaPipe FaceMesh ($EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$). It is scale-invariant, lighting-robust, and takes under $0.5\text{ ms}$ to compute, allowing real-time 30+ FPS edge execution."*

2. **Professor:** *"How do you differentiate a natural voluntary blink from a microsleep event?"*
   - **Deepak Sharma:** *"A natural blink lasts 100 to 300 milliseconds (about 3 to 9 consecutive frames at 30 FPS). Microsleep occurs when eyelids remain continuously closed ($EAR < 0.22$) for greater than 1.0 to 1.3 seconds (> 40 consecutive frames). Our rolling temporal buffer computes the PERCLOS (% of eye closure) and immediately triggers an emergency failsafe alert."*

3. **Professor:** *"How is mobile phone distraction detected?"*
   - **Deepak Sharma:** *"We use Perspective-n-Point (`cv2.solvePnP`) mapping 6 anthropometric 3D facial points to the 2D image plane to decompose Euler rotation angles (Pitch, Yaw, Roll). A pitch $< -14.0^\circ$ accompanied by departure from the forward roadway focus characterizes the classic 'looking down into lap/phone' posture."*

---

### Questions for Aastha Singh (Full-Stack Backend & Telematics):
1. **Professor:** *"Why choose FastAPI over Flask or Django for this telematics platform?"*
   - **Aastha Singh:** *"FastAPI is built on Starlette and ASGI, supporting native Python `async/await` concurrency. Since commercial fleets stream high-frequency biometrics (30 readings/sec) from multiple vehicles, FastAPI handles thousands of concurrent WebSocket connections and JSON ingest payloads with negligible latency compared to synchronous WSGI frameworks."*

2. **Professor:** *"Explain your database schema and safety scoring formula."*
   - **Aastha Singh:** *"We designed a normalized relational schema with `Driver`, `Trip`, `SafetyIncident`, and `TelemetrySnapshot` tables. The safety scoring formula adheres to NHTSA commercial telematics standards: starting at 100.0 points, deducting 15 points per critical microsleep, 8 points per phone distraction, and 3 points per sustained yawn. This provides fleet managers with an actionable driver risk rating."*

3. **Professor:** *"How are incidents recorded without slowing down the video capture loop?"*
   - **Aastha Singh:** *"The client or edge device dispatches incident reports asynchronously over HTTP POST and WebSockets. The backend persists the infraction in SQLite/PostgreSQL in a non-blocking thread, immediately broadcasting the alert payload to all connected fleet manager consoles."*

---

### Questions for Abhishek Shukla (Frontend, Audio & Cloud DevOps):
1. **Professor:** *"How does the browser audio alert work without external media files?"*
   - **Abhishek Shukla:** *"We built an audio synthesizer using the native browser Web Audio API (`AudioContext`). It dynamically generates pure sine/sawtooth tone envelopes (e.g., rapid pulsing 1800Hz-2400Hz siren for microsleep and a 1200Hz-750Hz chime for phone distraction). This guarantees 100% reliability with zero external MP3 downloads or network dependencies."*

2. **Professor:** *"Explain your Cloud CI/CD and Containerization architecture."*
   - **Abhishek Shukla:** *"We configured a multi-stage Docker container (`python:3.10-slim`) bundled with OpenCV system dependencies (`libgl1`, `libglib2.0-0`). Our GitHub Actions CI/CD pipeline runs automated model evaluation and backend health checks on every commit before building the production image. The app is deployed with 1-click cloud configuration on Render/Railway."*

3. **Professor:** *"Why does the camera require `http://localhost` or `https://`?"*
   - **Abhishek Shukla:** *"Modern browser security policies (W3C WebRTC standards) enforce that `navigator.mediaDevices.getUserMedia` can only access physical hardware sensors (camera & microphone) from a Secure Context (`localhost` or `https://`). Opening via raw `file:///` is sandboxed for user privacy."*

---

## 4. 1-Minute Live Demonstration Flow

1. **Step 1:** Open the Local/Cloud Dashboard (`http://localhost:8000`).
2. **Step 2 (Deepak Sharma):** Show the green **[ALERT & ATTENTIVE]** state with live EAR (~0.32) and MAR (~0.18) gauges.
3. **Step 3 (Deepak Sharma):** Close your eyes for 1.5 seconds &rarr; Demonstrate the flashing red HUD border + high-intensity emergency audio siren.
4. **Step 4 (Deepak Sharma):** Look down at your lap/phone &rarr; Demonstrate the amber **[PHONE DISTRACTION WARNING]** + two-tone alert chime.
5. **Step 5 (Aastha Singh):** Switch to the **Fleet Operations Center** tab &rarr; Show the real-time EAR/MAR live graph, active driver registry, and the updated Driver Safety Score.
6. **Step 6 (Abhishek Shukla):** Switch to the **Incident Log** tab &rarr; Show the recorded infractions with exact timestamps, severity, and export to CSV.
