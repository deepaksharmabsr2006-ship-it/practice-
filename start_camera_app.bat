@echo off
title Driver Safety Telematics - Live Camera Launcher
echo ======================================================================
echo   PROJECT 119: DRIVER FATIGUE & DISTRACTION SAFETY MONITOR
echo   Team: Deepak Sharma (Lead), Aastha Singh, Abhishek Shukla
echo ======================================================================
echo.
echo [1/3] Navigating to project directory...
cd /d "%~dp0"

echo [2/3] Starting Local Full-Stack Web & Telematics Server on port 8000...
start /b uvicorn backend.app:app --host 0.0.0.0 --port 8000

echo [3/3] Opening Google Chrome / Default Browser at http://localhost:8000 ...
timeout /t 2 /nobreak >nul
start http://localhost:8000

echo.
echo ======================================================================
echo   LIVE SERVER RUNNING!
echo   Web Dashboard & Webcam Live Feed: http://localhost:8000
echo   Swagger Interactive API Docs:    http://localhost:8000/docs
echo   Press Ctrl+C in this window to stop the server when finished.
echo ======================================================================
echo.
pause
