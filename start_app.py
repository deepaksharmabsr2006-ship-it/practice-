"""
Automated 1-Click Project Launcher.
Starts the FastAPI full-stack server on http://localhost:8000 and automatically
opens the system's default web browser with full WebRTC camera hardware permissions enabled.
"""

import sys
import time
import webbrowser
import subprocess
import os

PORT = 8000
URL = f"http://localhost:{PORT}"

def main():
    print("=" * 70)
    print("  PROJECT 119: DRIVER FATIGUE & DISTRACTION SAFETY MONITOR")
    print("  Team: Deepak Sharma (Lead) | Aastha Singh | Abhishek Shukla")
    print("=" * 70)
    print(f"\n[1/2] Opening default browser at {URL} ...")

    # Change to project root directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Open browser after short delay to give uvicorn time to bind
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(URL)

    import threading
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    print(f"[2/2] Launching FastAPI ASGI Telematics Server on port {PORT}...")
    try:
        import uvicorn
        uvicorn.run("backend.app:app", host="0.0.0.0", port=PORT, reload=False)
    except ImportError:
        print("\n[Notice] uvicorn not found in environment. Launching standard Python web server...")
        from web_app import run_web_server
        run_web_server(port=PORT)

if __name__ == "__main__":
    main()
