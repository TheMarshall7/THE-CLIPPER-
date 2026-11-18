"""GTA6Clipper - self-contained program (no Python, no Claude Code, nothing else to install).

Entry point of the packaged build (scripts/build_standalone.py). It runs the dashboard web server inside
this process and opens your browser. Data (settings, database, clips, songs) lives next to the exe.
Set GTA6_DATA_DIR to keep that data somewhere else (e.g. an existing project folder).
"""
import multiprocessing
import os
import socket
import sys
import time
import urllib.request
import webbrowser

BASE_PORT = 8765


def ours(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)
        return True
    except Exception:
        return False


def free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def main() -> int:
    multiprocessing.freeze_support()
    os.system("title GTA 6 Clip Engine")
    print("=" * 60)
    print("  GTA 6 Clip Engine")
    print("=" * 60)
    from engine import runtime
    from engine.config import ROOT

    runtime.prepare()
    for port in range(BASE_PORT, BASE_PORT + 10):      # already running? just open it
        if ours(port):
            print(f"Already running. Opening http://127.0.0.1:{port}")
            webbrowser.open(f"http://127.0.0.1:{port}")
            time.sleep(2)
            return 0
    port = next((p for p in range(BASE_PORT, BASE_PORT + 10) if free(p)), None)
    if port is None:
        print(f"Ports {BASE_PORT}-{BASE_PORT + 9} are in use. Close other programs and try again.")
        input("Press Enter to close...")
        return 1
    print(f"Data folder: {ROOT}")
    try:
        runtime.first_run()
    except Exception as e:                              # never block startup on hardware detection
        print(f"Setup note: {e}")
    if not runtime.ffmpeg_ok():
        print("\n!! FFmpeg was not found (expected in the 'bin' folder next to this program).")
        print("   Clip processing needs it. Put ffmpeg.exe and ffprobe.exe in that folder.\n")
    from engine.dashboard import server

    print(f"\nDashboard: http://127.0.0.1:{port}")
    print("Keep this window open while you work. Close it, or press Quit on the page, to stop.\n")
    try:
        server.main(port=port, open_browser=True)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
