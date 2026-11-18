"""GTA6Clipper.exe - double-click launcher for the dashboard. No Claude Code needed.

A small exe that lives in the project folder. It uses the project's own .venv (where
FFmpeg-adjacent Python packages, Whisper and the CUDA libraries are installed), makes
sure first-run setup is done, starts the dashboard, opens the browser, and stops the
engine when this window closes (or when you press Quit on the page).
"""
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

BASE_PORT = 8765


def project_root() -> Path:
    here = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    for d in (here, here.parent, here.parent.parent):
        if (d / "engine").is_dir() and (d / ".venv").is_dir():
            return d
    return here


def ours(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)
        return True
    except Exception:
        return False


def free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def pause(msg: str = "Press Enter to close...") -> None:
    try:
        input(msg)
    except EOFError:
        time.sleep(5)


def main() -> int:
    """Open the desktop window (engine.gui) with the project's own Python; no console window."""
    root = project_root()
    pyw = root / ".venv" / "Scripts" / "pythonw.exe"
    if not pyw.exists():
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0, f"Could not find the engine's Python at {pyw}. Keep GTA6Clipper.exe inside the gta6-clip-engine folder.",
            "GTA 6 Clip Engine", 0x10)
        return 1
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    subprocess.Popen([str(pyw), "-m", "engine.gui"], cwd=str(root), env=env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
