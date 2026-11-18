"""Runtime preparation shared by every way of starting the engine (source checkout, launcher, packaged exe).

Makes FFmpeg findable (a bundled `bin` folder wins over whatever is on PATH), keeps third-party noise
quiet, and creates the folders the engine writes to.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from engine.config import ROOT

FROZEN = bool(getattr(sys, "frozen", False))
BUNDLE = Path(getattr(sys, "_MEIPASS", ROOT))            # where PyInstaller unpacked read-only resources


def hide_child_consoles() -> None:
    """Windows: the app runs without a console (pythonw / windowed exe), so every console program it starts
    (ffmpeg, ffprobe, demucs, yt-dlp's helpers) would otherwise pop up its own terminal window. A batch of
    100 clips meant hundreds of them. Patch Popen once so every child is created with no window."""
    if os.name != "nt":
        return
    import subprocess

    if getattr(subprocess.Popen, "_gta6_no_window", False):
        return
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    _orig = subprocess.Popen.__init__

    def _init(self, *args, **kw):
        kw["creationflags"] = kw.get("creationflags", 0) | flag
        _orig(self, *args, **kw)

    subprocess.Popen.__init__ = _init
    subprocess.Popen._gta6_no_window = True


def prepare() -> None:
    hide_child_consoles()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    # a bundled FFmpeg (next to the exe, or inside the bundle) takes priority over PATH
    for d in (ROOT / "bin", BUNDLE / "bin"):
        if (d / "ffmpeg.exe").exists() or (d / "ffmpeg").exists():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            break
    for sub in ("config", "database", "storage", "library/songs", "library/artists", "logs", "models"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)


def first_run(log=print) -> None:
    """Pick hardware-appropriate settings the first time the engine starts (same as `clipper setup`, without the CLI)."""
    from engine.config import CONFIG_PATH, SECRETS_PATH, load_settings, save_settings
    from engine.db import Database
    from engine.hardware import detect_environment, recommend

    if not SECRETS_PATH.exists():
        SECRETS_PATH.write_text("# Optional credentials. The AI key is normally set from the dashboard Settings tab.\n"
                                "# ANTHROPIC_API_KEY=\n# OPENAI_API_KEY=\n# TWITCH_CLIENT_ID=\n# TWITCH_CLIENT_SECRET=\n"
                                "# YOUTUBE_API_KEY=\n", encoding="utf-8")
    s = load_settings()
    if not CONFIG_PATH.exists():
        log("First run: detecting your hardware...")
        env = detect_environment(str(ROOT))
        rec = recommend(env)
        s.transcription.model, s.transcription.device, s.transcription.compute_type = (
            rec.whisper_model, rec.whisper_device, rec.whisper_compute)
        s.max_concurrent_jobs, s.video.encoder = rec.max_concurrent_jobs, rec.encoder
        save_settings(s)
        for n in rec.notes:
            log("  " + n)
    Database(s.db_path)
    try:                                   # the face model is not shipped in the source repo: fetch it on first run
        from engine.video.webcam import ensure_face_model
        if not ensure_face_model():
            log("Could not download the face model (no internet?). Clips will use the full-frame layout until it is available.")
    except Exception:
        pass


def ffmpeg_ok() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
