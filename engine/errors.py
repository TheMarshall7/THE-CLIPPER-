"""One place for turning crashes into something a person can act on.

  friendly(exc)       plain-language explanation of what went wrong and what to try
  log_exception(...)  full traceback appended to logs/errors.log (so nothing is ever lost silently)
  install_hooks(...)  catches errors in background threads and anywhere else nothing else would
"""
from __future__ import annotations

import errno
import sqlite3
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

from engine.config import ROOT

LOG_PATH = ROOT / "logs" / "errors.log"
_last_seen: dict[str, float] = {}


def friendly(exc: BaseException) -> str:
    """What happened, in plain words. Never raises."""
    try:
        name = type(exc).__name__
        text = str(exc).strip()
        low = text.lower()
        if isinstance(exc, FileNotFoundError) and ("ffmpeg" in low or "ffprobe" in low):
            return "FFmpeg was not found. It should be in the 'bin' folder next to the app; re-install or restore that folder."
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
            return "The disk is full. Free some space (clear old clips or the storage folder) and try again."
        if isinstance(exc, FileNotFoundError):
            return f"A file or folder could not be found: {getattr(exc, 'filename', None) or text[:120]}"
        if isinstance(exc, PermissionError):
            return "Windows blocked access to a file. Close any program using it (a video player, antivirus scan) and try again."
        if isinstance(exc, MemoryError):
            return "The computer ran out of memory. Close other programs or lower 'clips at the same time' and try again."
        if isinstance(exc, sqlite3.OperationalError) and "locked" in low:
            return "The database was busy. Try again in a few seconds."
        if "ffmpeg failed" in low or "ffprobe failed" in low or name == "MediaError":
            return f"A video/audio file could not be processed ({text[:160]}). The source may be damaged or unsupported."
        if name in ("ConnectionError", "Timeout", "ConnectTimeout", "ReadTimeout", "URLError", "gaierror") or \
                isinstance(exc, (ConnectionError, TimeoutError)) or "timed out" in low or "name resolution" in low:
            return "Network problem. Check your internet connection and try again."
        if name == "AIError" or "api key" in low or " 401" in low or " 403" in low:
            if "429" in low or "quota" in low or "rate" in low:
                return "The AI service is rate-limiting you or out of credit. Wait a bit or check your plan; the engine can run without AI."
            return "The AI service rejected the request. Check the API key in Settings (the light there shows if it works)."
        if name == "DownloadError" or "yt_dlp" in low or "unable to download" in low:
            return "That video could not be downloaded (private, removed, region-locked, or the link is wrong)."
        if name == "SourceError":
            return text or "That source could not be used."
        if isinstance(exc, (IndexError, KeyError, AttributeError, TypeError, ValueError, ZeroDivisionError)):
            return f"Unexpected problem ({name}: {text[:120]}). Details were saved to logs/errors.log."
        return f"{name}: {text[:200]}" if text else name
    except Exception:
        return "Something went wrong. Details were saved to logs/errors.log."


def log_exception(where: str, exc: BaseException, tb: Optional[object] = None) -> None:
    """Append the full traceback to logs/errors.log. Never raises."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > 2_000_000:          # keep it from growing forever
            LOG_PATH.replace(LOG_PATH.with_suffix(".old.log"))
        lines = traceback.format_exception(type(exc), exc, tb or exc.__traceback__)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {where} ===\n{''.join(lines)}")
    except Exception:
        pass


def should_notify(exc: BaseException, cooldown: float = 20.0) -> bool:
    """Don't pop a dialog for the same problem over and over (a failing timer would otherwise bury the user)."""
    key = f"{type(exc).__name__}:{str(exc)[:80]}"
    now = time.time()
    if now - _last_seen.get(key, 0) < cooldown:
        return False
    _last_seen[key] = now
    return True


def install_hooks(report: Callable[[str, BaseException], None]) -> None:
    """Route uncaught errors (any thread, or outside the window's own callbacks) to `report(where, exc)`."""
    def on_thread(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit or args.exc_value is None:
            return
        log_exception(f"thread {args.thread.name if args.thread else '?'}", args.exc_value, args.exc_traceback)
        report("background task", args.exc_value)

    def on_main(exc_type, exc, tb) -> None:
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            return sys.__excepthook__(exc_type, exc, tb)
        log_exception("app", exc, tb)
        report("app", exc)

    threading.excepthook = on_thread
    sys.excepthook = on_main
