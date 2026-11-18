"""GTA6Clipper desktop app entry point (packaged build and source checkout).

    GTA6Clipper.exe                                   opens the window
    GTA6Clipper.exe --selftest [video] [song] [log]   checks the packaged app end to end (no window), exit code 0 = healthy
"""
import multiprocessing
import os
import sys
import traceback
from pathlib import Path


def selftest(args: list[str]) -> int:
    import tempfile
    import time

    here = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
    video = Path(args[0]) if len(args) > 0 else None
    song = Path(args[1]) if len(args) > 1 else None
    logp = Path(args[2]) if len(args) > 2 else here / "selftest.log"
    os.environ.setdefault("GTA6_DATA_DIR", tempfile.mkdtemp(prefix="gta6_selftest_"))   # never touch real data
    out = open(logp, "w", encoding="utf-8")
    failed = []

    def say(m: str) -> None:
        out.write(m + "\n")
        out.flush()

    def check(name: str, fn) -> object:
        t = time.time()
        try:
            r = fn()
            say(f"PASS {name} ({time.time() - t:.1f}s): {r}")
            return r
        except Exception:
            failed.append(name)
            say(f"FAIL {name}\n{traceback.format_exc()}")
            return None

    from engine import runtime
    from engine.config import ROOT, load_settings

    say(f"frozen={getattr(sys, 'frozen', False)} data={ROOT}")
    runtime.prepare()
    check("first run setup", lambda: runtime.first_run(log=lambda m: None) or "ok")
    check("ffmpeg + ffprobe", lambda: runtime.ffmpeg_ok() or (_ for _ in ()).throw(RuntimeError("ffmpeg missing")))
    check("window toolkit (tkinter + customtkinter)", lambda: __import__("customtkinter").__version__)
    check("app modules", lambda: [__import__(m) for m in (
        "engine.gui", "engine.pipeline", "engine.dashboard.server", "engine.naming", "engine.music.detect", "yt_dlp",
        "faster_whisper", "librosa", "cv2", "soundfile")] and "all import")

    def face():
        from engine.video import webcam
        return type(webcam._detector()).__name__

    check("face detector (YuNet model)", face)

    def gpu():
        import ctranslate2
        return f"cuda devices: {ctranslate2.get_cuda_device_count()}"

    check("GPU speech engine", gpu)

    def music_scan():
        import numpy as np
        from engine.music import detect
        x = (np.random.default_rng(1).normal(0, 0.1, detect.SAMPLE_RATE * 8)).astype("float32")
        return f"{len(detect.analyze_block(x, None))} outputs"

    check("music analysis (librosa/numba)", music_scan)

    if video and video.exists():
        def pipeline():
            from engine.db import Database
            from engine.music.library import import_song
            from engine.pipeline import Engine, enqueue, run_queue
            s = load_settings()
            db = Database(s.db_path)
            slug = None
            if song and song.exists():
                sid = import_song(db, s.library / "songs", song, "Selftest Song", "Selftest", "selftest", "selftest", False, 3, {},
                                  s.transcription, log=lambda m: None)
                slug = db.one("SELECT slug FROM songs WHERE id=?", (sid,))["slug"]
            eng = Engine(s, log=lambda m: say("   " + m))
            sid = eng.add_source(str(video), "Selftest", "explicitly_authorized")
            enqueue(db, sid, {"max_clips": 2, "song": slug})
            res = run_queue(s, log=lambda m: say("   " + m))
            clips = db.query("SELECT id, status, error FROM clips")
            bad = [dict(c) for c in clips if c["status"] != "rendered"]
            if res.get("failed") or not clips or bad:
                raise RuntimeError(f"run={res} clips={len(clips)} not rendered={bad}")
            from engine import naming
            folders = sorted(p.name for p in naming.LIBRARY.iterdir()) if naming.LIBRARY.exists() else []
            if not folders:
                raise RuntimeError("no folders in the Clips library")
            return f"{len(clips)} clips rendered; folders: {folders}"

        check("full job: video -> clips (with song)", pipeline)
    say("RESULT: " + ("HEALTHY" if not failed else "FAILED: " + ", ".join(failed)))
    out.close()
    return 1 if failed else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if "--selftest" in sys.argv:
        code = selftest(sys.argv[sys.argv.index("--selftest") + 1:])
        sys.stdout.flush()
        os._exit(code)
    from engine.gui import main

    main()
    sys.exit(0)
