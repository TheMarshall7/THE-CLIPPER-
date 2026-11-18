"""Watch folder: drop a video file (or put links in links.txt) into the folder and it is queued automatically.

Files are only picked up once their size has stopped changing (so a half-copied VOD is never started), and each
one is queued exactly once (tracked in <folder>/.queued.json). The files themselves are never moved or deleted.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".flv", ".ts", ".m4v", ".avi"}
INTERVAL_S = 8.0


def resolve_folder(root: Path, folder: str) -> Path:
    p = Path(folder or "Watch")
    return p if p.is_absolute() else root / p


def scan_once(folder: Path, seen: dict, queue_fn: Callable[[str], None], log: Callable[[str], None] = lambda m: None) -> int:
    """Queue everything new in `folder`. `seen` carries file sizes between scans. Returns how many were queued."""
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / ".queued.json"
    try:
        done = set(json.loads(state_path.read_text(encoding="utf-8")))
    except Exception:
        done = set()
    queued = 0
    todo: list[tuple[str, str]] = []                 # (key, what to queue)
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            key = f"{f.name}|{f.stat().st_size}"
            if any(d.startswith(f.name + "|") for d in done):
                continue
            size = f.stat().st_size
            if seen.get(f.name) != size or time.time() - f.stat().st_mtime < 5:      # still being written
                seen[f.name] = size
                continue
            todo.append((key, str(f)))
    links = folder / "links.txt"
    if links.exists():
        for line in links.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip().strip('"')
            if line and not line.startswith("#") and f"link|{line}" not in done:
                todo.append((f"link|{line}", line))
    for key, what in todo:
        try:
            queue_fn(what)
            done.add(key)
            queued += 1
            log(f"Watch folder: queued {Path(what).name if not what.startswith('http') else what}")
        except Exception as e:                       # leave it out of `done` so it is retried
            log(f"Watch folder: could not queue {what}: {e}")
    if queued:
        state_path.write_text(json.dumps(sorted(done)), encoding="utf-8")
    return queued


class Watcher(threading.Thread):
    def __init__(self, root: Path, load_settings: Callable, queue_fn: Callable[..., dict], wake: Callable[[], None],
                 log: Callable[[str], None]):
        super().__init__(daemon=True, name="watch-folder")
        self.root, self.load_settings, self.queue_fn, self.wake, self.log = root, load_settings, queue_fn, wake, log
        self.seen: dict = {}
        self.stop_flag = threading.Event()
        self.last_error = ""

    def run(self) -> None:
        while not self.stop_flag.wait(INTERVAL_S):
            try:
                w = self.load_settings().watch
                if not w.enabled:
                    continue

                def queue(src: str) -> None:
                    self.queue_fn(src, None, None, None, w.max_clips or None, "auto", w.song or None)

                if scan_once(resolve_folder(self.root, w.folder), self.seen, queue, self.log):
                    self.wake()
            except Exception as e:                   # the watcher must never die
                if str(e) != self.last_error:
                    self.last_error = str(e)
                    from engine.errors import friendly
                    self.log(f"Watch folder problem: {friendly(e)}")
