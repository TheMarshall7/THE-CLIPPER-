"""Human-readable names for finished clips.

Every clip gets a folder named  "<Stream title> - <Song> - Clip 03"  (clip number counts within its stream),
kept under  <data folder>/Clips/ . The engine's own working folders (storage/clips/00005) stay as they are.
"""
from __future__ import annotations

import re
from pathlib import Path

from engine.config import ROOT

LIBRARY = ROOT / "Clips"
NO_SONG = "No song"


def safe(text: str | None, limit: int) -> str:
    """Make `text` safe for a Windows file name and cap its length."""
    t = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text or "")
    t = re.sub(r"\s+", " ", t).strip(" .-_")
    if len(t) > limit:
        t = t[:limit].rstrip(" .-_") + "…"
    return t


def clip_number(db, clip_id: int, stream_id: int) -> int:
    return db.one("SELECT COUNT(*) FROM clips WHERE stream_id=? AND id<=?", (stream_id, clip_id))[0]


def song_title(db, clip_id: int) -> str | None:
    """The song actually mixed into the clip's current render (its chosen placement)."""
    row = db.one("SELECT s.title FROM music_placements mp JOIN songs s ON s.id=mp.song_id "
                 "WHERE mp.clip_id=? AND mp.chosen=1 ORDER BY mp.id DESC", (clip_id,))
    return row["title"] if row else None


def clip_label(db, clip_id: int) -> str:
    clip = db.one("SELECT stream_id FROM clips WHERE id=?", (clip_id,))
    stream = db.one("SELECT s.title, st.name streamer FROM streams s LEFT JOIN streamers st ON st.id=s.streamer_id WHERE s.id=?",
                    (clip["stream_id"],))
    title = safe(stream["title"] or stream["streamer"], 48) or f"Video {clip['stream_id']}"
    song = safe(song_title(db, clip_id), 28) or NO_SONG
    return f"{title} - {song} - Clip {clip_number(db, clip_id, clip['stream_id']):02d}"


def inside(path: Path, root: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except Exception:
        return False
