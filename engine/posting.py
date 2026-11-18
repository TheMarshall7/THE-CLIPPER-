"""Posting helpers: ready-to-paste captions per platform, export folders, best posting times, and results tracking.

Clips you mark "Post" in the Review tab land here. Once you log how a posted clip did (views, likes), the engine
learns your best posting hours and which kinds of clips perform.
"""
from __future__ import annotations

import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from engine.db import Database, loads

PLATFORMS = {"tiktok": "TikTok", "instagram": "Instagram Reels", "youtube_shorts": "YouTube Shorts"}

# general guidance (local time) until your own results say otherwise
DEFAULT_WINDOWS = {
    "tiktok": "6-10 pm weekdays, 9 am-12 pm weekends",
    "instagram": "11 am-1 pm and 7-9 pm",
    "youtube_shorts": "12-3 pm and 7-9 pm",
}


def to_post(db: Database, posted: bool = False) -> list[dict]:
    rows = db.query("SELECT c.*, r.posted, r.posted_at, r.posted_platform FROM clips c JOIN clip_review r ON r.clip_id=c.id "
                    "WHERE r.decision='post' AND r.posted=? ORDER BY r.decided_at DESC", (1 if posted else 0,))
    return [dict(r) for r in rows]


def caption(clip: dict, platform: str) -> str:
    meta = loads(clip.get("metadata_json"), {}) or {}
    hooks = loads(clip.get("hooks_json"), []) or []
    var = (meta.get("platform_variants") or {}).get(platform) or {}
    text = var.get("caption") or meta.get("caption") or (hooks[0] if hooks else meta.get("title", ""))
    song = meta.get("song") or {}
    extra = []
    if song.get("credit"):
        extra.append(song["credit"])
    if song.get("cta"):
        extra.append(song["cta"])
    tags = " ".join(h for h in song.get("hashtags", []) if h not in text)
    out = text.strip()
    if tags:
        out += " " + tags
    if extra:
        out += "\n" + "\n".join(extra)
    if meta.get("attribution") and meta["attribution"] not in out:
        out += "\n" + meta["attribution"]
    return out.strip()


def export(db: Database, settings, clip_id: int, platform: str) -> Path:
    """Copy the video + caption into exports/<date>/<platform>/<clip name>/ ready to upload."""
    from engine import naming

    clip = db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
    if not clip or not clip["video_path"]:
        raise ValueError("clip has no video")
    dest = settings.exports / time.strftime("%Y-%m-%d") / platform / naming.safe(naming.clip_label(db, clip_id), 100)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(clip["video_path"], dest / "video.mp4")
    (dest / "caption.txt").write_text(caption(dict(clip), platform), encoding="utf-8")
    return dest


def record_result(db: Database, clip_id: int, platform: str, views: int, likes: int = 0, shares: int = 0, comments: int = 0) -> None:
    db.execute("DELETE FROM clip_performance WHERE clip_id=? AND platform=?", (clip_id, platform))
    db.insert("clip_performance", {"clip_id": clip_id, "platform": platform, "views": views, "likes": likes, "shares": shares,
                                   "comments": comments, "recorded_at": time.time()})


def best_times(db: Database) -> dict[str, str]:
    """Per platform: the learned best posting hours (needs 4+ logged results), else the general window."""
    out = {p: f"{DEFAULT_WINDOWS[p]} (general guidance)" for p in PLATFORMS}
    rows = db.query("SELECT p.platform, p.views, r.posted_at FROM clip_performance p JOIN clip_review r ON r.clip_id=p.clip_id "
                    "WHERE r.posted_at IS NOT NULL AND p.views IS NOT NULL")
    by: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["platform"]][time.localtime(r["posted_at"]).tm_hour].append(int(r["views"]))
    for plat, hours in by.items():
        if plat in out and sum(len(v) for v in hours.values()) >= 4:
            ranked = sorted(hours.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True)[:2]
            out[plat] = "your best hours: " + ", ".join(f"{h % 12 or 12} {'am' if h < 12 else 'pm'}" for h, _ in ranked) + " (from your results)"
    return out
