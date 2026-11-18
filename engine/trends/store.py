"""Reference clips: import, metrics, normalized performance (PRD M15).

Sources are only what the user supplies or an official API returns: pasted URLs / saved files, and
(optionally) the YouTube Data API. Metrics that are unavailable are stored as null, never estimated.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

from engine.db import Database, loads
from engine.ingestion.sources import SourceError, _ydl, classify
from engine.trends import analyze


def _date(upload: Optional[str]) -> Optional[str]:
    if upload and len(upload) == 8:
        return f"{upload[:4]}-{upload[4:6]}-{upload[6:]}"
    return upload


def import_reference(db: Database, settings, source: str, streamer: Optional[str] = None,
                     moment_type: Optional[str] = None, log=print) -> int:
    """Download (or open) a reference clip, measure it, store it. Returns reference_clips.id."""
    root = settings.storage / "trends"
    root.mkdir(parents=True, exist_ok=True)
    platform = classify(source)
    metrics: dict = {"views": None, "likes": None, "comments": None, "shares": None, "saves": None,
                     "followers": None, "views_per_day": None}
    account = publish = None
    if platform == "file":
        path = Path(source).resolve()
        url = str(path)
    else:
        try:
            with _ydl({"skip_download": False, "format": "best[height<=1080]/best", "outtmpl": str(root / "%(id)s.%(ext)s"),
                       "noplaylist": True}) as ydl:
                raw = ydl.extract_info(source, download=True)
                path = Path(ydl.prepare_filename(raw))
        except Exception as e:
            raise SourceError(f"could not fetch reference clip: {e}") from e
        url = raw.get("webpage_url") or source
        account = raw.get("uploader") or raw.get("channel")
        publish = _date(raw.get("upload_date"))
        metrics.update({"views": raw.get("view_count"), "likes": raw.get("like_count"), "comments": raw.get("comment_count"),
                        "followers": raw.get("channel_follower_count")})
        metrics["views_per_day"] = _per_day(metrics["views"], publish)
    existing = db.one("SELECT id FROM reference_clips WHERE url=?", (url,))
    row = {"url": url, "platform": platform, "account": account, "publish_date": publish,
           "metrics_json": metrics, "moment_type": moment_type, "streamer": streamer}
    vis, aud = analyze.analyze_file(path)
    row["duration"] = vis.get("duration_s")
    if existing:
        rid = existing["id"]
        db.update("reference_clips", rid, {k: v for k, v in row.items() if k != "url"})
        db.execute("DELETE FROM reference_clip_features WHERE reference_clip_id=?", (rid,))
    else:
        row["created_at"] = time.time()
        rid = db.insert("reference_clips", row)
    db.insert("reference_clip_features", {"reference_clip_id": rid, "visual_json": vis, "audio_json": aud,
                                          "created_at": time.time()})
    log(f"reference {rid}: {vis.get('cuts_per_10s')} cuts/10s, layout={vis.get('layout', {}).get('kind')}, "
        f"music={aud.get('music', {}).get('kind')}, {aud.get('integrated_lufs')} LUFS")
    return rid


def _per_day(views: Optional[int], publish: Optional[str]) -> Optional[float]:
    if views is None or not publish:
        return None
    try:
        age = (dt.date.today() - dt.date.fromisoformat(publish[:10])).days
    except ValueError:
        return None
    return round(views / max(1, age), 2)


def set_metrics(db: Database, rid: int, **m) -> None:
    """User-entered metrics (e.g. from a platform analytics page for a clip we cannot fetch)."""
    row = db.one("SELECT * FROM reference_clips WHERE id=?", (rid,))
    cur = loads(row["metrics_json"], {})
    cur.update({k: v for k, v in m.items() if v is not None})
    cur["views_per_day"] = _per_day(cur.get("views"), row["publish_date"]) or cur.get("views_per_day")
    db.update("reference_clips", rid, {"metrics_json": cur})


def youtube_search(query: str, limit: int = 25, days: int = 30) -> list[str]:
    """Official YouTube Data API: recent Shorts for a query -> URLs to import. Needs YOUTUBE_API_KEY."""
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise SourceError("YOUTUBE_API_KEY is not set (config/secrets.env). Paste reference URLs instead.")
    after = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get("https://www.googleapis.com/youtube/v3/search", timeout=20, params={
        "part": "snippet", "type": "video", "videoDuration": "short", "order": "viewCount", "q": query,
        "publishedAfter": after, "maxResults": min(50, limit), "key": key})
    if r.status_code != 200:
        raise SourceError(f"YouTube API error {r.status_code}: {r.text[:200]}")
    return [f"https://www.youtube.com/watch?v={i['id']['videoId']}" for i in r.json().get("items", []) if i.get("id", {}).get("videoId")]


def performance_index(rows: list[dict]) -> dict[int, Optional[float]]:
    """views/day relative to the account's size, then expressed as a multiple of the set's median.

    reach = views_per_day / sqrt(max(followers, 1000)): a clip from a small account that still pulled
    views counts for more than the same views from a huge account. Unavailable metrics -> None."""
    import statistics

    reach: dict[int, float] = {}
    for r in rows:
        m = r["metrics"]
        vpd = m.get("views_per_day")
        if vpd is None:
            continue
        reach[r["id"]] = vpd / max(m.get("followers") or 1000, 1000) ** 0.5
    if not reach:
        return {r["id"]: None for r in rows}
    med = statistics.median(reach.values()) or 1e-9
    return {r["id"]: (round(reach[r["id"]] / med, 3) if r["id"] in reach else None) for r in rows}


def all_references(db: Database) -> list[dict]:
    out = []
    for r in db.query("SELECT r.*, f.visual_json, f.audio_json FROM reference_clips r "
                      "LEFT JOIN reference_clip_features f ON f.reference_clip_id=r.id ORDER BY r.id DESC"):
        out.append({"id": r["id"], "url": r["url"], "platform": r["platform"], "account": r["account"],
                    "publish_date": r["publish_date"], "created_at": r["created_at"], "duration": r["duration"],
                    "moment_type": r["moment_type"], "streamer": r["streamer"],
                    "metrics": loads(r["metrics_json"], {}), "visual": loads(r["visual_json"], {}),
                    "audio": loads(r["audio_json"], {})})
    perf = performance_index(out)
    for r in out:
        r["perf"] = perf.get(r["id"])
    return out
