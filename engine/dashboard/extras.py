"""Dashboard additions: discovery, clip editor, full settings, trends and recipes (PRD §14 B/D/E, M15)."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from engine import editing
from engine.config import load_settings, save_settings
from engine.db import Database, loads

router = APIRouter()
STATIC = Path(__file__).parent / "static"


def _db() -> Database:
    return Database(load_settings().db_path)


def _render_bg(label: str, clip_id: int) -> None:
    from engine.dashboard import server
    from engine.pipeline import Engine

    server._bg_render(label, lambda: Engine(load_settings(), log=lambda m: None).render_clip(clip_id))


@router.get("/extras.js")
def extras_js():
    return FileResponse(STATIC / "extras.js", media_type="application/javascript", headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------ B. stream discovery
@router.get("/api/discover")
def discover(query: str = "GTA 6", live: bool = True, platforms: str = "twitch,youtube", limit: int = 20) -> dict:
    from engine.discovery.platforms import discover as run_discover

    s = load_settings()
    db = _db()
    perms = {f"{r['platform']}:{r['name'].lower()}": r["permission_status"] for r in db.query("SELECT * FROM streamers")}
    excluded = {f"{r['platform']}:{r['name'].lower()}" for r in db.query("SELECT * FROM streamers WHERE excluded=1")}
    plats = [p for p in platforms.split(",") if p]
    keys = {"twitch": bool(os.environ.get("TWITCH_CLIENT_ID") and os.environ.get("TWITCH_CLIENT_SECRET")),
            "youtube": bool(os.environ.get("YOUTUBE_API_KEY"))}
    found, notes = run_discover(query, s.vertical_keywords, live, limit, plats, perms)
    found = [f for f in found if f"{f.platform}:{f.streamer.lower()}" not in excluded]
    done = {r["source_url"]: r["status"] for r in db.query("SELECT source_url, status FROM streams")}
    return {"results": [{"streamer": f.streamer, "platform": f.platform, "title": f.title, "url": f.url,
                         "viewers": f.viewer_count, "views": f.total_views, "duration": f.duration_s, "live": f.is_live,
                         "score": f.discovery_score, "recording": f.recording_available,
                         "permission": perms.get(f"{f.platform}:{f.streamer.lower()}", "unknown"),
                         "status": done.get(f.url)} for f in found],
            "notes": notes, "keys": keys,
            "kick": "Kick has no public discovery API configured: add Kick channels under Creators and paste their VOD links."}


# ------------------------------------------------------------------ D. clip editor
@router.get("/api/clips/{clip_id}/detail")
def clip_detail(clip_id: int) -> dict:
    db = _db()
    c = db.one("SELECT c.*, s.duration stream_duration, s.platform, s.source_url FROM clips c JOIN streams s ON s.id=c.stream_id "
               "WHERE c.id=?", (clip_id,))
    if not c:
        raise HTTPException(404)
    meta = loads(c["metadata_json"], {}) or {}
    lay = loads(c["layout_json"], {}) or {}
    return {"id": c["id"], "start": c["start"], "end": c["end"], "stream_duration": c["stream_duration"],
            "hooks": meta.get("hooks") or loads(c["hooks_json"], []) or [], "hook_choice": c["hook_choice"],
            "title": meta.get("title"), "description": meta.get("description"),
            "layout": lay.get("layout"), "cam_box": lay.get("cam_box"), "frame_size": lay.get("frame_size"),
            "focus_x": lay.get("gameplay_focus_x", 0.5), "gameplay_mode": lay.get("gameplay_mode"),
            "caption": loads(c["caption_style_json"], {}) or {}, "recipe": (lay.get("recipe") or {}).get("recipe_id"),
            "recipe_off": bool((lay.get("recipe") or {}).get("off"))}


class EditReq(BaseModel):
    start: Optional[float] = None
    end: Optional[float] = None
    hook_choice: Optional[int] = None
    hook_text: Optional[str] = None
    layout: Optional[str] = None
    cam_box: Optional[list[int]] = None
    focus_x: Optional[float] = None
    gameplay_mode: Optional[str] = None
    caption: Optional[dict] = None
    title: Optional[str] = None
    description: Optional[str] = None
    recipe: Optional[str] = None


@router.post("/api/clips/{clip_id}/edit")
def clip_edit(clip_id: int, req: EditReq) -> dict:
    try:
        r = editing.edit_clip(_db(), clip_id, **req.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    if r["needs_render"]:
        _render_bg(f"clip {clip_id} (edit)", clip_id)
    return r


# ------------------------------------------------------------------ E. settings
_EDITABLE: dict[str, tuple[str, type, Any]] = {
    # key -> (dotted path, type, (min, max) or choices)
    "clip_min": ("clips.min_duration", float, (5, 120)),
    "clip_max": ("clips.max_duration", float, (10, 180)),
    "sensitivity": ("clips.sensitivity", float, (0, 1)),
    "quality_threshold": ("clips.quality_threshold", float, (0, 100)),
    "max_clips_per_stream": ("clips.max_clips_per_stream", int, (1, 60)),
    "whisper_model": ("transcription.model", str, ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "distil-large-v3"]),
    "whisper_device": ("transcription.device", str, ["auto", "cuda", "cpu"]),
    "width": ("video.width", int, [720, 1080]),
    "height": ("video.height", int, [1280, 1920]),
    "encoder": ("video.encoder", str, ["auto", "libx264", "h264_nvenc"]),
    "gameplay_mode": ("video.gameplay_mode", str, ["fill", "fit_blur"]),
    "cap_font": ("captions.font", str, None),
    "cap_size": ("captions.font_size", int, (30, 140)),
    "cap_highlight": ("captions.highlight_color", str, None),
    "cap_words": ("captions.max_words_per_line", int, (1, 6)),
    "cap_position": ("captions.position", str, ["seam", "lower", "upper"]),
    "cap_animation": ("captions.animation", str, ["highlight", "pop", "none"]),
    "storage_dir": ("storage_dir", str, None),
    "max_concurrent_jobs": ("max_concurrent_jobs", int, (1, 4)),
    "priority": ("processing_priority", str, ["low", "normal"]),
    "end_card": ("music.end_card", bool, None),
    "end_card_seconds": ("music.end_card_seconds", float, (1.5, 3.0)),
    "remove_source_music": ("music.remove_source_music", bool, None),
    "ducking": ("music.ducking", str, ["smooth", "tight"]),
    "duck_hold_s": ("music.duck_hold_s", float, (0.5, 6.0)),
    "beat_sync_cuts": ("music.beat_sync_cuts", bool, None),
    "pre_drop_dropout": ("music.pre_drop_dropout", str, ["auto", "on", "off"]),
    "pre_drop_seconds": ("music.pre_drop_seconds", float, (0.0, 0.8)),
    "lyrics": ("captions.lyrics", str, ["auto", "on", "off"]),
    "promo_build_bars": ("music.promo_build_bars", int, (1, 8)),
    "speed_ramps": ("music.speed_ramps", str, ["auto", "on", "off"]),
    "beat_fx": ("music.beat_fx", str, ["auto", "on", "off"]),
    "beat_fx_strength": ("music.beat_fx_strength", float, (0.3, 2.0)),
}


def _get(s, path: str):
    for p in path.split("."):
        s = getattr(s, p)
    return s


@router.get("/api/settings/engine")
def engine_settings() -> dict:
    from engine.music import separate

    s = load_settings()
    return {"values": {k: _get(s, v[0]) for k, v in _EDITABLE.items()},
            "choices": {k: v[2] for k, v in _EDITABLE.items() if isinstance(v[2], list)},
            "ranges": {k: v[2] for k, v in _EDITABLE.items() if isinstance(v[2], tuple)},
            "demucs": separate.available()}


@router.post("/api/settings/engine")
def save_engine_settings(body: dict) -> dict:
    s = load_settings()
    changed = []
    for k, val in body.items():
        if k not in _EDITABLE:
            raise HTTPException(400, f"unknown setting {k}")
        path, typ, rule = _EDITABLE[k]
        try:
            val = typ(val) if typ is not bool else bool(val)
        except (TypeError, ValueError):
            raise HTTPException(400, f"{k}: expected {typ.__name__}")
        if isinstance(rule, list) and val not in rule:
            raise HTTPException(400, f"{k}: must be one of {rule}")
        if isinstance(rule, tuple) and not (rule[0] <= val <= rule[1]):
            raise HTTPException(400, f"{k}: must be between {rule[0]} and {rule[1]}")
        if k == "storage_dir" and not str(val).strip():
            raise HTTPException(400, "storage location cannot be empty")
        obj, last = s, path.split(".")
        for p in last[:-1]:
            obj = getattr(obj, p)
        setattr(obj, last[-1], val)
        changed.append(k)
    if s.clips.min_duration >= s.clips.max_duration:
        raise HTTPException(400, "minimum clip length must be below the maximum")
    save_settings(s)
    return {"ok": True, "changed": changed}


# ------------------------------------------------------------------ M15 trends + recipes
@router.get("/api/trends/references")
def trend_refs() -> list[dict]:
    from engine.trends import store

    return store.all_references(_db())


class ImportReq(BaseModel):
    urls: list[str]
    streamer: Optional[str] = None
    moment: Optional[str] = None


@router.post("/api/trends/import")
def trend_import(req: ImportReq) -> dict:
    from engine.dashboard import server
    from engine.trends import report, store

    s = load_settings()
    results: list[dict] = []

    def run():
        db = _db()
        for u in [x.strip() for x in req.urls if x.strip()]:
            try:
                results.append({"url": u, "id": store.import_reference(db, s, u, req.streamer, req.moment, lambda m: None)})
            except Exception as e:
                db.log("error", f"reference import failed for {u}: {e}")
        report.refresh(db, lambda m: None)

    server._bg_render(f"importing {len(req.urls)} reference clip(s)", run)
    return {"ok": True}


class MetricsReq(BaseModel):
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    saves: Optional[int] = None
    followers: Optional[int] = None


@router.post("/api/trends/references/{rid}/metrics")
def trend_metrics(rid: int, req: MetricsReq) -> dict:
    from engine.trends import report, store

    db = _db()
    store.set_metrics(db, rid, **req.model_dump())
    report.refresh(db, lambda m: None)
    return {"ok": True}


@router.post("/api/trends/references/{rid}/delete")
def trend_delete(rid: int) -> dict:
    _db().execute("DELETE FROM reference_clips WHERE id=?", (rid,))
    return {"ok": True}


@router.get("/api/trends/report")
def trend_report(window: str = "7d", refresh: bool = False) -> dict:
    from engine.trends import report

    if window not in report.WINDOWS:
        raise HTTPException(400, "window must be 24h, 7d or 30d")
    db = _db()
    rep = report.refresh(db, lambda m: None)[window] if refresh or not report.latest(db, window) else report.latest(db, window)
    return rep


@router.get("/api/recipes")
def recipes_list() -> list[dict]:
    from engine.trends import recipes

    return recipes.list_all(_db())


class RecipeReq(BaseModel):
    moment: str = "any"
    window: str = "30d"


@router.post("/api/recipes/create")
def recipes_create(req: RecipeReq) -> dict:
    from engine.trends import recipes, report, store

    db = _db()
    rep = report.latest(db, req.window) or report.refresh(db, lambda m: None)[req.window]
    refs = store.all_references(db)
    if not refs:
        raise HTTPException(400, "Import some reference clips first (Trends tab).")
    art = db.one("SELECT * FROM artists ORDER BY id LIMIT 1")
    rec = recipes.from_report(rep, refs, req.moment, n=len(recipes.list_all(db)) + 1)
    tw = recipes.twists(rec, rep, {"visual_style": loads(art["visual_style_json"], {})} if art else None)
    return {"recipe_id": recipes.save(db, rec, tw)}


@router.post("/api/recipes/{recipe_id}/{action}")
def recipes_action(recipe_id: str, action: str) -> dict:
    from engine.trends import recipes

    status = {"promote": "default", "retire": "retired", "restore": "candidate"}.get(action)
    if not status:
        raise HTTPException(400)
    recipes.set_status(_db(), recipe_id, status)
    return {"ok": True}


class PerfReq(BaseModel):
    platform: str
    views: int
    likes: Optional[int] = None
    shares: Optional[int] = None
    saves: Optional[int] = None
    comments: Optional[int] = None


@router.post("/api/clips/{clip_id}/performance")
def clip_perf(clip_id: int, req: PerfReq) -> dict:
    from engine.trends import recipes

    recipes.record_performance(_db(), clip_id, req.platform, req.views, req.likes, req.shares, req.saves, req.comments)
    return {"ok": True}
