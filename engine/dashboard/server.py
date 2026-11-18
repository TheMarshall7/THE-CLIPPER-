"""Local dashboard: FastAPI backend + one static page (PRD §14).

Runs only on 127.0.0.1. A background worker thread processes the persistent job
queue, so the page just adds jobs and polls for progress, logs and results.

    python -m engine.dashboard.server        (or double-click GTA6Clipper.exe)
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from engine import profiles
from engine.config import PROVIDERS, SECRET_NAMES, load_settings, save_settings, secret_hint, set_secret
from engine.db import PERMISSION_STATUSES, Database, loads

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="GTA 6 Clip Engine")
_settings = load_settings()
_db = Database(_settings.db_path)


# ------------------------------------------------------------------ background worker
class Worker:
    def __init__(self):
        self.busy = False
        self.last_error: Optional[str] = None
        self.started = time.time()
        self._wake = threading.Event()
        threading.Thread(target=self._loop, daemon=True, name="clip-worker").start()

    def wake(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        from engine.pipeline import run_queue

        _db.execute("UPDATE jobs SET status='queued' WHERE status='running'")   # resume after a crash/close
        while True:
            self._wake.wait(timeout=3)
            self._wake.clear()
            if not _db.one("SELECT id FROM jobs WHERE status='queued'"):
                continue
            self.busy = True
            try:
                run_queue(load_settings(), log=lambda m: None)   # engine writes its own log rows
            except Exception as e:
                self.last_error = f"{e}"
                _db.log("error", "worker: " + traceback.format_exc()[-1500:])
            finally:
                self.busy = False


worker = Worker()


# ------------------------------------------------------------------ page
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


# ------------------------------------------------------------------ status
@app.get("/api/overview")
def overview() -> dict:
    from engine.pipeline import Engine

    stats = Engine(load_settings(), log=lambda m: None).stats()
    s = load_settings()
    running = _db.one("SELECT s.id, s.title, s.status, s.duration, st.name FROM streams s LEFT JOIN streamers st "
                      "ON st.id=s.streamer_id WHERE s.status LIKE 'running:%' ORDER BY s.updated_at DESC LIMIT 1")
    return {
        "stats": stats,
        "worker": {"busy": worker.busy or bool(RENDERS), "renders": list(RENDERS.values()), "last_error": worker.last_error,
                   "queued": _db.one("SELECT COUNT(*) FROM jobs WHERE status='queued'")[0]},
        "current": dict(running) if running else None,
        "ai": {"mode": s.ai.mode, "model": s.ai.model if s.ai.mode == "api" else None},
        "whisper": f"{s.transcription.model} on {s.transcription.device}",
    }


@app.get("/api/health")
def health() -> dict:
    """Everything the engine needs, so problems show up on the page instead of as silent failures."""
    import shutil

    from engine.video.webcam import YUNET_PATH

    st = load_settings()
    try:
        import ctranslate2
        gpu = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        gpu = False
    free_gb = round(shutil.disk_usage(str(st.storage.anchor or ".")).free / 1e9, 1)
    items = [
        {"name": "FFmpeg", "ok": bool(shutil.which("ffmpeg")), "detail": "video/audio processing" if shutil.which("ffmpeg") else "missing: install FFmpeg and add it to PATH (winget install Gyan.FFmpeg)"},
        {"name": "Face detection model", "ok": YUNET_PATH.exists(), "detail": "ready" if YUNET_PATH.exists() else "missing: run setup"},
        {"name": "Transcription", "ok": True, "detail": f"Whisper {st.transcription.model} on " + ("GPU" if gpu else "CPU (slower)")},
        {"name": "AI", "ok": True,
         "detail": "rules only (no AI)" if st.ai.mode == "heuristic" else
         (f"{st.ai.provider} / {st.ai.model}" if os.environ.get(SECRET_NAMES[st.ai.provider]) else "API key missing: using rules instead")},
        {"name": "Disk space", "ok": free_gb > 10, "detail": f"{free_gb} GB free"},
    ]
    return {"items": items, "ok": all(i["ok"] for i in items[:2])}


@app.post("/api/shutdown")
def shutdown() -> dict:
    """Quit the engine from the page (the launcher window closes with it)."""
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return {"ok": True}


@app.get("/api/ai-status")
def ai_status() -> dict:
    from engine.ai.llm import get_provider

    s = load_settings()
    prov, msg = get_provider(s.ai)
    return {"mode": s.ai.mode, "active": prov is not None, "message": msg}


def _ai_view() -> dict:
    s = load_settings()
    return {"mode": s.ai.mode, "provider": s.ai.provider, "model": s.ai.model, "openai_base_url": s.ai.openai_base_url,
            "providers": [{"key": k, "label": v["label"], "models": v["models"], "default_model": v["default_model"],
                           "needs_url": v["needs_url"]} for k, v in PROVIDERS.items()],
            "keys": {k: secret_hint(v) for k, v in SECRET_NAMES.items()}}


@app.get("/api/settings/ai")
def get_ai_settings() -> dict:
    return _ai_view()


class AISettingsReq(BaseModel):
    mode: str
    provider: str = "anthropic"
    model: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None          # only sent when the user typed a new one
    clear_key: bool = False


@app.post("/api/settings/ai")
def set_ai_settings(body: AISettingsReq) -> dict:
    if body.mode not in ("heuristic", "api"):
        raise HTTPException(400, "mode must be heuristic or api")
    if body.provider not in PROVIDERS:
        raise HTTPException(400, f"provider must be one of: {', '.join(PROVIDERS)}")
    url = body.openai_base_url.strip().rstrip("/")
    if PROVIDERS[body.provider]["needs_url"] and body.mode == "api":
        if not (url.startswith("https://") or url.startswith(("http://localhost", "http://127.0.0.1"))):
            raise HTTPException(400, "The API address must start with https:// (or be a local http://localhost gateway).")
    if body.mode == "api" and not body.model.strip():
        raise HTTPException(400, "Enter the model name to use.")
    st = load_settings()
    st.ai.mode = body.mode  # type: ignore[assignment]
    st.ai.provider = body.provider  # type: ignore[assignment]
    if body.model.strip():
        st.ai.model = body.model.strip()
    st.ai.openai_base_url = url or st.ai.openai_base_url
    save_settings(st)
    name = SECRET_NAMES[body.provider]
    if body.clear_key:
        set_secret(name, "")
    elif body.api_key and body.api_key.strip():
        key = body.api_key.strip()
        if len(key) < 12 or " " in key:
            raise HTTPException(400, "That does not look like an API key.")
        set_secret(name, key)
    return _ai_view()


@app.post("/api/settings/ai/test")
def test_ai() -> dict:
    """One tiny request with the saved settings. Reports a plain-language result; never returns the key."""
    from engine.ai.llm import AIError, get_provider

    st = load_settings()
    if st.ai.mode != "api":
        return {"ok": True, "message": "AI is off: the engine uses its rules. Nothing to test."}
    prov, msg = get_provider(st.ai)
    if prov is None:
        return {"ok": False, "message": msg}
    try:
        prov.complete_json("Reply with a single JSON object only.", 'Return {"ok": true}')
    except AIError as e:
        return {"ok": False, "message": str(e)}
    return {"ok": True, "message": f"Connected: {prov.name} / {prov.model} replied. ({prov.usage_line()})"}


@app.get("/api/profiles")
def get_profiles() -> list[dict]:
    """The video types the engine can rank, and what each one cares about."""
    return profiles.describe()


class ProfileReq(BaseModel):
    profile: str


@app.post("/api/streams/{stream_id}/profile")
def set_stream_profile(stream_id: int, body: ProfileReq) -> dict:
    """Change a stream's video type and re-rank it (transcript and audio analysis are reused; existing clips stay)."""
    from engine.pipeline import enqueue

    if body.profile not in profiles.PROFILES:
        raise HTTPException(400, "unknown video type")
    r = _db.one("SELECT metadata_json FROM streams WHERE id=?", (stream_id,))
    if not r:
        raise HTTPException(404)
    md = loads(r["metadata_json"], {}) or {}
    md["content_type"], md["content_type_reason"] = body.profile, "chosen by you"
    _db.execute("UPDATE streams SET metadata_json=? WHERE id=?", (json.dumps(md), stream_id))
    _db.execute("UPDATE jobs SET status='failed' WHERE stream_id=? AND status='queued'", (stream_id,))
    jid = enqueue(_db, stream_id, {"force_from": "moments"})
    _db.log("info", f"re-ranking as '{profiles.get(body.profile).label}'", stream_id)
    worker.wake()
    return {"job_id": jid}


@app.get("/api/streams")
def streams() -> list[dict]:
    rows = _db.query("SELECT s.id, s.platform, s.title, s.source_url, s.duration, s.status, s.error, s.viewer_count, "
                     "s.discovery_score, s.updated_at, s.metadata_json, st.name streamer, st.permission_status FROM streams s "
                     "LEFT JOIN streamers st ON st.id=s.streamer_id ORDER BY s.id DESC LIMIT 100")
    out = []
    for r in rows:
        d = dict(r)
        md = loads(d.pop("metadata_json", None), {}) or {}
        ct = profiles.LEGACY.get(md.get("content_type", "auto"), md.get("content_type", "auto"))
        d["profile"], d["profile_reason"] = ct, md.get("content_type_reason")
        d["steps"] = {x["step"]: round(x["duration_s"] or 0, 1) for x in
                      _db.query("SELECT step, duration_s FROM stream_steps WHERE stream_id=? AND status='done'", (r["id"],))}
        d["clips"] = _db.one("SELECT COUNT(*) FROM clips WHERE stream_id=?", (r["id"],))[0]
        out.append(d)
    return out


@app.get("/api/logs")
def logs(after: int = 0, limit: int = 200) -> list[dict]:
    rows = _db.query("SELECT id, ts, level, stream_id, message FROM logs WHERE id>? ORDER BY id DESC LIMIT ?", (after, limit))
    return [dict(r) for r in reversed(rows)]


# ------------------------------------------------------------------ jobs
class ProcessReq(BaseModel):
    source: str                      # one link/path, or several separated by new lines
    streamer: Optional[str] = None
    permission: Optional[str] = None
    focus: Optional[str] = None
    max_clips: Optional[int] = None
    content_type: str = "auto"   # auto or a profile key (see /api/profiles)
    song: Optional[str] = None   # song slug for every clip of this job, "__none__" for no song, empty = automatic


def _queue_source(src: str, streamer, permission, focus, max_clips, content_type: str = "auto", song: Optional[str] = None) -> dict:
    from engine.pipeline import Engine, enqueue

    eng = Engine(load_settings(), log=lambda m: None)
    sid = eng.add_source(src, streamer or None, permission or None,
                         content_type=content_type if content_type == "auto" or content_type in profiles.PROFILES else "auto")
    jid = enqueue(_db, sid, {"focus": focus or None, "max_clips": max_clips, "song": song or None})
    _db.log("info", f"queued: {src}", sid)
    return {"source": src, "stream_id": sid, "job_id": jid}


@app.post("/api/process")
def process(req: ProcessReq) -> dict:
    if req.permission and req.permission not in PERMISSION_STATUSES:
        raise HTTPException(400, "bad permission")
    sources = [s.strip().strip('"') for s in req.source.replace("\r", "").split("\n") if s.strip()]
    if not sources:
        raise HTTPException(400, "Paste at least one link or file path.")
    queued, errors = [], []
    for src in sources:   # each link is checked on its own; one bad link doesn't stop the rest
        try:
            queued.append(_queue_source(src, req.streamer, req.permission, req.focus, req.max_clips, req.content_type, req.song))
        except Exception as e:
            errors.append({"source": src, "error": str(e)})
    if queued:
        worker.wake()
    if not queued:
        raise HTTPException(400, "; ".join(f"{e['source']}: {e['error']}" for e in errors))
    return {"queued": queued, "errors": errors}


UPLOAD_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".flv", ".ts", ".m4v", ".avi"}
SONG_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


async def _save_upload(file: UploadFile, dest_dir: Path, allowed: set[str]) -> Path:
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type {ext or '(none)'}; allowed: {', '.join(sorted(allowed))}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in Path(name).stem).strip() or "upload"
    dest = dest_dir / f"{stem}{ext}"
    k = 1
    while dest.exists():
        dest = dest_dir / f"{stem}_{k}{ext}"
        k += 1
    with open(dest, "wb") as out:          # stream to disk in 8 MB chunks (big VODs)
        while chunk := await file.read(8 << 20):
            out.write(chunk)
    return dest


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...), streamer: str = Form(""), permission: str = Form(""),
                       focus: str = Form(""), max_clips: str = Form(""), content_type: str = Form("auto")) -> dict:
    if permission and permission not in PERMISSION_STATUSES:
        raise HTTPException(400, "bad permission")
    dest = await _save_upload(file, load_settings().storage / "uploads", UPLOAD_EXTS)
    try:
        r = _queue_source(str(dest), streamer, permission, focus, int(max_clips) if max_clips.strip().isdigit() else None, content_type)
    except Exception as e:
        raise HTTPException(400, str(e))
    worker.wake()
    return {"queued": [r], "errors": [], "saved_to": str(dest)}


RENDERS: dict[int, str] = {}     # clip/candidate renders started from the page: id -> label


def _bg_render(label: str, fn) -> None:
    key = int(time.time() * 1000)
    RENDERS[key] = label

    def run():
        try:
            fn()
        except Exception as e:
            _db.log("error", f"{label} failed: {e}")
        finally:
            RENDERS.pop(key, None)
    threading.Thread(target=run, daemon=True).start()


@app.get("/api/streams/{stream_id}/candidates")
def stream_candidates(stream_id: int) -> list[dict]:
    out = []
    for r in _db.query("SELECT c.*, (SELECT id FROM clips k WHERE k.candidate_id=c.id) clip_id FROM candidates c "
                       "WHERE stream_id=? ORDER BY total_score DESC", (stream_id,)):
        out.append({"id": r["id"], "start": r["start"], "end": r["end"], "score": r["total_score"], "status": r["status"],
                    "reason": r["reject_reason"], "category": r["category"], "summary": r["summary"],
                    "why": (loads(r["ai_json"], {}) or {}).get("why"), "clip_id": r["clip_id"],
                    "subscores": loads(r["scores_json"], {})})
    return out


@app.post("/api/candidates/{cid}/render")
def render_candidate(cid: int) -> dict:
    from engine.pipeline import Engine

    c = _db.one("SELECT stream_id FROM candidates WHERE id=?", (cid,))
    if not c:
        raise HTTPException(404)
    _db.log("info", f"rendering moment {cid} on request (overriding the automatic rejection)", c["stream_id"])
    _bg_render(f"moment {cid}", lambda: Engine(load_settings(), log=lambda m: None).create_clips(c["stream_id"], [cid], 1))
    return {"ok": True}


class ClipMusicReq(BaseModel):
    song: Optional[str] = None       # slug, or None for automatic
    mode: Optional[str] = None
    off: bool = False
    drop_at: Optional[float] = None  # seconds into the CLIP where the hook should land
    enter_at: Optional[float] = None  # seconds into the CLIP where the song comes in
    ignore_detected: bool = False     # user listened: no music in this clip
    volume: Optional[str] = None      # subtle | balanced | featured | trailer
    dropout: Optional[bool] = None    # silent beat before the chorus: None = follow Settings
    lyrics: Optional[bool] = None     # song lyrics on screen: None = follow Settings
    ramps: Optional[bool] = None      # speed ramps around the hit: None = follow Settings
    fx: Optional[bool] = None         # beat-reactive picture: None = follow Settings


@app.post("/api/clips/{clip_id}/music")
def clip_music(clip_id: int, req: ClipMusicReq) -> dict:
    from engine.pipeline import Engine

    c = _db.one("SELECT layout_json, stream_id, start FROM clips WHERE id=?", (clip_id,))
    if not c:
        raise HTTPException(404)
    if req.mode and req.mode not in ("payoff_drop", "reaction_tail", "underscore", "cold_open"):
        raise HTTPException(400, "bad mode")
    lay = loads(c["layout_json"], {}) or {}
    lay["music"] = {"off": True} if req.off else {k: v for k, v in {"song": req.song, "mode": req.mode}.items() if v}
    if req.drop_at is not None and not req.off:
        # the page shows clip time; the engine needs source time. Map through the jump-cut plan.
        keep = lay.get("_keep") or []
        acc, src_t = 0.0, None
        for a, b in keep:
            if req.drop_at <= acc + (b - a):
                src_t = c["start"] + a + (req.drop_at - acc)
                break
            acc += b - a
        lay["music"].update({"drop_at": src_t if src_t is not None else c["start"] + req.drop_at, "mode": "payoff_drop"})
    def to_src(t: float) -> float:
        keep = lay.get("_keep") or []
        acc = 0.0
        for a, b in keep:
            if t <= acc + (b - a):
                return c["start"] + a + (t - acc)
            acc += b - a
        return c["start"] + t
    if req.enter_at is not None and not req.off:
        lay["music"]["enter_at"] = to_src(req.enter_at)
    if req.ignore_detected and not req.off:
        lay["music"]["ignore_detected"] = True
    if req.volume in ("subtle", "balanced", "featured", "trailer") and not req.off:
        lay["music"]["volume"] = req.volume
    for k in ("dropout", "lyrics", "ramps", "fx"):
        if getattr(req, k) is not None and not req.off:
            lay["music"][k] = getattr(req, k)
    _db.update("clips", clip_id, {"layout_json": lay})
    _db.log("info", f"re-rendering clip {clip_id} with music: {'off' if req.off else (req.song or 'auto')} {req.mode or ''}", c["stream_id"])
    _bg_render(f"clip {clip_id}", lambda: Engine(load_settings(), log=lambda m: None).render_clip(clip_id))
    return {"ok": True}


@app.post("/api/streams/{stream_id}/retry")
def retry(stream_id: int) -> dict:
    from engine.pipeline import enqueue

    _db.execute("UPDATE jobs SET status='failed' WHERE stream_id=? AND status='queued'", (stream_id,))
    jid = enqueue(_db, stream_id, {})
    worker.wake()
    return {"job_id": jid}


# ------------------------------------------------------------------ clips
@app.get("/api/clips")
def clips(min_score: float = 0, approval: Optional[str] = None, sort: str = "score", limit: int = 200) -> list[dict]:
    sql = ("SELECT c.*, s.platform, s.title stream_title, s.source_url, st.name streamer, k.category, k.summary, "
           "k.scores_json, k.ai_json FROM clips c JOIN streams s ON s.id=c.stream_id "
           "LEFT JOIN streamers st ON st.id=s.streamer_id LEFT JOIN candidates k ON k.id=c.candidate_id "
           "WHERE COALESCE(c.score,0)>=?")
    params: list = [min_score]
    if approval:
        sql += " AND c.approval=?"
        params.append(approval)
    # "new": the most recently made clips first (so a fresh batch is never hidden behind older, higher-scored clips)
    sql += (" ORDER BY c.created_at DESC, c.id DESC" if sort == "new" else " ORDER BY c.score DESC") + f" LIMIT {int(limit)}"
    out = []
    for r in _db.query(sql, params):
        qc = loads(r["qc_json"], {}) or {}
        meta = loads(r["metadata_json"], {}) or {}
        hooks = loads(r["hooks_json"], []) or []
        out.append({
            "id": r["id"], "stream_id": r["stream_id"], "streamer": r["streamer"], "platform": r["platform"],
            "start": r["start"], "end": r["end"], "duration": (loads(r["captions_json"], {}) or {}).get("duration") or (r["end"] - r["start"]),
            "score": r["score"], "status": r["status"], "approval": r["approval"], "publish_ready": bool(r["publish_ready"]),
            "block_reason": r["publish_block_reason"], "error": r["error"], "category": r["category"], "summary": r["summary"],
            "hooks": hooks, "hook_choice": r["hook_choice"], "title": meta.get("title"), "caption": meta.get("caption"),
            "subscores": loads(r["scores_json"], {}), "ai_mode": (loads(r["ai_json"], {}) or {}).get("mode"),
            "why": (loads(r["ai_json"], {}) or {}).get("why"),
            "music": qc.get("music", {}), "loudness": qc.get("checks", {}).get("integrated_lufs"),
            "cut_s": qc.get("checks", {}).get("jump_cuts_removed_s"), "has_video": bool(r["video_path"]),
            "updated_at": r["updated_at"],
        })
    return out


@app.get("/media/clip/{clip_id}")
def clip_video(clip_id: int, clean: bool = False):
    r = _db.one("SELECT video_path FROM clips WHERE id=?", (clip_id,))
    if not r or not r["video_path"]:
        raise HTTPException(404)
    p = Path(r["video_path"])
    if clean:
        p = p.with_name("clip_clean.mp4")
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4")


@app.get("/media/thumb/{clip_id}")
def clip_thumb(clip_id: int):
    r = _db.one("SELECT thumb_path FROM clips WHERE id=?", (clip_id,))
    if not r or not r["thumb_path"] or not Path(r["thumb_path"]).exists():
        raise HTTPException(404)
    return FileResponse(r["thumb_path"], media_type="image/jpeg")


class Approval(BaseModel):
    approval: str


@app.post("/api/clips/{clip_id}/approval")
def set_approval(clip_id: int, body: Approval) -> dict:
    if body.approval not in ("approved", "rejected", "pending"):
        raise HTTPException(400)
    _db.execute("UPDATE clips SET approval=?, updated_at=? WHERE id=?", (body.approval, time.time(), clip_id))
    return {"ok": True}


class ExportReq(BaseModel):
    clip_ids: Optional[list[int]] = None
    include_unverified: bool = False


@app.post("/api/export")
def export(req: ExportReq) -> dict:
    from engine.pipeline import Engine

    msgs: list[str] = []
    eng = Engine(load_settings(), log=msgs.append)
    ids = req.clip_ids or [r["id"] for r in _db.query("SELECT id FROM clips WHERE approval='approved' AND status='rendered'")]
    paths = eng.export(ids, req.include_unverified)
    return {"exported": [str(p) for p in paths], "messages": msgs, "folder": str(eng.s.exports)}


@app.post("/api/open-exports")
def open_exports() -> dict:
    p = load_settings().exports
    if hasattr(os, "startfile"):
        os.startfile(str(p))
    return {"folder": str(p)}


# ------------------------------------------------------------------ songs / streamers
@app.get("/api/songs")
def songs() -> list[dict]:
    out = []
    for r in _db.query("SELECT s.*, a.name artist, (SELECT COUNT(*) FROM song_usage u WHERE u.song_id=s.id) uses "
                       "FROM songs s LEFT JOIN artists a ON a.id=s.artist_id"):
        a = loads(r["analysis_json"], {}) or {}
        out.append({"id": r["id"], "slug": r["slug"], "title": r["title"], "artist": r["artist"], "bpm": a.get("bpm"), "key": a.get("key"),
                    "explicit": bool(r["explicit"]), "error": a.get("error"),
                    "official_sounds": loads(r["official_sounds_json"], {}) or {},
                    "lyrics": len(a.get("lyrics", [])),
                    "mood": a.get("mood", {}).get("tags", []), "priority": r["campaign_priority"], "uses": r["uses"],
                    "sections": [{"label": x["label"], "start": x["start"], "end": x["end"], "hook": x.get("is_hook")}
                                 for x in a.get("sections", [])]})
    return out


IMPORTS: dict[str, dict] = {}   # song uploads being analyzed in the background


@app.post("/api/songs/upload")
async def upload_song(file: UploadFile = File(...), title: str = Form(...), artist: str = Form(...),
                      confirm_rights: str = Form(""), explicit: str = Form(""), priority: int = Form(3),
                      tiktok_sound: str = Form(""), instagram_sound: str = Form(""), youtube_sound: str = Form("")) -> dict:
    if confirm_rights not in ("true", "on", "1"):
        raise HTTPException(400, "Confirm you own or control the rights to this song.")
    if not title.strip() or not artist.strip():
        raise HTTPException(400, "Title and artist are required.")
    s = load_settings()
    tmp = await _save_upload(file, s.storage / "uploads" / "songs", SONG_EXTS)
    key = f"{time.time():.3f}"
    IMPORTS[key] = {"title": title, "artist": artist, "state": "analyzing", "message": "Analyzing beat, sections, hook, mood…",
                    "started": time.time()}
    sounds = {k: v.strip() for k, v in {"tiktok": tiktok_sound, "instagram": instagram_sound,
                                        "youtube_shorts": youtube_sound}.items() if v.strip()}

    def run():
        from engine.music.library import import_song
        try:
            sid = import_song(Database(s.db_path), s.library / "songs", tmp, title.strip(), artist.strip(),
                              "Uploader confirmed ownership/control of rights in the dashboard", "dashboard",
                              explicit in ("true", "on", "1"), max(1, min(5, int(priority))), sounds, s.transcription,
                              log=lambda m: _db.log("info", f"[music] {m}"))
            IMPORTS[key].update(state="done", message="Ready to use", song_id=sid)
            _db.log("info", f"[music] imported '{title}' by {artist}")
        except Exception as e:
            IMPORTS[key].update(state="error", message=str(e)[:300])
            _db.log("error", f"[music] import failed for '{title}': {e}")
        finally:
            tmp.unlink(missing_ok=True)    # the library keeps its own copy

    threading.Thread(target=run, daemon=True).start()
    return {"import_id": key}


@app.get("/api/songs/imports")
def song_imports() -> list[dict]:
    now = time.time()
    return [{"id": k, **v} for k, v in IMPORTS.items() if v["state"] == "analyzing" or now - v["started"] < 120]


@app.post("/api/songs/{song_id}/delete")
def delete_song(song_id: int) -> dict:
    r = _db.one("SELECT file_path FROM songs WHERE id=?", (song_id,))
    if not r:
        raise HTTPException(404)
    _db.execute("UPDATE music_placements SET song_id=NULL WHERE song_id=?", (song_id,))
    _db.execute("DELETE FROM song_usage WHERE song_id=?", (song_id,))
    _db.execute("DELETE FROM songs WHERE id=?", (song_id,))
    p = Path(r["file_path"])
    if p.exists() and load_settings().library in p.parents:
        p.unlink()
    return {"ok": True}


class SongPatch(BaseModel):
    priority: Optional[int] = None
    explicit: Optional[bool] = None
    title: Optional[str] = None
    artist: Optional[str] = None


@app.post("/api/songs/{song_id}")
def patch_song(song_id: int, body: SongPatch) -> dict:
    if not _db.one("SELECT id FROM songs WHERE id=?", (song_id,)):
        raise HTTPException(404)
    if body.title is not None:
        if not body.title.strip():
            raise HTTPException(400, "Title can't be empty.")
        # the slug stays the same so clips and placements that reference it keep working
        _db.execute("UPDATE songs SET title=? WHERE id=?", (body.title.strip(), song_id))
    if body.artist is not None:
        name = body.artist.strip()
        if not name:
            raise HTTPException(400, "Artist can't be empty.")
        art = _db.one("SELECT id FROM artists WHERE name=?", (name,))
        aid = art["id"] if art else _db.insert("artists", {"name": name, "created_at": time.time()})
        _db.execute("UPDATE songs SET artist_id=? WHERE id=?", (aid, song_id))
    if body.priority is not None:
        _db.execute("UPDATE songs SET campaign_priority=? WHERE id=?", (max(1, min(5, body.priority)), song_id))
    if body.explicit is not None:
        _db.execute("UPDATE songs SET explicit=? WHERE id=?", (int(body.explicit), song_id))
    return {"ok": True}


@app.get("/media/song/{song_id}")
def song_audio(song_id: int):
    r = _db.one("SELECT file_path FROM songs WHERE id=?", (song_id,))
    if not r or not Path(r["file_path"]).exists():
        raise HTTPException(404)
    return FileResponse(r["file_path"])


@app.get("/api/artists")
def artists() -> list[dict]:
    out = []
    for r in _db.query("SELECT * FROM artists ORDER BY name"):
        out.append({"id": r["id"], "name": r["name"], "description": r["brand_description"] or "",
                    "good_fit": loads(r["good_fit_moments_json"], []) or [], "avoid": loads(r["avoid_moments_json"], []) or [],
                    "cta": loads(r["cta_options_json"], []) or [], "sound_tag": r["sound_tag_format"] or "",
                    "volume": r["default_volume"] or "balanced"})
    return out


class ArtistReq(BaseModel):
    name: str
    description: str = ""
    good_fit: list[str] = []
    avoid: list[str] = []
    cta: list[str] = []
    sound_tag: str = ""
    volume: str = "balanced"


@app.post("/api/artists")
def save_artist(a: ArtistReq) -> dict:
    if not a.name.strip():
        raise HTTPException(400, "Artist name required")
    row = _db.one("SELECT id FROM artists WHERE name=?", (a.name.strip(),))
    aid = row["id"] if row else _db.insert("artists", {"name": a.name.strip(), "created_at": time.time()})
    _db.update("artists", aid, {"brand_description": a.description, "good_fit_moments_json": [x for x in a.good_fit if x],
                                "avoid_moments_json": [x for x in a.avoid if x], "cta_options_json": [x for x in a.cta if x],
                                "sound_tag_format": a.sound_tag or None,
                                "default_volume": a.volume if a.volume in ("subtle", "balanced", "featured") else "balanced"})
    return {"id": aid}


@app.get("/api/streamers")
def streamers() -> list[dict]:
    return [dict(r) for r in _db.query("SELECT id, name, platform, channel_url, permission_status, allows_added_music, "
                                       "favorite, excluded FROM streamers ORDER BY name")]


class StreamerReq(BaseModel):
    name: str
    platform: str
    channel_url: Optional[str] = None
    permission_status: str = "unknown"
    allows_added_music: Optional[bool] = None


@app.post("/api/streamers")
def add_streamer(body: StreamerReq) -> dict:
    if body.platform not in ("twitch", "youtube", "kick", "file"):
        raise HTTPException(400, "platform must be twitch, youtube, kick or file")
    if body.permission_status not in PERMISSION_STATUSES:
        raise HTTPException(400, "bad permission")
    row = _db.one("SELECT id FROM streamers WHERE platform=? AND name=?", (body.platform, body.name.strip()))
    fields = {"permission_status": body.permission_status, "channel_url": body.channel_url or None, "watchlist": 1,
              "allows_added_music": None if body.allows_added_music is None else int(body.allows_added_music)}
    if row:
        _db.update("streamers", row["id"], fields)
        return {"id": row["id"]}
    return {"id": _db.insert("streamers", {"name": body.name.strip(), "platform": body.platform, "created_at": time.time(), **fields})}


class MusicFlag(BaseModel):
    allows_added_music: Optional[bool] = None


@app.post("/api/streamers/{sid}/music")
def set_streamer_music(sid: int, body: MusicFlag) -> dict:
    _db.execute("UPDATE streamers SET allows_added_music=? WHERE id=?",
                (None if body.allows_added_music is None else int(body.allows_added_music), sid))
    return {"ok": True}


class PermReq(BaseModel):
    permission_status: str


@app.post("/api/streamers/{sid}/permission")
def set_permission(sid: int, body: PermReq) -> dict:
    if body.permission_status not in PERMISSION_STATUSES:
        raise HTTPException(400)
    _db.execute("UPDATE streamers SET permission_status=? WHERE id=?", (body.permission_status, sid))
    return {"ok": True}


from engine.dashboard.extras import router as _extras_router  # noqa: E402

app.include_router(_extras_router)


def main(port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn
    import webbrowser

    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    import sys

    port = 8765
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    main(port=port, open_browser="--no-browser" not in sys.argv)
