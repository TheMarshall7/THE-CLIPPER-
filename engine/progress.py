"""What the engine is doing right now and how long it will take (for big batches).

Everything is derived from the database: running jobs, finished pipeline steps, and how fast clips have been
finishing so far. No extra bookkeeping in the pipeline.
"""
from __future__ import annotations

import time
from typing import Optional

from engine.db import Database

STEPS = ("audio", "features", "transcribe", "music_scan", "chat", "moments", "clips")
LABELS = {"audio": "Extracting audio", "features": "Analysing sound", "transcribe": "Transcribing speech",
          "music_scan": "Checking for game music", "chat": "Reading chat", "moments": "Picking the best moments",
          "clips": "Rendering clips"}


def fmt_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "estimating…"
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s left" if s > 5 else "almost done"
    if s < 3600:
        return f"~{round(s / 60)} min left"
    return f"~{s // 3600}h {round((s % 3600) / 60):02d}m left"


def _avg_step_seconds(db: Database) -> dict[str, float]:
    out = {}
    for r in db.query("SELECT step, AVG(duration_s) a FROM stream_steps WHERE status='done' AND duration_s IS NOT NULL GROUP BY step"):
        out[r["step"]] = float(r["a"] or 0)
    return out


def _render_rate(db: Database, stream_id: int) -> Optional[float]:
    """Clips finished per second on this stream so far (None until there are two to measure)."""
    rows = db.query("SELECT created_at, updated_at FROM clips WHERE stream_id=? AND status='rendered'", (stream_id,))
    if len(rows) < 2:
        return None
    span = max(r["updated_at"] for r in rows) - min(r["created_at"] for r in rows)
    return len(rows) / span if span > 1 else None


def snapshot(db: Database) -> dict:
    now = time.time()
    running = [dict(r) for r in db.query("SELECT j.*, s.title FROM jobs j LEFT JOIN streams s ON s.id=j.stream_id "
                                         "WHERE j.status='running' ORDER BY j.id")]
    queued = db.one("SELECT COUNT(*) FROM jobs WHERE status='queued'")[0]
    last = db.one("SELECT message FROM logs ORDER BY id DESC LIMIT 1")
    out = {"active": bool(running or queued), "queued": queued, "videos": [], "pct": 0.0, "eta_s": None,
           "headline": "Idle", "detail": last["message"][:140] if last else "", "clips_done": 0, "clips_total": 0}
    if not out["active"]:
        return out
    avg = _avg_step_seconds(db)
    per_video_total = sum(avg.get(s, 0) for s in STEPS) or None
    eta_total = 0.0
    eta_known = True
    done_clips = total_clips = 0
    for j in running:
        sid = j["stream_id"]
        done_steps = {r["step"] for r in db.query("SELECT step FROM stream_steps WHERE stream_id=? AND status='done'", (sid,))}
        step = next((s for s in STEPS if s not in done_steps), "clips")
        clips = db.query("SELECT status FROM clips WHERE stream_id=?", (sid,))
        n_total = len(clips)
        n_done = sum(1 for c in clips if c["status"] in ("rendered", "failed"))
        done_clips += n_done
        total_clips += n_total
        v = {"stream_id": sid, "title": j.get("title") or f"Video {sid}", "step": step, "label": LABELS[step],
             "clips_done": n_done, "clips_total": n_total}
        remaining = 0.0
        if step == "clips" and n_total:
            rate = _render_rate(db, sid)
            if rate:
                remaining = (n_total - n_done) / rate
            else:
                rs = db.one("SELECT AVG(render_seconds) FROM clips WHERE render_seconds IS NOT NULL")[0]
                if rs:
                    remaining = (n_total - n_done) * float(rs)
                else:
                    eta_known = False
        else:
            idx = STEPS.index(step)
            for s in STEPS[idx:]:
                if s in avg:
                    remaining += avg[s] * (0.5 if s == step else 1.0)
                else:
                    eta_known = False
        eta_total += remaining
        out["videos"].append(v)
    if queued and per_video_total:
        eta_total += queued * per_video_total
    elif queued:
        eta_known = False
    out["eta_s"] = eta_total if eta_known else None
    out["clips_done"], out["clips_total"] = done_clips, total_clips
    if running:
        v = out["videos"][0]
        out["headline"] = f"{v['label']} — {v['title'][:48]}"
        if v["step"] == "clips" and v["clips_total"]:
            out["headline"] = f"Rendering clips {done_clips}/{total_clips}"
        # progress: finished steps count equally, rendering counts by clips
        sid = v["stream_id"]
        steps_done = db.one("SELECT COUNT(*) FROM stream_steps WHERE stream_id=? AND status='done' AND step!='clips'", (sid,))[0]
        frac = steps_done / (len(STEPS) - 1) * 0.5
        if v["step"] == "clips" and v["clips_total"]:
            frac = 0.5 + 0.5 * v["clips_done"] / v["clips_total"]
        elif v["step"] == "clips":
            frac = 0.5
        out["pct"] = min(0.99, frac)
    else:
        out["headline"] = f"{queued} video(s) waiting to start"
    out["headline"] += f"   ·   {fmt_eta(out['eta_s'])}" + (f"   ·   +{queued} queued" if running and queued else "")
    return out
