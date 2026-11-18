"""`clipper` command-line interface (PRD §18).

Natural-language requests in Claude Code map onto these commands; the CLI is the
reliable, repeatable layer underneath.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from engine.config import CONFIG_PATH, PROVIDERS, ROOT, SECRET_NAMES, SECRETS_PATH, connect_provider, connect_with_key, load_settings, save_settings
from engine.db import PERMISSION_STATUSES, Database, loads

app = typer.Typer(add_completion=False, help="GTA 6 Viral Clip Intelligence Engine", no_args_is_help=True)
streamer_app = typer.Typer(help="Manage streamers, watchlist and permissions", no_args_is_help=True)
clips_app = typer.Typer(help="Review, edit and approve clips", no_args_is_help=True)
queue_app = typer.Typer(help="Persistent job queue", no_args_is_help=True)
discover_app = typer.Typer(help="Find GTA 6 streams via official APIs", no_args_is_help=True)
ai_app = typer.Typer(help="Choose the AI that judges clips (any provider, or rules only)", no_args_is_help=True)
app.add_typer(streamer_app, name="streamer")
app.add_typer(clips_app, name="clips")
app.add_typer(queue_app, name="queue")
app.add_typer(discover_app, name="discover")
app.add_typer(ai_app, name="ai")
console = Console()


def _engine():
    from engine.pipeline import Engine

    return Engine(load_settings(), log=lambda m: console.print(f"[dim]{time.strftime('%H:%M:%S')}[/] {m}"))


def _db() -> Database:
    return Database(load_settings().db_path)


def _fmt_t(t: Optional[float]) -> str:
    if t is None:
        return "-"
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ------------------------------------------------------------------ AI
@ai_app.command("key")
def ai_key(key: Optional[str] = typer.Argument(None, help="The API key. Left out, you are prompted (input is hidden)."),
           provider: Optional[str] = typer.Option(None, help=" | ".join(PROVIDERS) + ". Only needed if the key prefix is not recognised.")):
    """Paste any supported API key: the provider is detected from it, AI is switched on, and the rules stay as fallback."""
    if not key:
        key = typer.prompt("Paste API key", hide_input=True)
    try:
        name = connect_provider(provider, key) if provider else connect_with_key(key)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    spec = PROVIDERS[name]
    console.print(f"[green]Saved.[/] AI judging is on: {spec['label']}, model [bold]{spec['default_model']}[/].")
    console.print("Change the model in the dashboard Settings tab. Test it with `clipper ai test`.")


@ai_app.command("off")
def ai_off():
    """Switch AI judging off. The engine goes back to rules only. Keys are kept."""
    s = load_settings()
    s.ai.mode = "heuristic"
    save_settings(s)
    console.print("AI is off: the engine uses its rules.")


@ai_app.command("status")
def ai_status_cmd():
    """Show which AI is set and whether its key is present."""
    s = load_settings()
    if s.ai.mode != "api":
        console.print("AI is off: the engine uses its rules.")
    else:
        has = "key set" if os.environ.get(SECRET_NAMES[s.ai.provider]) else "[red]no key[/]"
        console.print(f"AI on: {s.ai.provider} / {s.ai.model} ({has})")
    console.print("Providers: " + ", ".join(PROVIDERS))


@ai_app.command("test")
def ai_test():
    """Send one tiny request with the saved settings and report the result."""
    from engine.ai.llm import AIError, get_provider

    s = load_settings()
    prov, msg = get_provider(s.ai)
    if prov is None:
        console.print(f"[red]{msg}[/]")
        raise typer.Exit(1)
    try:
        prov.complete_json("Reply with a single JSON object only.", 'Return {"ok": true}')
    except AIError as e:
        console.print(f"[red]Failed:[/] {e}")
        raise typer.Exit(1)
    console.print(f"[green]Works.[/] {prov.name} / {prov.model} replied. ({prov.usage_line()})")


# ------------------------------------------------------------------ setup
@app.command()
def setup(apply: bool = typer.Option(True, help="Write recommended settings")):
    """Inspect the machine, configure the engine, init the database and write a setup report."""
    from engine.hardware import detect_environment, recommend

    s = load_settings()
    console.print("[bold]Inspecting environment...[/]")
    env = detect_environment(str(ROOT))
    rec = recommend(env)
    if apply:
        s.transcription.model, s.transcription.device, s.transcription.compute_type = rec.whisper_model, rec.whisper_device, rec.whisper_compute
        s.max_concurrent_jobs = rec.max_concurrent_jobs
        s.video.encoder = rec.encoder
        save_settings(s)
    if not SECRETS_PATH.exists():
        SECRETS_PATH.write_text("# Optional credentials. The AI key is normally set from the dashboard Settings tab.\n# ANTHROPIC_API_KEY=\n# OPENAI_API_KEY=\n# TWITCH_CLIENT_ID=\n# TWITCH_CLIENT_SECRET=\n# YOUTUBE_API_KEY=\n", encoding="utf-8")
    Database(s.db_path)
    for d in (s.storage, s.library / "songs", s.library / "artists", ROOT / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    from engine.video.webcam import YUNET_PATH, YUNET_URL
    if not YUNET_PATH.exists():
        import requests
        console.print("Downloading YuNet face model (OpenCV model zoo, ~230 KB)...")
        YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(YUNET_URL, timeout=60)
        r.raise_for_status()
        YUNET_PATH.write_bytes(r.content)

    manual: list[str] = []
    if not env.ffmpeg:
        manual.append("Install FFmpeg and put it on PATH (winget install Gyan.FFmpeg).")
    if s.ai.mode == "api" and not os.environ.get(SECRET_NAMES[s.ai.provider]):
        manual.append("AI mode is on but no API key is set: add it in the dashboard Settings tab (until then the rules are used).")
    manual.append("Optional: AI picks and hooks need an API key (dashboard Settings tab). Without one the engine uses its rules.")
    if env.disk_free_gb < 20:
        manual.append(f"Only {env.disk_free_gb} GB free disk space; clips and audio need room.")
    manual.append("Optional: add TWITCH_CLIENT_ID/SECRET and YOUTUBE_API_KEY to config/secrets.env for discovery.")
    manual.append("Set each streamer's permission status (`clipper streamer permission`). Unknown = never publish-ready.")

    report = ["# Setup report", f"_Generated {time.strftime('%Y-%m-%d %H:%M')}_", "", "## Environment"]
    d = env.to_dict()
    for k in ("os", "python", "cpu", "cpu_threads", "ram_gb", "vram_mb", "cuda_for_whisper", "ffmpeg", "nvenc", "node",
              "yt_dlp", "disk_free_gb"):
        report.append(f"- **{k}**: {d[k]}")
    report += ["", "## Configuration chosen",
               f"- AI: **{'rules only (no AI)' if s.ai.mode == 'heuristic' else s.ai.provider + ' / ' + s.ai.model}**",
               f"- Whisper: `{s.transcription.model}` on {s.transcription.device} ({s.transcription.compute_type})",
               f"- Encoder: {s.video.encoder}; max concurrent jobs: {s.max_concurrent_jobs}",
               "", "## Notes", *[f"- {n}" for n in rec.notes], "", "## Needs your action", *[f"- {m}" for m in manual]]
    (ROOT / "logs" / "setup_report.md").write_text("\n".join(report), encoding="utf-8")
    console.print("\n".join(report))


@app.command()
def doctor():
    """Show detected environment as JSON."""
    from engine.hardware import detect_environment, recommend

    env = detect_environment(str(ROOT))
    console.print_json(json.dumps({"environment": env.to_dict(), "recommendation": recommend(env).__dict__}, default=str))


# ------------------------------------------------------------------ processing
@app.command()
def process(sources: list[str] = typer.Argument(..., help="Video URL(s) or local file(s)"),
            streamer: Optional[str] = typer.Option(None, help="Streamer name (defaults to channel name)"),
            permission: Optional[str] = typer.Option(None, help=f"One of: {', '.join(PERMISSION_STATUSES)}"),
            chat: Optional[Path] = typer.Option(None, help="Chat log file (Twitch/YouTube/CSV) for chat signal"),
            content_type: str = typer.Option("auto", "--type", help="auto | stream | gameplay | game_trailer | movie_trailer | podcast | sports"),
            focus: Optional[str] = typer.Option(None, help="e.g. 'funny reactions' or 'shocking discoveries'"),
            max_clips: Optional[int] = typer.Option(None, help="Clips to render per stream"),
            workers: Optional[int] = typer.Option(None, help="Concurrent streams (default from settings)")):
    """Add sources to the queue and process them now."""
    from engine.pipeline import enqueue, run_queue

    eng = _engine()
    if permission and permission not in PERMISSION_STATUSES:
        raise typer.BadParameter(f"permission must be one of {PERMISSION_STATUSES}")
    from engine import profiles as _profiles
    if content_type != "auto" and content_type not in _profiles.PROFILES:
        raise typer.BadParameter(f"--type must be auto or one of {', '.join(_profiles.PROFILES)}")
    for src in sources:
        try:
            sid = eng.add_source(src, streamer, permission, str(chat.resolve()) if chat else None, content_type=content_type)
            enqueue(eng.db, sid, {"focus": focus, "max_clips": max_clips})
            console.print(f"queued stream {sid}: {src}")
        except Exception as e:
            console.print(f"[red]Cannot add {src}: {e}[/]")
    res = run_queue(eng.s, workers, eng._log)
    console.print(res)
    clips_list(min_score=None, status=None, streamer=None, limit=30)


@app.command()
def reanalyze(stream_id: int, focus: Optional[str] = typer.Option(None), from_step: str = typer.Option("moments"),
              max_clips: Optional[int] = typer.Option(None)):
    """Re-run analysis from a step (cached transcription is reused)."""
    eng = _engine()
    ids = eng.process_stream(stream_id, focus=focus, max_clips=max_clips, force_from=from_step)
    console.print(f"clips: {ids}")


@queue_app.command("add")
def queue_add(sources: list[str], streamer: Optional[str] = None, permission: Optional[str] = None,
              focus: Optional[str] = None, max_clips: Optional[int] = None):
    from engine.pipeline import enqueue

    eng = _engine()
    for src in sources:
        try:
            sid = eng.add_source(src, streamer, permission)
            console.print(f"job {enqueue(eng.db, sid, {'focus': focus, 'max_clips': max_clips})} -> stream {sid}")
        except Exception as e:
            console.print(f"[red]{src}: {e}[/]")


@queue_app.command("run")
def queue_run(workers: Optional[int] = None):
    """Process queued jobs (resumes interrupted ones)."""
    from engine.pipeline import run_queue

    s = load_settings()
    console.print(run_queue(s, workers, lambda m: console.print(m)))


@queue_app.command("list")
def queue_list():
    t = Table("job", "stream", "status", "attempts", "error")
    for r in _db().query("SELECT * FROM jobs ORDER BY id DESC LIMIT 50"):
        t.add_row(str(r["id"]), str(r["stream_id"]), r["status"], str(r["attempts"]), (r["error"] or "")[:80])
    console.print(t)


@app.command()
def streams():
    """List streams and their processing status."""
    t = Table("id", "streamer", "platform", "title", "duration", "viewers", "status", "score")
    for r in _db().query("SELECT s.*, st.name FROM streams s LEFT JOIN streamers st ON st.id=s.streamer_id ORDER BY s.id DESC"):
        t.add_row(str(r["id"]), r["name"] or "-", r["platform"], (r["title"] or "")[:40], _fmt_t(r["duration"]),
                  str(r["viewer_count"] or "-"), r["status"], str(r["discovery_score"] or "-"))
    console.print(t)


@app.command()
def transcript(stream_id: int, fmt: str = typer.Option("txt", help="txt | srt | json"), out: Optional[Path] = None):
    """Export a stream transcript."""
    from engine.transcription.whisper import to_srt

    segs = _engine().load_segments(stream_id)
    if fmt == "srt":
        text = to_srt(segs)
    elif fmt == "json":
        text = json.dumps(segs, indent=1)
    else:
        text = "\n".join(f"[{_fmt_t(s['start'])}] {s['text']}" for s in segs)
    if out:
        out.write_text(text, encoding="utf-8")
        console.print(f"wrote {out}")
    else:
        console.print(text)


# ------------------------------------------------------------------ clips
@clips_app.command("list")
def clips_list(min_score: Optional[float] = typer.Option(None), status: Optional[str] = typer.Option(None),
               streamer: Optional[str] = typer.Option(None), limit: int = 50):
    """List clips (highest score first)."""
    sql = ("SELECT c.*, s.platform, st.name streamer, s.title FROM clips c JOIN streams s ON s.id=c.stream_id "
           "LEFT JOIN streamers st ON st.id=s.streamer_id WHERE 1=1")
    params: list = []
    if min_score is not None:
        sql += " AND c.score>=?"
        params.append(min_score)
    if status:
        sql += " AND (c.status=? OR c.approval=?)"
        params += [status, status]
    if streamer:
        sql += " AND st.name LIKE ?"
        params.append(f"%{streamer}%")
    sql += " ORDER BY c.score DESC LIMIT ?"
    params.append(limit)
    t = Table("id", "score", "streamer", "platform", "at", "dur", "status", "approval", "ready", "hook")
    for r in _db().query(sql, params):
        hooks = loads(r["hooks_json"], [])
        t.add_row(str(r["id"]), f"{r['score']:.0f}", r["streamer"] or "-", r["platform"], _fmt_t(r["start"]),
                  f"{r['end'] - r['start']:.0f}s", r["status"], r["approval"], "yes" if r["publish_ready"] else "no",
                  (hooks[r["hook_choice"]] if hooks else "")[:50])
    console.print(t)


@clips_app.command("show")
def clips_show(clip_id: int):
    """Full detail for one clip: scores, AI reasoning, hooks, metadata, QC."""
    db = _db()
    c = db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
    if not c:
        raise typer.Exit(f"no clip {clip_id}")
    cand = db.one("SELECT * FROM candidates WHERE id=?", (c["candidate_id"],))
    out = {
        "clip": {k: c[k] for k in ("id", "stream_id", "start", "end", "score", "status", "approval", "publish_ready",
                                   "publish_block_reason", "video_path", "render_seconds", "error")},
        "hooks": loads(c["hooks_json"]), "hook_choice": c["hook_choice"],
        "category": cand["category"] if cand else None, "summary": cand["summary"] if cand else None,
        "subscores": loads(cand["scores_json"]) if cand else None, "signals": loads(cand["signals_json"]) if cand else None,
        "ai_mode": (loads(cand["ai_json"], {}) or {}).get("mode") if cand else None,
        "layout": loads(c["layout_json"]), "metadata": loads(c["metadata_json"]), "qc": loads(c["qc_json"]),
    }
    console.print_json(json.dumps(out, default=str))


@clips_app.command("candidates")
def clips_candidates(stream_id: int, show_rejected: bool = True):
    """Every evaluated moment for a stream, including rejected ones and why."""
    t = Table("id", "score", "at", "dur", "category", "status", "reason / summary")
    for r in _db().query("SELECT * FROM candidates WHERE stream_id=? ORDER BY total_score DESC", (stream_id,)):
        if r["status"] == "rejected" and not show_rejected:
            continue
        t.add_row(str(r["id"]), f"{r['total_score']:.0f}", _fmt_t(r["start"]), f"{r['end'] - r['start']:.0f}s",
                  r["category"] or "-", r["status"], (r["reject_reason"] or r["summary"] or "")[:70])
    console.print(t)


@clips_app.command("approve")
def clips_approve(clip_ids: list[int]):
    db = _db()
    for cid in clip_ids:
        db.execute("UPDATE clips SET approval='approved', updated_at=? WHERE id=?", (time.time(), cid))
    console.print(f"approved {clip_ids}")


@clips_app.command("reject")
def clips_reject(clip_ids: list[int]):
    db = _db()
    for cid in clip_ids:
        db.execute("UPDATE clips SET approval='rejected', updated_at=? WHERE id=?", (time.time(), cid))
    console.print(f"rejected {clip_ids}")


@clips_app.command("edit")
def clips_edit(clip_id: int,
               start: Optional[float] = typer.Option(None, help="New start (seconds in source)"),
               end: Optional[float] = typer.Option(None, help="New end (seconds in source)"),
               hook: Optional[int] = typer.Option(None, help="Hook index 0-2"),
               hook_text: Optional[str] = typer.Option(None, help="Replace the chosen hook with your own text"),
               layout: Optional[str] = typer.Option(None, help="split | fullcam | gameplay_only"),
               cam_box: Optional[str] = typer.Option(None, help="Manual webcam crop x,y,w,h (source pixels)"),
               focus_x: Optional[float] = typer.Option(None, help="Gameplay horizontal focus 0-1"),
               gameplay_mode: Optional[str] = typer.Option(None, help="fill | fit_blur"),
               font: Optional[str] = None, font_size: Optional[int] = None, highlight: Optional[str] = None,
               words_per_line: Optional[int] = None, position: Optional[str] = None, animation: Optional[str] = None,
               new_hooks: bool = typer.Option(False, help="Regenerate hooks + metadata")):
    """Change boundaries, crop, layout, captions or hook, then re-render."""
    eng = _engine()
    c = eng.db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
    if not c:
        raise typer.Exit(f"no clip {clip_id}")
    fields: dict = {}
    if start is not None:
        fields["start"] = start
    if end is not None:
        fields["end"] = end
    if hook is not None:
        fields["hook_choice"] = hook
    lay = loads(c["layout_json"], {})
    if layout or cam_box:
        lay["manual"] = True
        if layout:
            lay["layout"] = layout
        if cam_box:
            lay["cam_box"] = [int(v) for v in cam_box.split(",")]
            lay.setdefault("layout", "split")
    if focus_x is not None:
        lay["gameplay_focus_x"] = focus_x
    if gameplay_mode:
        lay["gameplay_mode"] = gameplay_mode
    fields["layout_json"] = lay
    style = loads(c["caption_style_json"], {})
    for k, v in {"font": font, "font_size": font_size, "highlight_color": highlight, "max_words_per_line": words_per_line,
                 "position": position, "animation": animation}.items():
        if v is not None:
            style[k] = v
    fields["caption_style_json"] = style
    if hook_text:
        meta = loads(c["metadata_json"], {})
        hooks = meta.get("hooks", [])
        idx = hook if hook is not None else c["hook_choice"]
        while len(hooks) <= idx:
            hooks.append("")
        hooks[idx] = hook_text
        meta["hooks"] = hooks
        fields["metadata_json"] = meta
    eng.db.update("clips", clip_id, fields)
    eng.render_clip(clip_id, regenerate_meta=new_hooks)
    clips_show(clip_id)


@clips_app.command("open")
def clips_open(clip_id: int):
    """Open the rendered clip in the default player."""
    import os

    c = _db().one("SELECT video_path FROM clips WHERE id=?", (clip_id,))
    if c and c["video_path"]:
        os.startfile(c["video_path"]) if hasattr(os, "startfile") else subprocess.run(["xdg-open", c["video_path"]])


@app.command()
def export(clip_ids: Optional[list[int]] = typer.Argument(None, help="Clip ids (default: all approved)"),
           include_unverified: bool = typer.Option(False, help="Also export clips whose creator permission is not verified (labelled)")):
    """Export approved clips with metadata, post text and attribution."""
    eng = _engine()
    if not clip_ids:
        clip_ids = [r["id"] for r in eng.db.query("SELECT id FROM clips WHERE approval='approved' AND status='rendered'")]
    paths = eng.export(clip_ids, include_unverified)
    for p in paths:
        console.print(f"exported -> {p}")
    if not paths:
        console.print("nothing exported (approve clips first: `clipper clips approve <id>`)")


# ------------------------------------------------------------------ streamers
@streamer_app.command("add")
def streamer_add(name: str, platform: str = typer.Option(..., help="twitch | youtube | kick"),
                 url: Optional[str] = None, permission: str = typer.Option("unknown"),
                 watch: bool = typer.Option(True, help="Add to watchlist")):
    eng = _engine()
    sid = eng.upsert_streamer(name, platform, url, permission)
    eng.db.update("streamers", sid, {"watchlist": watch})
    console.print(f"streamer {sid}: {name} ({platform}) permission={permission}")


@streamer_app.command("permission")
def streamer_permission(name: str, status: str = typer.Argument(..., help=" | ".join(PERMISSION_STATUSES)),
                        monetize: Optional[bool] = typer.Option(None, help="Creator allows monetized reuse"),
                        added_music: Optional[bool] = None, promotion: Optional[bool] = None,
                        attribution: Optional[str] = typer.Option(None, help="e.g. 'Clip: {streamer} - {url}'"),
                        rules: Optional[str] = typer.Option(None, help="Clipping program rules text/link")):
    if status not in PERMISSION_STATUSES:
        raise typer.BadParameter(f"status must be one of {PERMISSION_STATUSES}")
    db = _db()
    rows = db.query("SELECT id FROM streamers WHERE name=?", (name,))
    if not rows:
        raise typer.Exit(f"no streamer named {name}")
    fields = {"permission_status": status}
    for k, v in {"allows_monetization": monetize, "allows_added_music": added_music,
                 "allows_third_party_promotion": promotion, "requires_attribution_format": attribution,
                 "clipping_program_rules": rules}.items():
        if v is not None:
            fields[k] = v
    if status in ("explicitly_authorized", "official_clipping_program"):
        fields["allows_clipping"] = True
    for r in rows:
        db.update("streamers", r["id"], fields)
    console.print(f"{name}: {fields}")


@streamer_app.command("list")
def streamer_list():
    t = Table("id", "name", "platform", "permission", "monetize", "fav", "watch", "excluded")
    for r in _db().query("SELECT * FROM streamers ORDER BY favorite DESC, name"):
        t.add_row(str(r["id"]), r["name"], r["platform"], r["permission_status"], str(r["allows_monetization"]),
                  "★" if r["favorite"] else "", "✓" if r["watchlist"] else "", "✗" if r["excluded"] else "")
    console.print(t)


@streamer_app.command("favorite")
def streamer_favorite(name: str, off: bool = False):
    _db().execute("UPDATE streamers SET favorite=? WHERE name=?", (0 if off else 1, name))


@streamer_app.command("exclude")
def streamer_exclude(name: str, off: bool = False):
    _db().execute("UPDATE streamers SET excluded=? WHERE name=?", (0 if off else 1, name))


# ------------------------------------------------------------------ discovery
@discover_app.command("streams")
def discover_streams(query: str = typer.Option("GTA 6", help="Search query"),
                     live: bool = typer.Option(True, help="--no-live for recent uploads/VODs"),
                     limit: int = 20, platform: list[str] = typer.Option(["twitch", "youtube"]),
                     save: bool = typer.Option(False, help="Store results as discovered streams")):
    """Find today's biggest GTA 6 streams/VODs. Requires API keys in config/secrets.env."""
    import time as _t

    from engine.discovery.platforms import discover

    s = load_settings()
    db = Database(s.db_path)
    perms = {f"{r['platform']}:{r['name'].lower()}": r["permission_status"] for r in db.query("SELECT * FROM streamers")}
    excluded = {f"{r['platform']}:{r['name'].lower()}" for r in db.query("SELECT * FROM streamers WHERE excluded=1")}
    found, notes = discover(query, s.vertical_keywords, live, limit, platform, perms)
    found = [f for f in found if f"{f.platform}:{f.streamer.lower()}" not in excluded]
    t = Table("score", "streamer", "platform", "viewers", "views", "duration", "live", "title", "url")
    for f in found:
        t.add_row(f"{f.discovery_score:.0f}", f.streamer, f.platform, str(f.viewer_count if f.viewer_count is not None else "n/a"),
                  str(f.total_views if f.total_views is not None else "n/a"), _fmt_t(f.duration_s), "●" if f.is_live else "",
                  f.title[:40], f.url)
        if save:
            now = _t.time()
            sid = db.one("SELECT id FROM streamers WHERE platform=? AND name=?", (f.platform, f.streamer))
            sid = sid["id"] if sid else db.insert("streamers", {"name": f.streamer, "platform": f.platform,
                                                                "channel_url": f.channel_url, "created_at": now})
            db.execute("INSERT OR IGNORE INTO streams(streamer_id, platform, source_url, title, channel_url, duration, started_at, "
                       "viewer_count, total_views, is_live, discovery_score, status, created_at, updated_at) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (sid, f.platform, f.url, f.title, f.channel_url, f.duration_s, f.started_at, f.viewer_count,
                        f.total_views, int(f.is_live), f.discovery_score, "discovered", now, now))
    console.print(t)
    for n in notes:
        console.print(f"[yellow]{n}[/]")


@discover_app.command("vods")
def discover_vods(login: str, limit: int = 3):
    """Recent Twitch VODs for a streamer login."""
    from engine.discovery.platforms import DiscoveryError, Twitch

    try:
        vods = Twitch().recent_vods(login, limit)
    except DiscoveryError as e:
        raise typer.Exit(str(e))
    for v in vods:
        console.print(f"{v.url}  {_fmt_t(v.duration_s)}  views={v.total_views}  {v.title}")


# ------------------------------------------------------------------ music
music_app = typer.Typer(help="Song library and artist brand profiles (music promotion)", no_args_is_help=True)
app.add_typer(music_app, name="music")


@music_app.command("import")
def music_import(file: Path, title: str = typer.Option(...), artist: str = typer.Option(...),
                 confirm_rights: bool = typer.Option(False, "--confirm-rights",
                                                     help="I own or control the rights to this recording"),
                 explicit: bool = False, priority: int = typer.Option(3, min=1, max=5),
                 tiktok_sound: Optional[str] = typer.Option(None, help="Official TikTok sound name/link"),
                 instagram_sound: Optional[str] = None, youtube_sound: Optional[str] = None,
                 lyrics: bool = typer.Option(True, help="Transcribe timed lyrics with Whisper")):
    """Import a song you own, analyze it once (BPM, bars, sections, hook, key, mood, lyrics)."""
    from engine.music.library import import_song

    if not confirm_rights:
        raise typer.BadParameter("Pass --confirm-rights: only songs you own or control can be placed in clips.")
    s = load_settings()
    sid = import_song(Database(s.db_path), s.library / "songs", file, title, artist,
                      "User confirmed ownership/control of rights via CLI", "cli-user", explicit, priority,
                      {k: v for k, v in {"tiktok": tiktok_sound, "instagram": instagram_sound, "youtube_shorts": youtube_sound}.items() if v},
                      s.transcription if lyrics else None, log=lambda m: console.print(m))
    music_show(sid)


@music_app.command("list")
def music_list():
    t = Table("id", "slug", "title", "artist", "bpm", "key", "mood", "priority", "uses")
    db = _db()
    for r in db.query("SELECT s.*, a.name artist, (SELECT COUNT(*) FROM song_usage u WHERE u.song_id=s.id) uses "
                      "FROM songs s LEFT JOIN artists a ON a.id=s.artist_id"):
        a = loads(r["analysis_json"], {}) or {}
        t.add_row(str(r["id"]), r["slug"], r["title"], r["artist"] or "-", str(a.get("bpm", "?")), a.get("key", "?"),
                  ",".join(a.get("mood", {}).get("tags", [])), str(r["campaign_priority"]), str(r["uses"]))
    console.print(t)


@music_app.command("show")
def music_show(song: str):
    """Analysis for a song (id or slug)."""
    from engine.music.library import effective_analysis

    db = _db()
    r = db.one("SELECT * FROM songs WHERE id=? OR slug=?", (song, song))
    if not r:
        raise typer.Exit(f"no song {song}")
    a = effective_analysis(r)
    console.print(f"[bold]{r['title']}[/]  {a.get('bpm')} BPM  {a.get('key')}  mood={a.get('mood')}")
    for s in a.get("sections", []):
        console.print(f"  {_fmt_t(s['start'])}-{_fmt_t(s['end'])}  {s['label']:<7} energy={s['energy']:.2f}"
                      f"{'  <- HOOK' if s.get('is_hook') else ''}{'  (vocals)' if s.get('vocals') else ''}")
    for p in a.get("promo_segments", []):
        console.print(f"  promo {p['length']}s: {p['start']:.2f}-{p['end']:.2f} hook={p['contains_hook']}")
    if a.get("lyrics"):
        console.print(f"  {len(a['lyrics'])} timed lyric lines")


@music_app.command("correct")
def music_correct(song: str, bpm: Optional[float] = None, hook_start: Optional[float] = None,
                  mood_tags: Optional[str] = None, valence: Optional[float] = None, arousal: Optional[float] = None):
    """Override automatic analysis (your corrections always win)."""
    import numpy as np

    from engine.music.library import effective_analysis

    db = _db()
    r = db.one("SELECT * FROM songs WHERE id=? OR slug=?", (song, song))
    ov = loads(r["analysis_overrides_json"], {}) or {}
    a = effective_analysis(r)
    if hook_start is not None:
        bars = np.array(a["bars"])
        db_ = float(bars[np.argmin(np.abs(bars - hook_start))])
        ov["hook"] = {**a["hook"], "start": hook_start, "downbeat": db_}
    if mood_tags or valence is not None or arousal is not None:
        m = dict(a.get("mood", {}))
        if mood_tags:
            m["tags"] = [t.strip() for t in mood_tags.split(",")]
        if valence is not None:
            m["valence"] = valence
        if arousal is not None:
            m["arousal"] = arousal
        m["source"] = "user"
        ov["mood"] = m
    if bpm:
        ov["bpm"] = bpm
    db.execute("UPDATE songs SET analysis_overrides_json=? WHERE id=?", (json.dumps(ov), r["id"]))
    music_show(str(r["id"]))


@music_app.command("artist")
def music_artist(name: str, description: Optional[str] = None, vibe: Optional[str] = typer.Option(None, help="comma list"),
                 good_fit: Optional[str] = typer.Option(None, help="comma list of moment types, e.g. 'chase,getaway,flex'"),
                 avoid: Optional[str] = typer.Option(None, help="comma list, e.g. 'tragedy,slur,sexual,mocking'"),
                 sound_tag: Optional[str] = typer.Option(None, help="e.g. '♪ {artist} – {title}'"),
                 cta: Optional[str] = typer.Option(None, help="comma list, e.g. 'Out now everywhere,Pre-save in bio'"),
                 volume: Optional[str] = typer.Option(None, help="subtle | balanced | featured")):
    """Create/update an artist brand profile (read by the matcher before placing songs)."""
    db = _db()
    row = db.one("SELECT id FROM artists WHERE name=?", (name,))
    aid = row["id"] if row else db.insert("artists", {"name": name, "created_at": time.time()})
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    fields = {k: v for k, v in {
        "brand_description": description, "vibe_keywords_json": split(vibe) if vibe else None,
        "good_fit_moments_json": split(good_fit) if good_fit else None, "avoid_moments_json": split(avoid) if avoid else None,
        "sound_tag_format": sound_tag, "cta_options_json": split(cta) if cta else None, "default_volume": volume}.items() if v is not None}
    if fields:
        db.update("artists", aid, fields)
    r = db.one("SELECT * FROM artists WHERE id=?", (aid,))
    console.print_json(json.dumps({k: r[k] for k in r.keys()}, default=str))


@clips_app.command("music")
def clips_music(clip_id: int, song: Optional[str] = typer.Option(None, help="Force a song slug"),
                mode: Optional[str] = typer.Option(None, help="payoff_drop | reaction_tail | underscore | cold_open"),
                off: bool = typer.Option(False, help="Export without music"),
                on: bool = typer.Option(False, help="Re-enable music")):
    """Put a song on a clip / try another song / change placement mode, then re-render."""
    eng = _engine()
    c = eng.db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
    lay = loads(c["layout_json"], {})
    m = lay.get("music", {})
    if off:
        m = {"off": True}
    elif on:
        m.pop("off", None)
    if song:
        m["song"] = song
    if mode:
        m["mode"] = mode
    lay["music"] = m
    eng.db.update("clips", clip_id, {"layout_json": lay})
    eng.render_clip(clip_id)
    qc = loads(eng.db.one("SELECT qc_json FROM clips WHERE id=?", (clip_id,))["qc_json"], {})
    console.print_json(json.dumps(qc.get("music", {}), default=str))
    for r in eng.db.query("SELECT mp.fit_score, mp.reasoning, mp.chosen FROM music_placements mp WHERE clip_id=? "
                          "ORDER BY fit_score DESC", (clip_id,)):
        console.print(f"{'►' if r['chosen'] else ' '} {r['reasoning']}")


# ------------------------------------------------------------------ housekeeping
@app.command()
def stats():
    """Overview dashboard numbers."""
    console.print_json(json.dumps(_engine().stats()))


@app.command()
def cleanup(audio: bool = typer.Option(False, help="Also delete cached WAVs of processed streams (transcripts are kept)")):
    """Remove temporary files. The song library is never touched."""
    s = load_settings()
    shutil.rmtree(s.temp, ignore_errors=True)
    freed = 0
    for p in (s.storage / "clips").glob("*/source_segment.mp4"):
        freed += p.stat().st_size
        p.unlink()
    if audio:
        db = Database(s.db_path)
        for r in db.query("SELECT id FROM streams WHERE status='processed'"):
            w = s.stream_dir(r["id"]) / "audio.wav"
            if w.exists():
                freed += w.stat().st_size
                w.unlink()
    console.print(f"freed {freed / 1e6:.1f} MB")


@app.command()
def logs(stream_id: Optional[int] = None, limit: int = 40):
    db = _db()
    rows = db.query("SELECT * FROM logs " + ("WHERE stream_id=? " if stream_id else "") + "ORDER BY id DESC LIMIT ?",
                    ((stream_id, limit) if stream_id else (limit,)))
    for r in reversed(rows):
        console.print(f"{time.strftime('%H:%M:%S', time.localtime(r['ts']))} {r['level']:5} {r['stream_id'] or '':>4} {r['message'][:300]}")


@app.command()
def perf(stream_id: Optional[int] = None):
    """Measured per-step timings and peak memory."""
    t = Table("stream", "step", "seconds", "peak RSS MB")
    for r in _db().query("SELECT * FROM stream_steps " + ("WHERE stream_id=? " if stream_id else "") + "ORDER BY stream_id, finished_at",
                         ((stream_id,) if stream_id else ())):
        t.add_row(str(r["stream_id"]), r["step"], f"{r['duration_s']:.1f}", str(r["peak_rss_mb"]))
    console.print(t)


from engine.trends.cli import trends_app  # noqa: E402

app.add_typer(trends_app, name="trends")

validate_app = typer.Typer(help="Check the face-cam finder / music detector on your own real footage", no_args_is_help=True)
app.add_typer(validate_app, name="validate")


@validate_app.command("facecam")
def validate_facecam(video: Path, out: Path = typer.Option(Path("storage/validation/facecam"), help="Where annotated frames go")):
    """Run face-cam detection on 6 spots in a real stream and save frames with the chosen crop drawn on."""
    from engine import validate

    r = validate.facecam(video, out)
    for w in r["windows"]:
        console.print(f"{w['at_s']:>7}s  {w['layout']:<14} conf {w['confidence']}  {w['method']}  -> {w['image']}")
    console.print(r["how_to_read"])


@validate_app.command("music")
def validate_music(video: Path, truth: Optional[str] = typer.Option(None, help='Known ranges, e.g. "0-30:music,30-95:none"'),
                   threshold: float = 0.55):
    """Score the source-music detector against ranges you labelled by ear."""
    from engine import validate

    r = validate.music(video, truth, threshold)
    for k, v in r.items():
        console.print(f"{k}: {v}")


if __name__ == "__main__":
    app()
