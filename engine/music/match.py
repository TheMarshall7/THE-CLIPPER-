"""Song-to-clip matching (M5) and clip-level music eligibility (M4, M10).

The Music Fit Score is SEPARATE from the viral score: a great song never makes a weak
clip look stronger. Hard rules run first; only eligible songs get scored.
"""
from __future__ import annotations

import datetime as dt
import time
from typing import Optional

from engine.db import Database, loads
from engine.music.library import effective_analysis
from engine.music.placement import Placement, place, preferred_modes

# clip moment -> (arousal, valence) the soundtrack should have
MOMENT_MOOD = {
    "funny": (0.60, 0.80), "fail": (0.60, 0.70), "memorable_dialogue": (0.50, 0.60),
    "crazy_gameplay": (0.90, 0.40), "dramatic": (0.80, 0.25), "surprising_event": (0.70, 0.45),
    "unexpected_discovery": (0.60, 0.55), "first_reaction": (0.60, 0.55), "emotional_reaction": (0.55, 0.40),
    "interesting_observation": (0.45, 0.55), "controversial_opinion": (0.55, 0.40),
    "unexpected_interaction": (0.60, 0.65), "story_development": (0.60, 0.35),
}
WEIGHTS = {"mood": 0.25, "energy_arc": 0.20, "window": 0.20, "lyric": 0.15, "brand": 0.10, "campaign": 0.10}


def clip_music_block(streamer, conflicts_cover: float, music_is_content: bool, constant_bgm: bool) -> Optional[str]:
    """Reasons the music module is disabled for a whole clip (visible in the dashboard/CLI)."""
    if streamer is not None:
        if streamer["allows_added_music"] == 0:
            return "creator does not allow added music"
        if streamer["allows_third_party_promotion"] == 0:
            return "creator does not allow third-party promotion"
        if streamer["permission_status"] == "official_clipping_program" and streamer["allows_added_music"] is None:
            return "clipping program rules unknown for added music (set allows_added_music)"
        if streamer["permission_status"] == "not_authorized":
            return "creator not authorized"
    if music_is_content:
        return "existing music is part of the moment (singing along / reacting) - leaving it alone"
    if constant_bgm:
        return "streamer runs constant background music; song would layer on top"
    if conflicts_cover > 0.85:
        return "source music covers almost the whole clip"
    return None


def song_ineligible(song, artist, clip_text: str, category: str, explicit_ok: bool, db: Database,
                    rotation_window: int, rotation_max: int, campaign_slug: Optional[str]) -> Optional[str]:
    if not db.one("SELECT id FROM song_rights_confirmations WHERE song_id=?", (song["id"],)):
        return "rights not confirmed"
    a = effective_analysis(song)
    if not a or "bars" not in a:
        return "song analysis missing (enter BPM/sections)"
    today = dt.date.today().isoformat()
    if song["campaign_start"] and today < song["campaign_start"] and song["slug"] != campaign_slug:
        return "campaign not started"
    if song["campaign_end"] and today > song["campaign_end"] and song["slug"] != campaign_slug:
        return "campaign ended"
    if song["explicit"] and not explicit_ok:
        return "explicit song; clean audio required"
    if artist is not None:
        avoid = [x.lower() for x in loads(artist["avoid_moments_json"], []) or []]
        hay = f"{category} {clip_text}".lower()
        hit = next((x for x in avoid if x and x in hay), None)
        if hit:
            return f"brand avoid-list: '{hit}'"
    if campaign_slug != song["slug"]:
        recent = [r["song_id"] for r in db.query("SELECT song_id FROM song_usage ORDER BY id DESC LIMIT ?", (rotation_window,))]
        if recent.count(song["id"]) >= rotation_max:
            return f"rotation limit ({rotation_max} in last {rotation_window} clips)"
    return None


def fit_score(song_a: dict, artist, placement: Placement, category: str, clip_energy: float,
              priority: int, lyric_score: Optional[float], usage_recent: int) -> tuple[float, dict, str]:
    ta, tv = MOMENT_MOOD.get(category, (0.6, 0.5))
    mood = song_a.get("mood", {})
    d = ((mood.get("arousal", 0.5) - ta) ** 2 + (mood.get("valence", 0.5) - tv) ** 2) ** 0.5
    mood_s = max(0.0, 100 * (1 - d / 0.8))
    # section energy where the song plays vs how hard the clip hits
    secs = [s for s in song_a.get("sections", []) if s["end"] > placement.song_in and s["start"] < placement.song_out]
    sec_e = max((s["energy"] for s in secs), default=0.5) / max(1e-6, max((s["energy"] for s in song_a.get("sections", [])), default=1))
    arc = max(0.0, 100 * (1 - abs(sec_e - clip_energy)))
    if placement.mode == "payoff_drop":
        arc = min(100.0, arc + 10)                      # hook hits exactly on the payoff
    window = 100 * placement.window_quality
    lyric = 50.0 if lyric_score is None else lyric_score  # neutral when not evaluated, never zero
    good = [x.lower() for x in (loads(artist["good_fit_moments_json"], []) or [])] if artist is not None else []
    brand = 100.0 if any(g in category.replace("_", " ") for g in good) else (60.0 if not good else 40.0)
    camp = 100 * min(5, max(1, priority or 3)) / 5
    parts = {"mood": round(mood_s, 1), "energy_arc": round(arc, 1), "window": round(window, 1),
             "lyric": round(lyric, 1), "brand": round(brand, 1), "campaign": round(camp, 1)}
    total = sum(WEIGHTS[k] * v for k, v in parts.items()) - 4.0 * usage_recent   # diminishing bonus for overuse
    return round(max(0.0, min(100.0, total)), 1), parts, ("lyric relevance not evaluated" if lyric_score is None else "")


def explain(song, placement: Placement, score: float, parts: dict, conflicts: list) -> str:
    seg = f"{_mmss(placement.song_in)}-{_mmss(placement.song_out)}"
    mode = placement.mode.replace("_", " ")
    radio = "source music detected and avoided" if conflicts else "no source music"
    sync = (f"hook downbeat lands on the payoff at {placement.payoff_sync['clip_time']:.1f}s"
            if placement.payoff_sync else f"enters at {placement.clip_in:.1f}s after the reaction")
    return f"{song['title']} ({seg}) · Fit {score:.0f} · {mode}: {sync}; {radio}; mood {parts['mood']:.0f}, energy {parts['energy_arc']:.0f}."


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def best_matches(db: Database, clip_ctx: dict, settings, top: int = 3) -> tuple[list[dict], list[str]]:
    """clip_ctx: duration, payoff, conflicts[(a,b)], speech[(a,b)], category, text, energy, speech_rate, explicit_ok."""
    ms = settings.music
    songs = db.query("SELECT * FROM songs")
    out, skipped = [], []
    try:
        from engine.review import taste_model
        taste = taste_model(db)
    except Exception:
        taste = {}
    from engine.music.placement import talk_ratio
    modes = preferred_modes(clip_ctx["category"], clip_ctx["speech_rate"], clip_ctx["energy"],
                            talk_ratio(clip_ctx["speech"], clip_ctx["duration"]))
    if ms.style == "under":
        modes = ["underscore", "reaction_tail"]
    elif ms.style in ("over", "moment"):
        modes = ["payoff_drop", "reaction_tail"]
    for s in songs:
        artist = db.one("SELECT * FROM artists WHERE id=?", (s["artist_id"],))
        why = song_ineligible(s, artist, clip_ctx["text"], clip_ctx["category"], clip_ctx.get("explicit_ok", False), db,
                              ms.rotation_window, ms.rotation_max, ms.campaign_song)
        if why:
            skipped.append(f"{s['title']}: {why}")
            continue
        a = effective_analysis(s)
        p = place(a, clip_ctx["duration"], clip_ctx["payoff"], clip_ctx["conflicts"], clip_ctx["speech"], modes, ms.min_window_s)
        if p is None:
            skipped.append(f"{s['title']}: no clean music-free window")
            continue
        recent = sum(1 for r in db.query("SELECT song_id FROM song_usage ORDER BY id DESC LIMIT ?", (ms.rotation_window,))
                     if r["song_id"] == s["id"])
        score, parts, note = fit_score(a, artist, p, clip_ctx["category"], clip_ctx["energy"], s["campaign_priority"], None, recent)
        if getattr(settings, "review", None) is not None and settings.review.learn:     # songs you keep get a bonus, rejected ones a penalty
            rate = taste.get(("song", s["slug"]))
            if rate is not None:
                score = min(100.0, max(0.0, score + (rate - 0.5) * 24 * settings.review.learn_strength))
        if ms.campaign_song == s["slug"]:
            score = min(100.0, score + 25)
        out.append({"song_id": s["id"], "slug": s["slug"], "title": s["title"], "artist": artist["name"] if artist else "",
                    "path": s["file_path"], "placement": p, "fit": score, "parts": parts, "note": note,
                    "why": explain(s, p, score, parts, clip_ctx["conflicts"]), "official_sounds": loads(s["official_sounds_json"], {}),
                    "sound_tag_format": artist["sound_tag_format"] if artist else None})
    out.sort(key=lambda m: m["fit"], reverse=True)
    return out[:top], skipped


def record_usage(db: Database, song_id: int, clip_id: int) -> None:
    db.insert("song_usage", {"song_id": song_id, "clip_id": clip_id, "used_at": time.time()})
