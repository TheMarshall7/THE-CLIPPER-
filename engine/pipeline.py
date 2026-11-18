"""End-to-end orchestration with per-step checkpoints (PRD §5, §16, §17, §20, §21).

Every stream runs these steps; each finished step is recorded in
`stream_steps`, so re-running a stream (or resuming after a crash) skips
completed work and never re-transcribes.

    audio -> features -> transcribe -> chat -> moments -> clips
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import psutil

from engine.ai.llm import get_provider
from engine.audio import analysis as audio_an
from engine.captions.ass import build_ass, write_ass
from engine.chat import activity as chat_an
from engine.clipping.boundaries import keep_intervals, remap, sentences, words_in
from engine import profiles
from engine.config import Settings
from engine.db import PUBLISHABLE_PERMISSIONS, Database, loads
from engine.ingestion.sources import SourceError, SourceInfo, classify, download_audio, download_chat, download_section, fetch_info
from engine.intelligence import moments
from engine.media import ffmpeg
from engine.metadata.generate import generate as gen_meta
from engine.music import detect as music_detect
from engine.music import match as music_match
from engine.music.mixing import SongPlan, render_tracks
from engine.video.compose import mux_audio
from engine.scoring.score import dedupe, penalties, rejection_reason, total_score
from engine.transcription.whisper import transcribe_wav
from engine.video.compose import RenderSpec, render
from engine.video.webcam import detect_layout, force_layout, too_blurry_cam, track_face

STEPS = ("audio", "features", "transcribe", "music_scan", "chat", "moments", "clips")


def _peak_rss_mb() -> float:
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1)


class Engine:
    @property
    def s(self) -> Settings:
        """Settings for the calling thread: a recipe-adjusted copy while that thread renders a clip, else the base."""
        return getattr(self._tl, "s", None) or self._base

    @s.setter
    def s(self, value: Settings) -> None:
        self._base = value

    def __init__(self, settings: Settings, log: Callable[[str], None] = print):
        self._tl = threading.local()
        self.s = settings
        self.db = Database(settings.db_path)
        self._log = log
        self._provider: Any = None
        self._provider_msg: Optional[str] = None

    # ------------------------------------------------------------ utils
    def log(self, msg: str, stream_id: Optional[int] = None, level: str = "info") -> None:
        self.db.log(level, msg, stream_id)
        prefix = f"[stream {stream_id}] " if stream_id else ""
        self._log(prefix + msg)

    @property
    def provider(self):
        if self._provider_msg is None:
            self._provider, self._provider_msg = get_provider(self.s.ai)
            self.log(self._provider_msg)
        return self._provider

    def _stream(self, stream_id: int):
        row = self.db.one("SELECT * FROM streams WHERE id=?", (stream_id,))
        if not row:
            raise KeyError(f"No stream {stream_id}")
        return row

    def _info(self, row) -> SourceInfo:
        md = loads(row["metadata_json"], {})
        return SourceInfo(platform=row["platform"], url=row["source_url"], title=row["title"],
                          duration=row["duration"], is_live=bool(row["is_live"]),
                          has_chat_replay=md.get("has_chat_replay", False))

    def _step(self, stream_id: int, name: str, fn: Callable[[], Any], force: bool = False) -> Any:
        done = self.db.one("SELECT * FROM stream_steps WHERE stream_id=? AND step=? AND status='done'", (stream_id, name))
        if done and not force:
            return loads(done["output_json"])
        self.db.execute("UPDATE streams SET status=?, updated_at=? WHERE id=?", (f"running:{name}", time.time(), stream_id))
        t0 = time.time()
        out = fn()
        self.db.execute(
            "INSERT OR REPLACE INTO stream_steps(stream_id, step, status, output_json, duration_s, peak_rss_mb, finished_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (stream_id, name, "done", json.dumps(out), round(time.time() - t0, 2), _peak_rss_mb(), time.time()))
        self.log(f"step '{name}' done in {time.time() - t0:.1f}s", stream_id)
        return out

    # ------------------------------------------------------------ ingest
    def upsert_streamer(self, name: str, platform: str, channel_url: Optional[str] = None,
                        permission: Optional[str] = None) -> int:
        row = self.db.one("SELECT id FROM streamers WHERE platform=? AND name=?", (platform, name))
        if row:
            if permission:
                self.db.update("streamers", row["id"], {"permission_status": permission})
            return row["id"]
        return self.db.insert("streamers", {"name": name, "platform": platform, "channel_url": channel_url,
                                            "permission_status": permission or "unknown", "created_at": time.time()})

    def add_source(self, source: str, streamer: Optional[str] = None, permission: Optional[str] = None,
                   chat_file: Optional[str] = None, content_type: Optional[str] = None) -> int:
        """content_type: 'auto' (decide from the title and the audio) or a profile key from engine/profiles.py
        (stream, gameplay, game_trailer, movie_trailer, podcast, sports)."""
        classify(source)
        info = fetch_info(source)
        existing = self.db.one("SELECT id, metadata_json FROM streams WHERE source_url=?", (info.url,))
        name = streamer or info.uploader or "unknown"
        sid = self.upsert_streamer(name, info.platform, info.channel_url, permission)
        md = {"has_chat_replay": info.has_chat_replay, **info.raw, "content_type": content_type or "auto"}
        if existing:
            old = loads(existing["metadata_json"], {}) or {}
            if content_type:
                old["content_type"] = content_type
            self.db.update("streams", existing["id"], {"streamer_id": sid, "metadata_json": old,
                                                       **({"chat_path": chat_file} if chat_file else {})})
            return existing["id"]
        now = time.time()
        return self.db.insert("streams", {
            "streamer_id": sid, "platform": info.platform, "source_url": info.url, "title": info.title,
            "channel_url": info.channel_url, "duration": info.duration, "started_at": info.started_at,
            "viewer_count": info.concurrent_view_count, "total_views": info.view_count, "is_live": info.is_live,
            "status": "queued", "chat_path": chat_file, "metadata_json": md, "created_at": now, "updated_at": now})

    # ------------------------------------------------------------ steps
    def process_stream(self, stream_id: int, focus: Optional[str] = None, max_clips: Optional[int] = None,
                       force_from: Optional[str] = None, song: Optional[str] = None) -> list[int]:
        row = self._stream(stream_id)
        info = self._info(row)
        sdir = self.s.stream_dir(stream_id)
        forced = set(STEPS[STEPS.index(force_from):]) if force_from else set()
        wav = sdir / "audio.wav"
        t_start = time.time()

        def step_audio():
            src = download_audio(info, sdir / "dl")
            ffmpeg.extract_audio(src, wav)
            dur = ffmpeg.wav_duration(wav)
            # remote sources: the compressed source audio is KEPT - clip renders cut their audio from it
            # (ranged downloads of separate audio streams come back truncated). `clipper cleanup --audio` removes it.
            self.db.update("streams", stream_id, {"audio_path": str(wav), "duration": row["duration"] or dur})
            return {"wav": str(wav), "duration": dur, "seconds": round(time.time() - t_start, 2)}

        a = self._step(stream_id, "audio", step_audio, "audio" in forced)
        if not wav.exists():  # temp files cleaned after an earlier run: re-extract
            a = self._step(stream_id, "audio", step_audio, force=True)
        duration = a["duration"]

        def step_features():
            feat = audio_an.compute_features(wav)
            base = audio_an.baseline(feat)
            ex = audio_an.excitement(feat, base)
            audio_an.save(sdir / "features.npz", feat, ex)
            self.db.execute("UPDATE streamers SET baseline_json=? WHERE id=?", (json.dumps(base.to_dict()), row["streamer_id"]))
            return {"path": str(sdir / "features.npz"), "baseline": base.to_dict(), "hops": int(ex["score"].size)}

        self._step(stream_id, "features", step_features, "features" in forced)

        def step_transcribe():
            def progress(t: float) -> None:
                pct = min(99, int(100 * t / max(1.0, duration)))
                self.db.execute("UPDATE streams SET status=?, updated_at=? WHERE id=?",
                                (f"running:transcribe {pct}%", time.time(), stream_id))
                self.log(f"transcribed {int(t // 60)}:{int(t % 60):02d} of {int(duration // 60)}:{int(duration % 60):02d} ({pct}%)", stream_id)

            segs = transcribe_wav(wav, sdir / "transcript_chunks", self.s.transcription,
                                  log=lambda m: self.log(m, stream_id), progress=progress)
            with self.db.tx() as con:
                con.execute("DELETE FROM transcript_segments WHERE stream_id=?", (stream_id,))
                con.executemany(
                    "INSERT INTO transcript_segments(stream_id,start,end,text,speaker,confidence,words_json) VALUES (?,?,?,?,?,?,?)",
                    [(stream_id, s["start"], s["end"], s["text"], s["speaker"], s["confidence"], json.dumps(s["words"])) for s in segs])
            (sdir / "transcript.json").write_text(json.dumps(segs), encoding="utf-8")
            words = sum(len(s["words"]) for s in segs)
            return {"segments": len(segs), "words": words, "audio_seconds": duration}

        self._step(stream_id, "transcribe", step_transcribe, "transcribe" in forced)

        def step_music_scan():
            """Source-music timeline for the whole stream (M4), using word timings as the speech mask."""
            words = [w for s in self.load_segments(stream_id) for w in s["words"]]
            tl = music_detect.music_timeline(wav, words)
            music_detect.save(sdir / "music.npz", tl)
            base = music_detect.stream_baseline(tl, self.s.music.detect_threshold)
            regs = music_detect.regions(tl, 0, duration, self.s.music.detect_threshold, base["constant_bgm"])
            return {"path": str(sdir / "music.npz"), **base, "regions": len(regs),
                    "music_seconds": round(sum(r["end"] - r["start"] for r in regs), 1)}

        self._step(stream_id, "music_scan", step_music_scan, "music_scan" in forced)

        def step_chat():
            chat_path = self._stream(stream_id)["chat_path"]
            if not chat_path:
                p = download_chat(info, sdir)
                chat_path = str(p) if p else None
            if not chat_path:
                return {"available": False}
            msgs = chat_an.parse_chat(Path(chat_path))
            act = chat_an.activity(msgs, duration)
            self.db.update("streams", stream_id, {"chat_path": chat_path})
            if act is None:
                return {"available": False, "messages": len(msgs)}
            np.savez_compressed(sdir / "chat.npz", **act)
            return {"available": True, "messages": len(msgs), "path": str(sdir / "chat.npz")}

        chat_out = self._step(stream_id, "chat", step_chat, "chat" in forced)

        def step_moments():
            sel0 = self.find_moments(stream_id, focus, chat_out, song=song)
            if not sel0 and self._retry_as_gameplay(stream_id):
                sel0 = self.find_moments(stream_id, focus, chat_out, song=song)
            return {"selected": sel0}

        sel = self._step(stream_id, "moments", step_moments, bool(forced & {"moments"}) or focus is not None)

        def step_clips():
            ids = self.create_clips(stream_id, sel["selected"], max_clips, song)
            return {"clips": ids}

        out = self._step(stream_id, "clips", step_clips, bool(forced & {"clips", "moments"}) or focus is not None)
        note = None
        if not out["clips"]:
            note = self._no_clips_note(stream_id)
            self.log(note, stream_id, "warning")
        self.db.update("streams", stream_id, {"status": "processed", "error": note, "updated_at": time.time()})
        if self.s.cleanup_temp:
            shutil.rmtree(self.s.temp, ignore_errors=True)
        if self._provider is not None and getattr(self._provider, "calls", 0):
            self.log("AI usage so far this session: " + self._provider.usage_line(), stream_id)
        return out["clips"]

    # ------------------------------------------------------------ intelligence
    def load_segments(self, stream_id: int) -> list[dict]:
        rows = self.db.query("SELECT start,end,text,speaker,confidence,words_json FROM transcript_segments "
                             "WHERE stream_id=? ORDER BY start", (stream_id,))
        return [{"start": r["start"], "end": r["end"], "text": r["text"], "speaker": r["speaker"],
                 "confidence": r["confidence"], "words": loads(r["words_json"], [])} for r in rows]

    def resolve_content_type(self, stream_id: int) -> str:
        """'auto' -> a profile key, decided once and stored (with the reason) on the stream.
        Title cues first, then how much of the audio is music. A heuristic: the Video type dropdown on
        New job and the Streams table always override it."""
        row = self._stream(stream_id)
        md = loads(row["metadata_json"], {}) or {}
        ct = profiles.LEGACY.get(md.get("content_type", "auto"), md.get("content_type", "auto"))
        if ct in profiles.PROFILES:
            return ct
        step = self.db.one("SELECT output_json FROM stream_steps WHERE stream_id=? AND step='music_scan'", (stream_id,))
        frac = float((loads(step["output_json"], {}) or {}).get("music_fraction", 0.0)) if step else 0.0
        wpm = self._words_per_min(stream_id)
        ct, why = profiles.detect(row["title"] or "", frac, self.s.vertical_keywords, wpm)
        md["content_type"], md["content_type_reason"] = ct, why
        self.db.update("streams", stream_id, {"metadata_json": md})
        self.log(f"video type: {profiles.get(ct).label} ({why})", stream_id)
        return ct

    def _words_per_min(self, stream_id: int) -> Optional[float]:
        """How much anyone talks in this video (None until it has been transcribed)."""
        row = self._stream(stream_id)
        tstep = self.db.one("SELECT output_json FROM stream_steps WHERE stream_id=? AND step='transcribe'", (stream_id,))
        if not tstep:
            return None
        t = loads(tstep["output_json"], {}) or {}
        mins = float(t.get("audio_seconds") or row["duration"] or 0) / 60.0
        return float(t["words"]) / mins if mins > 0.5 and t.get("words") is not None else None

    def _no_clips_note(self, stream_id: int) -> str:
        """Plain-language reason a video produced nothing, with what to try."""
        rows = self.db.query("SELECT reject_reason, COUNT(*) n FROM candidates WHERE stream_id=? AND status='rejected' "
                             "GROUP BY reject_reason ORDER BY n DESC LIMIT 3", (stream_id,))
        total = self.db.one("SELECT COUNT(*) FROM candidates WHERE stream_id=?", (stream_id,))[0]
        if not total:
            return ("No clips: no standout moments were found. The video may be too quiet or too short; try a lower 'sensitivity' "
                    "or pick the right Video type on New job.")
        top = (rows[0]["reject_reason"] or "").replace("AI rejected: ", "")[:110] if rows else ""
        return (f"No clips: all {total} candidate moments were rejected (most common: {top}). If this video has little talking, "
                "set Video type to 'Gameplay footage' on New job, or add guidance in AI suggestions.")

    def _retry_as_gameplay(self, stream_id: int) -> bool:
        """Every moment was rejected and most rejections are about missing speech, on a video whose type we guessed: it was
        probably misread as a talking video. Switch it to gameplay footage (no speech needed) and let the caller try again."""
        row = self._stream(stream_id)
        md = loads(row["metadata_json"], {}) or {}
        if "content_type_reason" not in md or md.get("content_type") not in ("stream", "podcast"):
            return False                                   # the user chose the type themselves: respect it
        rej = [r["reject_reason"] or "" for r in self.db.query(
            "SELECT reject_reason FROM candidates WHERE stream_id=? AND status='rejected'", (stream_id,))]
        sparse = sum(1 for r in rej if any(w in r.lower() for w in ("speech", "transcript", "silence", "dead air", "words")))
        wpm = self._words_per_min(stream_id)
        hardly_talking = wpm is not None and wpm < 60          # the real test: does anyone talk? (reasons differ with/without AI)
        if not rej or not (hardly_talking or sparse >= max(1, len(rej) // 2)):
            return False
        md["content_type"] = "gameplay"
        md["content_type_reason"] = "no moment had enough speech, so it is treated as gameplay footage"
        self.db.update("streams", stream_id, {"metadata_json": md})
        self.log("Nothing had enough speech to make a clip: trying again as gameplay footage (no talking needed).", stream_id)
        return True

    def profile_for(self, stream_id: int) -> "profiles.Profile":
        return profiles.get(self.resolve_content_type(stream_id))

    def find_moments(self, stream_id: int, focus: Optional[str], chat_out: dict, song: Optional[str] = None) -> list[int]:
        row = self._stream(stream_id)
        sdir = self.s.stream_dir(stream_id)
        _, ex = audio_an.load(sdir / "features.npz")
        # Source music (car radio, BGM) raises loudness/brightness but is NOT the streamer getting
        # excited - discount audio excitement wherever music was detected.
        if (sdir / "music.npz").exists():
            mp = music_detect.load(sdir / "music.npz")["prob"]
            n = min(mp.size, ex["score"].size)
            damp = np.ones(ex["score"].size)
            damp[:n] = 1 - 0.75 * np.clip((mp[:n] - 0.3) / 0.4, 0, 1)
            ex["score"] = ex["score"] * damp
        chat = dict(np.load(chat_out["path"])) if chat_out.get("available") else None
        chat_curve = np.interp(ex["t"], chat["t"], chat["score"]) if chat else None
        units = sentences(self.load_segments(stream_id))
        duration = float(row["duration"] or ex["t"][-1])
        streamer = self.db.one("SELECT * FROM streamers WHERE id=?", (row["streamer_id"],))
        prof = self.profile_for(stream_id)
        S = profiles.effective_settings(self.s, prof)      # this video type's ranking weights, pacing, music defaults
        # a 3 hour video deserves far more candidates than a 10 minute one
        S.clips.candidate_pool = max(S.clips.candidate_pool, int(round(duration / 3600 * S.clips.pool_per_hour)))
        cands = moments.propose(ex, chat, units, S, duration)
        self.log(f"{len(cands)} signal peaks proposed for evaluation ({prof.label})", stream_id)
        if focus:
            self.log(f"Using your instructions: \"{focus[:120]}\"" + ("" if self.provider else " (keyword mode: no AI key)"), stream_id)
        ctx = {"streamer": streamer["name"] if streamer else None, "title": row["title"], "focus": focus,
               "duration": duration, "profile": prof}
        promo_cfg = self._promo_setup(stream_id, prof, song, duration) if prof.song_first else None
        if promo_cfg:
            ctx["music_brief"] = promo_cfg["brief"]     # the AI judge knows exactly how the song will be cut in
        moments.evaluate(cands, units, ctx, S, self.provider, log=lambda m: self.log(m, stream_id))
        for c in cands:
            if promo_cfg:      # music promo: cut the clip around its biggest hit so the chorus can land on it
                from engine.music import promo
                ev = promo.biggest_event(ex["t"], ex["score"], c["start"], c["end"])
                if ev:
                    ev = (promo.refine_onset(sdir / "audio.wav", ev[0]), ev[1])
                    c["payoff"] = c["peak"] = ev[0]
                    c["start"], c["end"] = promo.reshape(ev[0], promo_cfg["build_s"], promo_cfg["after_s"], duration)
                    c["starts_mid_sentence"] = False          # the song carries the opening, not a sentence
                    c["_hit"] = ev[1]
            anchor = c.get("payoff") or c["peak"]
            c["signals"] = moments.signals_for(c["start"], c["end"], anchor, ex, chat_curve, units,
                                               S.video.max_pause if S.video.jump_cuts else None)
            ai = c["ai"]
            c["sub"] = {
                **{k: ai[k] for k in moments.RUBRIC},
                "audio_excitement": round(100 * c["signals"]["audio_peak"], 1),
                "chat_activity": None if c["signals"]["chat_peak"] is None else round(100 * c["signals"]["chat_peak"], 1),
            }
            c["penalties"] = penalties(c["end"] - c["start"], c["signals"]["peak_offset_s"],
                                       bool(c.get("starts_mid_sentence")), S.clips)
            c["total"] = total_score(c["sub"], S.weights, c["penalties"])
            if focus and self.provider is None:        # no AI to read the instructions: nudge the score by their keywords
                words = " ".join(w["word"] for w in words_in(self.load_segments(stream_id), c["start"], c["end"]))
                bias = moments.guidance_bias(focus, f"{words} {c.get('summary') or ''} {c.get('category') or ''}")
                if bias:
                    c["total"] = round(max(0.0, min(100.0, c["total"] + bias)), 1)
                    c["ai"]["guidance_bias"] = bias
            if promo_cfg and c.get("_hit") is not None:  # song first: blend in how well the moment carries the song
                from engine.music import promo
                ws_c = words_in(self.load_segments(stream_id), c["start"], c["end"])
                fit, parts = promo.music_fit(c["payoff"], c.pop("_hit"), ws_c, promo_cfg["conflicts"], (c["start"], c["end"]),
                                             promo_cfg["build_s"], promo_cfg["after_s"])
                c["ai"]["music_fit"] = {"score": fit, **parts, "song": promo_cfg["slug"], "viral": c["total"]}
                c["total"] = promo.blend(c["total"], fit)
            if self.s.review.learn:                    # what you kept / rejected in Review teaches the picks
                from engine import learn
                adj = learn.adjust(self.db, c["sub"], c.get("category"), c["end"] - c["start"], self.s.review.learn_strength)
                if adj:
                    c["total"] = round(max(0.0, min(100.0, c["total"] + adj)), 1)
                    c["ai"]["learned_adj"] = adj
            c["reject"] = rejection_reason(c, S.clips, prof.require_speech)
            c["ai"]["profile"] = prof.key
        survivors = dedupe([c for c in cands if not c["reject"]])
        keep_ids = {id(c) for c in survivors}
        for c in cands:
            if not c["reject"] and id(c) not in keep_ids:
                c["reject"] = "duplicate of a stronger overlapping moment"
        moments.rank(survivors, self.provider, S, log=lambda m: self.log(m, stream_id), profile=prof, focus=focus or "",
                     music_brief=ctx.get("music_brief", ""))
        for c in survivors:
            c["ai"]["rank_note"] = c.get("rank_note")
            c["sub"]["_penalties"] = c["penalties"]
        self.db.execute("DELETE FROM candidates WHERE stream_id=? AND id NOT IN (SELECT candidate_id FROM clips WHERE candidate_id IS NOT NULL)", (stream_id,))
        selected = []
        now = time.time()
        for c in sorted(cands, key=lambda c: c["total"], reverse=True):
            cid = self.db.insert("candidates", {
                "stream_id": stream_id, "start": c["start"], "end": c["end"], "peak": c["peak"], "payoff": c.get("payoff"),
                "category": c.get("category"), "summary": c.get("summary"), "signals_json": c["signals"],
                "scores_json": c["sub"], "total_score": c["total"], "ai_json": c["ai"],
                "status": "rejected" if c["reject"] else "selected", "reject_reason": c["reject"], "created_at": now})
            if not c["reject"]:
                selected.append(cid)
        self.log(f"{len(selected)} moments passed evaluation ({len(cands) - len(selected)} rejected)", stream_id)
        return selected

    def _copy_note(self, prof, music: dict) -> str:
        """Tone for AI-written hooks and captions, plus what the song is doing on screen so the copy doesn't fight it."""
        ch = music.get("chosen")
        if not ch:
            return prof.copy_note
        note = prof.copy_note + f" The clip has the song '{ch['title']}' by {ch['artist']} on it"
        if ch["placement"].get("payoff_sync"):
            note += f"; its chorus hits at {ch['placement']['payoff_sync']['clip_time']:.1f}s on the big moment"
        note += (". The song credit is shown separately, so don't name the song in the hooks. "
                 "Keep hooks very short (a few words) so the viewer stays on the moment and the music.")
        from engine.music import promo
        if promo.wants(self.s.captions.lyrics, prof.song_first):
            note += " Song lyrics appear on screen after the title card: never quote or compete with them."
        return note

    def _promo_setup(self, stream_id: int, prof, song: Optional[str], duration: float) -> Optional[dict]:
        """Music promo: the song the batch is cut for, its build-up / hook lengths and the game-music regions."""
        from engine.music import promo
        from engine.music.library import effective_analysis
        row = promo.choose_song(self.db, self.s, song)
        if row is None:
            self.log("Music promo: no song with confirmed rights and a beat grid yet, so moments are ranked without one", stream_id,
                     "warning")
            return None
        a = effective_analysis(row)
        build_s, after_s = promo.window_lengths(a, self.s.music.promo_build_bars)
        conflicts: list[tuple[float, float]] = []
        tl_path = self.s.stream_dir(stream_id) / "music.npz"
        if tl_path.exists():
            tl = music_detect.load(tl_path)
            base = music_detect.stream_baseline(tl, self.s.music.detect_threshold)
            conflicts = [(r["start"], r["end"]) for r in
                         music_detect.regions(tl, 0, duration, self.s.music.detect_threshold, base["constant_bgm"])]
        self.log(f"Music promo for '{row['title']}': {build_s:.1f}s build-up, chorus on the biggest hit, "
                 f"{after_s:.1f}s of hook after it", stream_id)
        art = self.db.one("SELECT name FROM artists WHERE id=?", (row["artist_id"],))
        brief = promo.ai_brief(row["title"], art["name"] if art else "", a, build_s, after_s, self.s, prof.song_first)
        return {"slug": row["slug"], "build_s": build_s, "after_s": after_s, "conflicts": conflicts, "brief": brief}

    # ------------------------------------------------------------ clips
    def clip_limit(self, stream_id: int, max_clips: Optional[int] = None) -> int:
        """How many clips to make: the user's number, else scaled by video length (3 h -> 60)."""
        if max_clips:
            return int(max_clips)
        hours = float(self._stream(stream_id)["duration"] or 0) / 3600
        return max(self.s.clips.max_clips_per_stream, int(round(hours * self.s.clips.clips_per_hour)))

    def render_workers(self, jobs: int) -> int:
        """Clips rendered at once. Encoding runs on the GPU (NVENC) and the filters on the CPU, so several clips
        overlap well; 'auto' leaves headroom for the rest of the PC."""
        w = self.s.clips.render_workers
        if w <= 0:
            cores = os.cpu_count() or 4
            gpu = self.s.video.encoder in ("h264_nvenc", "auto")
            w = min(6, max(1, cores // 3)) if gpu else max(1, cores // 6)
        return max(1, min(w, jobs))

    def create_clips(self, stream_id: int, candidate_ids: list[int], max_clips: Optional[int], song: Optional[str] = None) -> list[int]:
        limit = self.clip_limit(stream_id, max_clips)
        ids, todo = [], []
        if not song and self.profile_for(stream_id).song_first:
            from engine.music import promo
            row = promo.choose_song(self.db, self.s)
            song = row["slug"] if row is not None else None
        for cid in candidate_ids[:limit]:
            c = self.db.one("SELECT * FROM candidates WHERE id=?", (cid,))
            existing = self.db.one("SELECT id FROM clips WHERE candidate_id=?", (cid,))
            if existing:
                ids.append(existing["id"])
                continue
            now = time.time()
            row = {"stream_id": stream_id, "candidate_id": cid, "start": c["start"],
                   "end": c["end"], "score": c["total_score"], "status": "pending",
                   "created_at": now, "updated_at": now}
            if song == "__none__":                      # the job says: no song on these clips
                row["layout_json"] = {"music": {"off": True}}
            elif song:                                  # the job names the song for these clips
                row["layout_json"] = {"music": {"song": song}}
            clip_id = self.db.insert("clips", row)
            ids.append(clip_id)
            todo.append(clip_id)

        done, lock = [0], threading.Lock()

        def render_one(clip_id: int) -> None:
            try:
                self.render_clip(clip_id)
            except Exception as e:  # one failed render must not stop the others
                from engine import errors
                errors.log_exception(f"render clip {clip_id}", e)
                self.db.update("clips", clip_id, {"status": "failed", "error": errors.friendly(e)[:1000], "updated_at": time.time()})
                self.log(f"clip {clip_id} render failed: {errors.friendly(e)}", stream_id, "error")
            with lock:
                done[0] += 1
                if len(todo) > 3 and (done[0] % 5 == 0 or done[0] == len(todo)):
                    self.log(f"rendered {done[0]}/{len(todo)} clips", stream_id)

        workers = self.render_workers(len(todo))
        if workers <= 1:
            for clip_id in todo:
                render_one(clip_id)
        else:                                           # best-ranked first, several at a time
            self.log(f"rendering {len(todo)} clips, {workers} at a time", stream_id)
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="render") as pool:
                list(pool.map(render_one, todo))
        if todo and self.s.review.auto_run:
            try:
                from engine import review
                r = review.auto_review(self.db, self.s, todo, log=lambda m: self.log(m, stream_id))
                self.log(f"auto-review: rejected {r['rejected']} of {r['checked']} clips", stream_id)
            except Exception as e:      # reviewing must never fail the batch
                self.log(f"auto-review skipped: {e}", stream_id)
        return ids

    def render_more(self, stream_id: int, n: int = 10, song: Optional[str] = None) -> list[int]:
        """Render the next-best moments of a video that passed review but were left out of the first batch."""
        rows = self.db.query(
            "SELECT id FROM candidates WHERE stream_id=? AND status='selected' "
            "AND id NOT IN (SELECT candidate_id FROM clips WHERE candidate_id IS NOT NULL) "
            "ORDER BY total_score DESC LIMIT ?", (stream_id, int(n)))
        return self.create_clips(stream_id, [r["id"] for r in rows], int(n), song)

    def remaining_moments(self, stream_id: int) -> int:
        return self.db.one(
            "SELECT COUNT(*) FROM candidates WHERE stream_id=? AND status='selected' "
            "AND id NOT IN (SELECT candidate_id FROM clips WHERE candidate_id IS NOT NULL)", (stream_id,))[0]

    def recipe_for(self, clip_id: int) -> Optional[dict]:
        """Style recipe for this clip: the one chosen on the clip, else the promoted default (None = off)."""
        from engine.trends import recipes

        clip = self.db.one("SELECT layout_json FROM clips WHERE id=?", (clip_id,))
        chosen = (loads(clip["layout_json"], {}) or {}).get("recipe") if clip else None
        if chosen is not None:
            if chosen.get("off"):
                return None
            return recipes.get(self.db, chosen["recipe_id"]) if chosen.get("recipe_id") else None
        return recipes.default_recipe(self.db)

    def render_clip(self, clip_id: int, regenerate_meta: bool = False) -> dict:
        from engine.trends import recipes

        recipe = self.recipe_for(clip_id)
        if not recipe:
            return self._render_clip(clip_id, regenerate_meta)
        self._tl.s = recipes.apply(self.s, recipe)       # only this thread sees the recipe's settings
        try:
            res = self._render_clip(clip_id, regenerate_meta)
        finally:
            self._tl.s = None
        self.log(f"clip {clip_id} used style recipe {recipe['recipe_id']}")
        return res

    def _render_clip(self, clip_id: int, regenerate_meta: bool = False) -> dict:
        clip = self.db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
        stream = self._stream(clip["stream_id"])
        info = self._info(stream)
        cand = self.db.one("SELECT * FROM candidates WHERE id=?", (clip["candidate_id"],)) if clip["candidate_id"] else None
        streamer = self.db.one("SELECT * FROM streamers WHERE id=?", (stream["streamer_id"],))
        cdir = self.s.clip_dir(clip_id)
        prof = self.profile_for(stream["id"])
        S = profiles.effective_settings(self.s, prof)
        start, end = float(clip["start"]), float(clip["end"])
        dur = end - start
        t0 = time.time()
        self.db.update("clips", clip_id, {"status": "rendering", "updated_at": t0})

        # 1. video for this window only
        if info.platform == "file":
            src, src_start = Path(info.url), start
        else:  # 8 s padding each side so boundary growth/tail extension stays inside the download
            seg_start = max(0.0, start - 8.0)
            src = cdir / "source_segment.mp4"
            meta = loads(clip["layout_json"], {}).get("_segment", {})
            if not src.exists() or meta.get("start", 1e9) > seg_start or meta.get("end", -1) < end + 8.0:
                kept = sorted((self.s.stream_dir(stream["id"]) / "dl").glob("source_audio.*"))
                download_section(info, seg_start, end + 8.0, src, self.s.video.max_download_height,
                                 audio_file=kept[0] if kept else None)
                meta = {"start": seg_start, "end": end + 8.0}
            seg_start = meta.get("start", seg_start)
            src_start = start - seg_start
        pinfo = ffmpeg.probe(src)
        if not pinfo.has_video:
            raise SourceError("Source has no video track; cannot build a vertical clip from audio only.")

        # 2. layout (manual override wins)
        stored = loads(clip["layout_json"], {})
        if stored.get("manual"):
            layout = stored
        elif prof.layout == "frame":
            # trailers, footage and sports: keep the whole shot (faces on screen are actors/players, not a face cam)
            layout = {"layout": "gameplay_only", "cam_box": None, "face_box": None, "confidence": 1.0,
                      "frame_size": [pinfo.width, pinfo.height], "method": f"profile:{prof.key} (whole frame)", "manual": False}
        else:
            layout = detect_layout(src, src_start, src_start + dur).to_dict()
            layout = force_layout(src, src_start, src_start + dur, self.s.video.layout_style, layout)
            z = too_blurry_cam(layout, self.s.video, pinfo.width, pinfo.height) if self.s.video.layout_style == "auto" else None
            if z:      # the camera is so small it would have to be enlarged into a blur: show the whole picture instead
                layout = {"layout": "gameplay_only", "cam_box": None, "face_box": layout.get("face_box"), "confidence": 1.0,
                          "frame_size": layout.get("frame_size"), "method": f"cam too small to show sharply ({z:.1f}x zoom)",
                          "manual": False}
        if info.platform != "file":
            layout["_segment"] = {"start": seg_start, "end": meta.get("end", end + 8.0)}

        # 3. jump-cut plan on the source timeline, protecting the payoff and loud non-speech reactions
        segs = self.load_segments(stream["id"])
        words = words_in(segs, start, end)
        transcript = " ".join(w["word"] for w in words)
        payoff_abs = float(cand["payoff"]) if cand and cand["payoff"] else start + dur / 2
        protect = [(payoff_abs - 1.2, payoff_abs + 1.2)]
        ex_rise = np.zeros(0)
        quiet = None
        try:
            _, ex = audio_an.load(self.s.stream_dir(stream["id"]) / "features.npz")
            ex_rise = ex["z_rise"]
            hot = (ex["score"] > 0.55) & (ex["voiced"] < 0.5)          # screams/explosions without words
            for i in np.flatnonzero(hot & (ex["t"] >= start) & (ex["t"] <= end)):
                protect.append((float(ex["t"][i]) - 0.3, float(ex["t"][i]) + 0.8))

            def quiet(g0: float, g1: float) -> bool:
                """Dead air = mostly near the stream's noise floor. Loud gameplay (gunfire,
                engines, explosions, cutscene audio) between words is content and is kept."""
                sel = (ex["t"] >= g0) & (ex["t"] < g1)
                return not sel.any() or float(ex["voiced"][sel].mean()) < 0.35
        except (OSError, KeyError):
            pass

        def cut_plan(s0: float, e0: float, prot: list) -> tuple[list, list, float]:
            ws = words_in(segs, s0, e0)
            ka = keep_intervals(ws, s0, e0, S.video.max_pause, protect=prot, quiet=quiet) \
                if S.video.jump_cuts else [(s0, e0)]
            return ws, ka, sum(b - a for a, b in ka)

        words, keep_abs, out_dur = cut_plan(start, end, protect)
        # (a) minimum duration must hold AFTER dead air is removed: grow by whole sentences
        units = sentences(segs)
        media_dur = float(stream["duration"] or end)
        for _ in range(8):
            if out_dur >= S.clips.min_duration:
                break
            nxt = next((u for u in units if u["start"] >= end - 0.05 and u["start"] - end < 6), None)
            prv = next((u for u in reversed(units) if u["end"] <= start + 0.05 and start - u["end"] < 6), None)
            if nxt:
                end = min(media_dur, nxt["end"] + 0.6)
            elif prv:
                start = max(0.0, prv["start"] - 0.2)
            else:
                break
            words, keep_abs, out_dur = cut_plan(start, end, protect)
        for _ in range(4):               # still short (no neighbouring sentences)? extend by plain time, tail first
            need = S.clips.min_duration - out_dur
            if need <= 0.05:
                break
            room_after = max(0.0, media_dur - end)
            take_end = min(room_after, need * 1.15 if start <= 0.5 else need * 0.6)
            end += take_end
            start = max(0.0, start - (need * 1.15 - take_end))
            if take_end < 0.05 and start <= 0.0:
                break                    # the whole video is shorter than the minimum
            words, keep_abs, out_dur = cut_plan(start, end, protect)
        # (b) reaction-tail songs need room after the last word: let the reaction breathe (<= 6 s)
        if self.s.music.enabled and not prof.song_first and clip["score"] is not None and clip["score"] >= self.s.music.viral_gate \
                and self.db.one("SELECT id FROM songs") and not loads(clip["layout_json"], {}).get("music", {}).get("off"):
            from engine.music.placement import preferred_modes
            sig0 = loads(cand["signals_json"], {}) if cand else {}
            cat0 = (cand["category"] if cand else None) or ""
            if preferred_modes(cat0, len(words) / max(1e-6, out_dur), sig0.get("audio_peak", 0.5),
                               sum(w["end"] - w["start"] for w in words) / max(1e-6, out_dur))[0] == "reaction_tail":
                last_word = max((w["end"] for w in words), default=end)
                room = end - last_word
                want = self.s.music.min_window_s - 1.0
                if room < want and end + (want - room) <= media_dur:
                    ext = min(6.0, want - room)
                    tl_path = self.s.stream_dir(stream["id"]) / "music.npz"
                    free = True
                    if tl_path.exists():
                        free = not music_detect.regions(music_detect.load(tl_path), end, end + ext,
                                                        self.s.music.detect_threshold)
                    if free:
                        protect.append((last_word, end + ext))
                        end += ext
                        words, keep_abs, out_dur = cut_plan(start, end, protect)
        if (start, end) != (float(clip["start"]), float(clip["end"])):
            if info.platform != "file":   # stay inside the downloaded segment
                start = max(start, seg_start)
                end = min(end, layout["_segment"]["end"])
                words, keep_abs, out_dur = cut_plan(start, end, protect)
            self.db.update("clips", clip_id, {"start": start, "end": end})
            dur = end - start
            transcript = " ".join(w["word"] for w in words)
            src_start = start if info.platform == "file" else start - seg_start
        keep = [(a - start, b - start) for a, b in keep_abs]
        to_out = lambda t: remap(t, keep_abs)  # noqa: E731
        words_out = []
        for w in words:
            a, b = to_out(max(w["start"], start)), to_out(min(w["end"], end))
            if a is not None:
                words_out.append({"start": a, "end": b if b is not None else a + 0.25, "word": w["word"]})
        speech_out = [(w["start"], w["end"]) for w in words_out]
        # the reveal is usually on a CUT near the loudest sound: snap the payoff to it
        try:
            from engine.video.cuts import detect_cuts_scored, pick_reveal, snap_to_cut
            off = src_start - start                       # file time = source time + off
            scored = detect_cuts_scored(src, max(0.0, payoff_abs - 3.5 + off), payoff_abs + 5.5 + off)
            cat_now = (cand["category"] if cand else None) or ""
            action = prof.evidence in ("spectacle", "mixed") or cat_now in ("crazy_gameplay", "dramatic", "surprising_event", "fail")
            if action:
                def rise_at(t_file: float) -> float:
                    i = int((t_file - off) / 0.5)
                    return float(ex_rise[i:i + 3].max()) if 0 <= i < ex_rise.size else 0.0
                snapped = pick_reveal(scored, payoff_abs + off, rise_at) - off
            else:
                snapped = snap_to_cut(payoff_abs + off, [t for t, _ in scored]) - off
            if prof.song_first and abs(snapped - payoff_abs) > 0.3:
                snapped = payoff_abs           # song first: the chorus follows the sound of the hit, not a far-off cut
            if abs(snapped - payoff_abs) > 0.01:
                self.log(f"payoff moved to the scene cut at {snapped:.2f}s (audio peak {payoff_abs:.2f}s)", stream["id"])
                payoff_abs = snapped
        except Exception as e:  # visual cue is optional
            self.log(f"cut detection skipped: {e}", stream["id"])
        payoff_out = to_out(payoff_abs)
        if payoff_out is None:
            payoff_out = min((to_out(a) or 0.0 for a, b in keep_abs if a >= payoff_abs), default=out_dur / 2)

        # 4. music: conflicts -> eligibility -> top-3 matches -> placement
        music = self.plan_music(clip, stream, streamer, cand, start, end, keep_abs, words, words_out, speech_out,
                                payoff_out, out_dur, transcript)
        speeds = None
        if music.get("chosen"):
            tl = self._sync_to_song(clip_id, stream["id"], prof, music, keep_abs, keep, words, words_out,
                                    start, payoff_abs, payoff_out, out_dur)
            keep_abs, keep, speeds = tl["keep_abs"], tl["keep"], tl["speeds"]
            words_out, speech_out, payoff_out, out_dur = tl["words_out"], tl["speech_out"], tl["payoff_out"], tl["out_dur"]
        meta = loads(clip["metadata_json"])
        if meta is None or regenerate_meta:
            meta = gen_meta({
                "streamer": streamer["name"] if streamer else None, "platform": stream["platform"],
                "url": stream["source_url"], "category": cand["category"] if cand else "moment",
                "summary": cand["summary"] if cand else "", "transcript": transcript, "vertical": self.s.vertical,
                "attribution_format": streamer["requires_attribution_format"] if streamer else None,
                "profile_label": prof.label, "copy_note": self._copy_note(prof, music), "hook_style": prof.hook_style,
                "clip_word": prof.clip_word,
            }, self.provider, log=lambda m: self.log(m, stream["id"]))
        if not self.s.captions.credit_in_caption and meta.get("attribution"):       # no creator credit / link in the post text
            meta["description"] = (meta.get("description") or "").replace(meta["attribution"], "").strip()
            for v in (meta.get("platform_variants") or {}).values():
                v["caption"] = v.get("caption", "").replace(meta["attribution"], "").strip()
            meta["attribution"] = ""
        hooks = meta["hooks"]
        hook = hooks[min(clip["hook_choice"], len(hooks) - 1)] if hooks else None
        chosen = music.get("chosen")

        # 5. audio tracks: clean always, mixed when a song is placed (+ intelligibility check)
        audio_src = src.resolve() if pinfo.has_audio else None
        tracks: dict = {}
        stem: Optional[Path] = None
        if audio_src and chosen and music.get("source_removed"):
            try:
                from engine.music import separate
                stem = separate.voice_stem(audio_src, src_start, dur, cdir / "voice_stem.wav")
                music["separation"] = {"model": separate.MODEL, "residual_db": separate.residual_music_db(stem)}
            except Exception as e:
                music["blocked"] = f"source-music removal failed: {e}"[:200]
                chosen = music["chosen"] = None
        if audio_src:
            tracks = self.render_audio(audio_src, src_start, dur, keep, speech_out, words_out, cdir, chosen, voice_only=stem,
                                       speeds=speeds)
            if chosen and not tracks.get("mixed"):
                music["blocked"] = tracks.get("music_dropped", "mix failed intelligibility check")
                chosen = music["chosen"] = None
            elif chosen and (tracks.get("stats") or {}).get("error"):
                music["blocked"] = "the song could not be mixed: " + str(tracks["stats"]["error"])
                chosen = music["chosen"] = None
        elif chosen:                          # no audio track to mix into: never claim a song
            music["blocked"] = "the source has no audio track to mix a song into"
            chosen = music["chosen"] = None
        if chosen:                            # credit / hashtags / description only for a song that is really in the mix
            meta = self.song_metadata(meta, chosen)


        # 6. captions (output timeline)
        style = self.s.captions.model_copy(update=loads(clip["caption_style_json"], {}))
        plat_name = {"youtube": "YouTube", "twitch": "Twitch", "kick": "Kick"}.get(stream["platform"])
        attribution = (f"{streamer['name']}" + (f" · {plat_name}" if plat_name else "")) \
            if streamer and streamer["name"] != "unknown" and style.show_attribution else None
        band_bottom = None
        gm = stored.get("gameplay_mode") or ("fit_blur" if layout["layout"] == "gameplay_only" else self.s.video.gameplay_mode)
        if layout["layout"] == "gameplay_only" and gm == "fit_blur" and pinfo.width:
            band_h = self.s.video.width * pinfo.height / pinfo.width
            band_bottom = int(self.s.video.height / 2 + band_h / 2)
        tag = None
        if chosen and self.s.music.sound_tag:
            p = chosen["placement"]
            fmt = chosen.get("sound_tag_format") or "♪ {artist} – {title}"
            # music promo: the song credit stays up for as long as the song plays (people can't Shazam what they can't name)
            tag_end = p["clip_out"] if prof.song_first else p["clip_in"] + 4.3
            tag = (fmt.format(artist=chosen["artist"], title=chosen["title"]), p["clip_in"] + 0.3, tag_end)
        lyric_lines = None
        if chosen:
            from engine.music import promo
            lyr_forced = (stored.get("music") or {}).get("lyrics")       # this clip's own on/off beats the setting
            if lyr_forced if lyr_forced is not None else promo.wants(style.lyrics, prof.song_first):
                lyric_lines = promo.lyric_lines(promo.song_lyrics(self.db, chosen["song_id"]), chosen["placement"]) or None
        ass = build_ass(words_out, out_dur, style, self.s.video.width, self.s.video.height, layout["layout"],
                        self.s.video.cam_ratio, hook=hook, attribution=attribution, sound_tag=tag, band_bottom=band_bottom,
                        lyrics=lyric_lines)
        ass_file = write_ass(cdir / "captions.ass", ass)

        # 7. render video (+ clean variant by remuxing the same picture)
        out = cdir / "clip.mp4"
        main_audio = Path(tracks["mixed"]) if chosen and tracks.get("mixed") else (Path(tracks["clean"]) if tracks else None)
        if layout.get("layout") == "split" and self.s.video.face_tracking and layout.get("face_box") and not layout.get("manual"):
            try:      # follow the face through the clip; on any problem fall back to the fixed crop
                layout["face_track"] = track_face(src.resolve(), src_start, dur, keep, layout["face_box"])
                if speeds and any(x != 1.0 for x in speeds):      # face positions onto the speed-ramped timeline
                    from engine.music import promo
                    pcs = [(a, b, sp) for (a, b), sp in zip(keep, speeds)]
                    layout["face_track"] = [[promo.retime(t, keep, pcs) or t, cx, cy] for t, cx, cy in layout["face_track"]]
            except Exception as e:
                self.log(f"clip {clip_id}: face tracking skipped: {e}", stream["id"], "warning")
        spec = RenderSpec(src=src.resolve(), src_start=src_start, duration=dur, src_w=pinfo.width, src_h=pinfo.height,
                          src_fps=pinfo.fps, layout=layout, ass_file=ass_file, out=out, audio=main_audio, keep=keep,
                          punch_at=payoff_out if layout.get("layout") in ("split", "fullcam") else None,
                          gameplay_focus_x=float(stored.get("gameplay_focus_x", 0.5)),
                          gameplay_mode=stored.get("gameplay_mode"), speeds=list(speeds or []),
                          fx=chosen.get("fx") if chosen else None)
        render(spec, self.s.video)
        # AAC encoding can overshoot the WAV's true peak on hard transients (gunfire, explosions):
        # measure the real file and, if needed, re-limit with that much extra headroom and swap the audio in
        if main_audio is not None:
            tp_target = self.s.video.true_peak_db
            for _ in range(2):
                tp = ffmpeg.measure_loudness(out).get("true_peak_db")
                if tp is None or tp <= tp_target:
                    break
                tracks = self.render_audio(
                    audio_src, src_start, dur, keep, speech_out, words_out, cdir, chosen,
                    tp=tp_target - (tp - tp_target) - 0.3, check=False, voice_only=stem, speeds=speeds) if audio_src else tracks
                main_audio = Path(tracks["mixed"]) if chosen and tracks.get("mixed") else Path(tracks["clean"])
                tmp = out.with_name("clip_tp.mp4")
                mux_audio(out, main_audio, tmp, self.s.video.audio_bitrate)
                tmp.replace(out)
        clean_path = None
        if chosen and tracks.get("clean"):
            clean_path = mux_audio(out, Path(tracks["clean"]), cdir / "clip_clean.mp4", self.s.video.audio_bitrate)
        thumb = ffmpeg.grab_frame(out, min(max(0.0, payoff_out + 0.4), out_dur - 0.1), cdir / "thumb.jpg")

        # 8. quality control (incl. music checks)
        qc = self.quality_check(out, out_dur, words, cand, streamer, clip["score"])
        qc["checks"]["jump_cuts_removed_s"] = round(dur - out_dur, 2)
        qc["music"] = {k: v for k, v in music.items() if k not in ("chosen", "alternatives")}
        if chosen:
            p = chosen["placement"]
            overlap = sum(max(0.0, min(b, p["clip_out"]) - max(a, p["clip_in"])) for a, b in music.get("conflicts_out", []))
            song_row = self.db.one("SELECT * FROM songs WHERE id=?", (chosen["song_id"],))
            from engine.music.library import effective_analysis
            bars = np.array(effective_analysis(song_row).get("bars", [0]))
            on_bar = lambda t: float(np.min(np.abs(bars - t))) < 0.03  # noqa: E731
            qc["music"].update({"song": chosen["slug"], "fit": chosen["fit"], "mode": p["mode"], "volume": chosen.get("volume"),
                                "intelligibility": tracks.get("intelligibility"),
                                "overlap_with_source_music_s": round(overlap, 3),
                                "enters_on_bar": on_bar(p["song_in"]), "exits_on_bar": on_bar(p["song_out"])})
            if overlap > 0.05:
                qc["blocks"].append("song overlaps detected source music")
                qc["publish_ready"] = False
        if chosen and self.s.music.end_card:
            try:      # M8 end card: added after QC so the checked duration stays the clip's own; clean version has none
                from engine.video.endcard import append_end_card, make_card_image
                art = self.db.one("SELECT * FROM artists WHERE name=?", (chosen["artist"],))
                style = loads(art["visual_style_json"], {}) if art else {}
                ctas = loads(art["cta_options_json"], []) if art else []
                png = make_card_image(cdir / "end_card.png", self.s.video.width, self.s.video.height, chosen["title"],
                                      chosen["artist"], ctas[0] if ctas else "Full song in bio",
                                      accent=style.get("accent", "#FFD400"), background=style.get("background", "#0B0B10"),
                                      cover=Path(style["cover"]) if style.get("cover") else None)
                carded = out.with_name("clip_card.mp4")
                append_end_card(out, carded, png, self.s.music.end_card_seconds, self.s.video)
                carded.replace(out)
                qc["music"]["end_card_s"] = self.s.music.end_card_seconds
            except Exception as e:                  # a card is decoration; never fail the clip over it
                self.log(f"clip {clip_id}: end card skipped: {e}", stream["id"], "warning")
        vs = self.s.video
        if vs.lead_with_best or vs.loop_ending:
            from engine.video.finish import finish
            lead, loop = vs.lead_with_best, vs.loop_ending and not (chosen and self.s.music.end_card)
            for target in (out, clean_path):
                if target is None or not Path(target).exists():
                    continue
                try:      # decoration: a failure here must never lose the clip
                    finish(Path(target), vs, payoff_out, lead, loop)
                except Exception as e:
                    self.log(f"clip {clip_id}: lead-in/loop skipped: {e}", stream["id"], "warning")
            qc["checks"]["lead_in_s"] = vs.lead_seconds if lead else 0
            qc["checks"]["loop_ending"] = loop
        layout.update({k: v for k, v in stored.items() if k in ("gameplay_focus_x", "gameplay_mode", "music", "_folder", "recipe")})
        layout["_keep"] = keep
        layout["_speeds"] = list(speeds) if speeds else None
        fields = {"status": "rendered", "video_path": str(out), "thumb_path": str(thumb), "layout_json": layout,
                  "metadata_json": meta, "hooks_json": hooks, "captions_json": {"words": len(words_out), "duration": out_dur},
                  "qc_json": qc, "publish_ready": qc["publish_ready"],
                  "publish_block_reason": "; ".join(qc["blocks"]) or None,
                  "render_seconds": round(time.time() - t0, 2), "error": None, "updated_at": time.time()}
        self.db.update("clips", clip_id, fields)
        self.save_music(clip_id, stream["id"], music, chosen, clean_path)
        try:
            self.publish_to_library(clip_id)
        except Exception as e:                      # naming is a convenience; never fail a render over it
            self.log(f"clip {clip_id}: could not copy to the Clips folder: {e}", stream["id"], "warning")
        self.log(f"clip {clip_id} rendered ({out_dur:.1f}s after cutting {dur - out_dur:.1f}s of dead air, "
                 f"layout={layout['layout']}, score={clip['score']}, "
                 f"music={'%s/%s fit %.0f' % (chosen['slug'], chosen['placement']['mode'], chosen['fit']) if chosen else music.get('blocked') or 'none'}) "
                 f"in {time.time() - t0:.1f}s", stream["id"])
        return fields

    # ------------------------------------------------------------ readable clip library
    def publish_to_library(self, clip_id: int) -> Optional[Path]:
        """Copy the finished clip into  Clips/<Stream> - <Song> - Clip NN/  (re-renders move/rename it)."""
        from engine import naming

        clip = self.db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
        if not clip or not clip["video_path"] or not Path(clip["video_path"]).exists():
            return None
        label = naming.clip_label(self.db, clip_id)
        root = naming.LIBRARY
        dest = root / label
        marker = dest / ".clip_id"
        if marker.exists() and marker.read_text().strip() != str(clip_id):
            dest = root / f"{label} [{clip['stream_id']}]"     # same title/song/number from another video
            marker = dest / ".clip_id"
        layout = loads(clip["layout_json"], {}) or {}
        old = layout.get("_folder")
        if old and Path(old) != dest and naming.inside(Path(old), root):
            shutil.rmtree(old, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(clip_id))
        base = naming.safe(label, 120)
        for f in dest.glob("*.mp4"):
            f.unlink(missing_ok=True)
        shutil.copy2(clip["video_path"], dest / f"{base}.mp4")
        if clip["thumb_path"] and Path(clip["thumb_path"]).exists():
            shutil.copy2(clip["thumb_path"], dest / "thumbnail.jpg")
        clean = Path(clip["video_path"]).with_name("clip_clean.mp4")
        if clean.exists() and naming.song_title(self.db, clip_id):
            shutil.copy2(clean, dest / f"{base} (no song).mp4")
        layout["_folder"] = str(dest)
        self.db.update("clips", clip_id, {"layout_json": layout})
        return dest

    # ------------------------------------------------------------ music
    def plan_music(self, clip, stream, streamer, cand, start, end, keep_abs, words, words_out, speech_out,
                   payoff_out, out_dur, transcript) -> dict:
        prof = self.profile_for(stream["id"])
        ms = profiles.effective_settings(self.s, prof).music
        res: dict = {"eligible": False, "conflicts": [], "alternatives": [], "chosen": None}
        sdir = self.s.stream_dir(stream["id"])
        tl_path = sdir / "music.npz"
        constant_bgm = False
        regs: list[dict] = []
        if tl_path.exists():
            tl = music_detect.load(tl_path)
            base = music_detect.stream_baseline(tl, ms.detect_threshold)
            constant_bgm = base["constant_bgm"]
            regs = music_detect.regions(tl, start, end, ms.detect_threshold, constant_bgm)
        res["conflicts"] = regs
        # conflicts on the output timeline (clip edges to kept intervals)
        conf_out = []
        for r in regs:
            pts = [remap(t, keep_abs) for t in np.linspace(r["start"], r["end"], 40)]
            pts = [p for p in pts if p is not None]
            if pts:
                conf_out.append((max(0.0, min(pts) - 0.25), min(out_dur, max(pts) + 0.25)))
        manual = loads(clip["layout_json"], {}).get("music", {})
        from engine.trends import recipes as _recipes
        _rm = _recipes.music_mode(self.recipe_for(clip["id"]))
        if _rm and not manual.get("mode") and not manual.get("enter_at"):
            manual = {**manual, "mode": _rm}       # style recipe: preferred song placement style
        content = prof.key
        if manual.get("ignore_detected"):
            # the user listened and confirmed there's no music here (detector false positive, e.g. SFX)
            res["user_override"] = "detected music ignored - user confirmed none"
            conf_out, regs = [], []
        cover = sum(b - a for a, b in conf_out) / max(1e-6, out_dur)
        content = music_detect.music_is_content(words, regs)
        if ms.remove_source_music and (conf_out or constant_bgm) and not content:
            from engine.music import separate
            if separate.available():          # M4 option 2: strip the radio/BGM instead of placing around it
                res["source_removed"] = list(regs)
                conf_out, regs, cover, constant_bgm = [], [], 0.0, False
            else:
                res["separation_note"] = "source-music removal is on but Demucs is not installed"
        mute_source = False
        if manual.get("song") and prof.evidence == "spectacle" and not content and (conf_out or constant_bgm):
            # no commentary to protect, and you picked a song: the footage's own music is swapped out for yours
            res["source_music_replaced"] = list(regs)
            conf_out, regs, cover, constant_bgm = [], [], 0.0, False
            mute_source = True
        if not ms.enabled:
            res["blocked"] = "music module disabled"
        elif not self.db.one("SELECT id FROM songs"):
            res["blocked"] = "no songs in library"
        elif manual.get("off"):
            res["blocked"] = "music turned off for this clip"
        elif (clip["score"] is None or clip["score"] < ms.viral_gate) and not manual.get("song") and not prof.song_first:
            # the gate only governs AUTOMATIC placement; a song the user picks for this clip bypasses it
            res["blocked"] = f"viral score {clip['score']:.0f} below music gate {ms.viral_gate:.0f}"
        else:   # safety rules (permissions, source music, singing along, BGM) always apply
            res["blocked"] = music_match.clip_music_block(streamer, cover, content, constant_bgm)
        res["music_is_content"] = content
        res["constant_bgm"] = constant_bgm
        if res["blocked"]:
            return res
        sig = loads(cand["signals_json"], {}) if cand else {}
        if manual.get("drop_at") is not None:          # user: "land the drop on <this moment>"
            t = remap(float(manual["drop_at"]), keep_abs)
            if t is None:
                res["blocked"] = "the chosen drop point was cut out of the clip"
                return res
            payoff_out = t
        ctx = {"duration": out_dur, "payoff": payoff_out, "conflicts": conf_out, "speech": speech_out,
               "category": (cand["category"] if cand else None) or "interesting_observation", "text": transcript,
               "force_song": manual.get("song"),
               "energy": sig.get("audio_peak", 0.5), "speech_rate": len(words_out) / max(1e-6, out_dur)}
        matches, skipped = music_match.best_matches(self.db, ctx, self.s)
        if manual.get("song"):
            matches = sorted(matches, key=lambda m: m["slug"] != manual["song"])
            if not matches or matches[0]["slug"] != manual["song"]:
                res["blocked"] = f"chosen song can't be placed: " + ("; ".join(x for x in skipped if True)[:300] or "no clean window")
                return res
        if manual.get("mode"):
            from engine.music.placement import place
            for m in matches:
                song = self.db.one("SELECT * FROM songs WHERE id=?", (m["song_id"],))
                from engine.music.library import effective_analysis
                p = place(effective_analysis(song), out_dur, payoff_out, conf_out, speech_out, [manual["mode"]], self.s.music.min_window_s)
                if p:
                    m["placement"] = p
        if manual.get("enter_at") is not None and manual.get("drop_at") is not None and matches:
            from engine.music.library import effective_analysis
            from engine.music.placement import place_custom
            enter = remap(float(manual["enter_at"]), keep_abs)
            if enter is None:
                res["blocked"] = "the chosen song entry point was cut out of the clip"
                return res
            song = self.db.one("SELECT * FROM songs WHERE id=?", (matches[0]["song_id"],))
            p = place_custom(effective_analysis(song), out_dur, enter, payoff_out, conf_out, prefer_text=song["title"],
                             min_window=self.s.music.min_window_s)
            if p is None:
                res["blocked"] = "can't fit the song from that entry to that drop without crossing other music"
                return res
            matches[0]["placement"] = p
            matches[0]["why"] = (f"{matches[0]['title']} · user-directed: in at {enter:.1f}s, chorus lands at "
                                 f"{payoff_out:.1f}s, plays to the end")
        elif matches and not manual.get("mode"):
            # automatic TRAILER placement for action/cinematic moments: song takes over after the last
            # setup line and its title-line chorus lands on the payoff, then rides to the end
            from engine.music.library import effective_analysis
            from engine.music.placement import auto_entry, place_custom
            cat = ctx["category"]
            from engine.music.placement import talk_ratio
            talky = not prof.song_first and (ms.style == "under" or (ms.style == "auto" and talk_ratio(speech_out, out_dur) >= 0.5))      # just talking: keep the song under the voice, no trailer drop
            # "Big moments": aim the drop at a crash / explosion / scream where nobody is saying anything that must be heard
            spot = None
            if prof.song_first and manual.get("drop_at") is None:
                # song first: the clip was cut around its biggest hit (snapped to the scene cut): the chorus lands on it
                spot = {"drop": payoff_out, "reason": "the biggest hit in the clip", "by": "song-first"}
            elif ms.style == "moment" or (ms.style in ("auto", "over") and not talky):
                spot = self._music_spot(stream, start, end, keep_abs, words_out, conf_out, out_dur, cat, ms)
            if spot is not None:
                res["spot"] = spot
            if spot is not None or (not talky and (ms.style == "over" or prof.trailer_music or cat in (
                    "crazy_gameplay", "dramatic", "surprising_event", "fail", "emotional_reaction"))):
                drop_t = spot["drop"] if spot else payoff_out
                for m in matches:
                    song = self.db.one("SELECT * FROM songs WHERE id=?", (m["song_id"],))
                    a = effective_analysis(song)
                    enter = auto_entry(drop_t, speech_out, conf_out, float(a["bar_s"]))
                    p = None
                    if mute_source or prof.song_first:   # song from the first frame, building into the chorus on the hit
                        p = place_custom(a, out_dur, 0.0, drop_t, conf_out, prefer_text=song["title"],
                                         min_window=self.s.music.min_window_s, prefer_hook=prof.song_first)
                    if p is None and enter is not None:
                        p = place_custom(a, out_dur, enter, drop_t, conf_out, prefer_text=song["title"],
                                         min_window=self.s.music.min_window_s, prefer_hook=prof.song_first)
                    if p is not None:
                        p.mode = "moment_drop" if spot else "trailer_drop"
                        p.notes = [(f"{spot['by']}: big moment at {drop_t:.1f}s ({spot['reason']})" if spot else
                                    f"auto: in after the setup at {p.clip_in:.1f}s, chorus on the payoff at {drop_t:.1f}s")]
                        m["placement"] = p
                        m["fit"] = min(100.0, m["fit"] + 8)       # full build-up + hook on the payoff
                        m["why"] = (f"{m['title']} · {'big-moment drop' if spot else 'trailer drop'}: comes in at {p.clip_in:.1f}s, "
                                    f"the '{song['title']}' chorus hits at {drop_t:.1f}s, rides to the end")
                matches.sort(key=lambda m: m["fit"], reverse=True)
        res["skipped"] = skipped
        if not matches:
            res["blocked"] = "no song fits: " + "; ".join(skipped)[:300] if skipped else "no eligible songs"
            return res
        res["eligible"] = True
        for m in matches:
            m["placement"] = m["placement"].to_dict()
        res["alternatives"] = [{k: m[k] for k in ("slug", "title", "fit", "why", "parts")} for m in matches]
        res["chosen"] = matches[0]
        if mute_source:
            res["chosen"]["bed_override"] = -24.0      # the footage's own audio drops to almost nothing under the song
        res["chosen"]["volume"] = manual.get("volume") or ms.volume_preset
        res["chosen"]["song_first"] = prof.song_first
        res["volume"] = res["chosen"]["volume"]
        res["conflicts_out"] = conf_out
        return res

    def _sync_to_song(self, clip_id: int, stream_id: int, prof, music: dict, keep_abs, keep, words, words_out,
                      start: float, payoff_abs: float, payoff_out: float, out_dur: float) -> dict:
        """Lock the edit to the song:
          1. jump cuts onto its beat grid (moves nothing else)
          2. speed ramps: faster into the hit, slow-mo after it, follow-up hits nudged onto bar lines (whole beats only)
          3. a one-beat dropout right before the chorus hits
          4. beat-reactive picture (zoom pulse, flash, shake, colour)
        Returns the clip's timeline, which changes when the footage is sped up or slowed down."""
        from engine.music import promo
        from engine.music.library import effective_analysis
        chosen, ms = music["chosen"], self.s.music
        speech_out = [(w["start"], w["end"]) for w in words_out]
        tl = {"keep_abs": keep_abs, "keep": keep, "speeds": None, "words_out": words_out, "speech_out": speech_out,
              "payoff_out": payoff_out, "out_dur": out_dur}
        row = self.db.one("SELECT * FROM songs WHERE id=?", (chosen["song_id"],))
        if row is None:
            return tl
        row_c = self.db.one("SELECT layout_json FROM clips WHERE id=?", (clip_id,))
        own = ((loads(row_c["layout_json"], {}) or {}).get("music") or {}) if row_c else {}

        def wanted(key: str, setting: str) -> bool:      # this clip's own switch, else the setting
            return own[key] if own.get(key) is not None else promo.wants(setting, prof.song_first)
        a, pl = effective_analysis(row), chosen["placement"]
        if ms.beat_sync_cuts and len(keep_abs) > 1:
            new_keep, shifts = promo.snap_cuts_to_beats(keep_abs, promo.out_beats(a, pl), words,
                                                        (pl["clip_in"], pl["clip_out"]), fixed=[payoff_abs])
            if shifts:
                keep_abs, keep = new_keep, [(x - start, y - start) for x, y in new_keep]
                music["beat_sync"] = {"cuts_moved": len(shifts), "max_shift_s": max(abs(d) for d in shifts)}
                self.log(f"clip {clip_id}: {len(shifts)} cuts moved onto the beat", stream_id)
        tl["keep_abs"], tl["keep"] = keep_abs, keep
        # 2. speed ramps (only when the chorus is synced to a hit)
        sync = pl.get("payoff_sync")
        if sync and wanted("ramps", ms.speed_ramps) and float(a.get("bpm") or 0) > 0:
            beat = 60.0 / float(a["bpm"])
            hit_src = promo.cut_to_src(float(sync["clip_time"]), keep_abs)
            if hit_src is not None:
                pieces, info = promo.ramp_plan(keep_abs, hit_src, beat, words)
                hit_out = promo.src_to_out(hit_src, pieces)
                after = info.get("slow_mo", {}).get("end", hit_src)
                pieces, nudged = promo.nudge_hits(pieces, self._followup_hits(stream_id, after, keep_abs[-1][1]), hit_out,
                                                  float(a.get("bar_s") or 4 * beat), words, after)
                if info or nudged:
                    tl = self._apply_ramps(tl, pieces, start, chosen, music)
                    pl = chosen["placement"]
                    music["ramps"] = {**info, "hits_on_downbeats": nudged}
                    self.log(f"clip {clip_id}: speed ramps {', '.join(info) or 'none'}"
                             + (f", {len(nudged)} more hit(s) landed on a bar" if nudged else ""), stream_id)
        recipe = self.recipe_for(clip_id) or {}
        r_len = (recipe.get("audio") or {}).get("pre_payoff_dropout_s")
        forced = own.get("dropout")
        want = forced if forced is not None else (
            ms.pre_drop_dropout != "off" and bool(r_len or promo.wants(ms.pre_drop_dropout, prof.song_first)))
        if want:
            length = float(r_len) if r_len else (ms.pre_drop_seconds or None)
            win = promo.dropout_window(a, pl, length)
            # never silence a line the viewer has to hear (unless you switched the drop on for this clip yourself)
            if win and (forced or promo.talk_share(promo.meaningful(tl["words_out"]), win[0], win[1]) < 0.2):
                chosen["dropout"] = list(win)
                music["dropout"] = {"start": win[0], "end": win[1], "by": "you" if forced else "auto"}
            elif not win:
                music["dropout_note"] = "no chorus drop in this clip to go silent before"
        elif forced is False:
            music["dropout_note"] = "drop turned off for this clip"
        # 4. the picture moves with the music
        if wanted("fx", ms.beat_fx):
            fx = promo.fx_spec(a, pl, ms.beat_fx_strength)
            if fx:
                chosen["fx"] = music["fx"] = fx
        return tl

    def _apply_ramps(self, tl: dict, pieces, start: float, chosen: dict, music: dict) -> dict:
        """Move everything that lives on the clip's timeline onto the speed-ramped one."""
        from engine.music import promo
        ka = tl["keep_abs"]

        def rt(t):
            return promo.retime(t, ka, pieces)
        words_out = []
        for w in tl["words_out"]:
            a0, b0 = rt(w["start"]), rt(w["end"])
            if a0 is not None:
                words_out.append({**w, "start": a0, "end": b0 if b0 is not None else a0 + 0.25})
        new_dur = promo.out_len(pieces)
        chosen["placement"] = promo.resync_placement(chosen["placement"], ka, pieces, new_dur)
        if (music.get("spot") or {}).get("drop") is not None and chosen["placement"].get("payoff_sync"):
            music["spot"]["drop"] = chosen["placement"]["payoff_sync"]["clip_time"]
        music["conflicts_out"] = [(rt(x) or x, rt(y) or y) for x, y in music.get("conflicts_out", [])]
        new_abs = [(x, y) for x, y, _ in pieces]
        pay = rt(tl["payoff_out"])
        return {"keep_abs": new_abs, "keep": [(x - start, y - start) for x, y in new_abs], "speeds": [sp for _, _, sp in pieces],
                "words_out": words_out, "speech_out": [(w["start"], w["end"]) for w in words_out],
                "payoff_out": pay if pay is not None else tl["payoff_out"], "out_dur": new_dur}

    def _followup_hits(self, stream_id: int, after_src: float, end_src: float, top: int = 4) -> list[float]:
        """Source times where the next big hits start (after the main one), for landing them on bar lines."""
        from engine.music import promo
        try:
            _, ex = audio_an.load(self.s.stream_dir(stream_id) / "features.npz")
        except (OSError, KeyError):
            return []
        t, sc = ex["t"], ex["score"]
        sel = np.flatnonzero((t > after_src + 0.8) & (t < end_src - 0.5) & (sc > 0.5))
        peaks = [i for i in sel if sc[i] >= sc[max(0, i - 1)] and sc[i] >= sc[min(sc.size - 1, i + 1)]]
        peaks = sorted(peaks, key=lambda i: -sc[i])[:top]
        wav = self.s.stream_dir(stream_id) / "audio.wav"
        return sorted(promo.refine_onset(wav, float(t[i])) for i in peaks)

    def _music_spot(self, stream, start, end, keep_abs, words_out, conf_out, out_dur, category, ms) -> Optional[dict]:
        """Where the song should land: a big non-speech moment (AI judgment first, then rules). None if there isn't one."""
        from engine.music import moments
        try:
            fpath = self.s.stream_dir(stream["id"]) / "features.npz"
            events = moments.loud_events(fpath, start, end, remap, keep_abs) if fpath.exists() else []
            prov = self.provider if (ms.ai_placement and self.s.ai.mode == "api") else None
            from engine.music import promo
            prof = self.profile_for(stream["id"])
            extras = []
            if ms.pre_drop_dropout == "on" or promo.wants(ms.pre_drop_dropout, prof.song_first):
                extras.append("All sound cuts out for one beat right before your spot, so nothing important may be said there.")
            if ms.ducking == "smooth":
                extras.append("While people talk the song sits at a steady lower level.")
            return moments.pick_spot(prov, category, words_out, events, conf_out, out_dur, use_ai=prov is not None,
                                     log=lambda m: self.log(m, stream["id"]), extras=" ".join(extras))
        except Exception as e:               # choosing a spot is a refinement: any problem falls back to the normal placement
            self.log(f"music spot search skipped: {e}", stream["id"], "warning")
            return None

    def render_audio(self, src: Path, src_start: float, dur: float, keep, speech_out, words_out, cdir: Path,
                     chosen: Optional[dict], tp: Optional[float] = None, check: bool = True,
                     voice_only: Optional[Path] = None, speeds: Optional[list[float]] = None) -> dict:
        v, ms = self.s.video, self.s.music
        tp = v.true_peak_db if tp is None else tp
        plan = None
        if chosen:
            p = chosen["placement"]
            from engine.music.mixing import PRESETS
            pre = PRESETS.get(chosen.get("volume") or ms.volume_preset, PRESETS["balanced"])
            lvl = pre["level_db"] if p["mode"] != "underscore" else pre["level_db"] - 6
            plan = SongPlan(Path(chosen["path"]), p["song_in"], p["song_out"], p["clip_in"], p["fade_in"], p["fade_out"],
                            level_db=lvl, duck_db=ms.talk_duck_db if ms.talk_duck_db is not None else pre["duck_db"],
                            attack_ms=ms.attack_ms, release_ms=ms.release_ms,
                            hold_s=ms.duck_hold_s if ms.ducking == "smooth" else 0.0,
                            bed_db=chosen.get("bed_override", pre["bed_db"]),
                            dropout=tuple(chosen["dropout"]) if chosen.get("dropout") else None)
        res = render_tracks(src, src_start, dur, keep, speech_out, cdir, v.loudness_lufs, tp, plan, voice_only, speeds)
        if plan is None or len(words_out) < 4 or not check:
            return res
        # intelligibility check (M7): transcribe clean vs mixed; lower music and retry if words get lost
        try:
            base_acc = self._word_accuracy(Path(res["clean"]), words_out)
            for attempt in range(3):
                acc = self._word_accuracy(Path(res["mixed"]), words_out)
                res["intelligibility"] = {"clean": round(base_acc, 3), "mixed": round(acc, 3), "attempt": attempt,
                                          "song_level_db": plan.level_db}
                if base_acc - acc <= ms.intelligibility_max_drop:
                    return res
                plan.level_db -= 4.0
                plan.duck_db += 3.0
                res.update(render_tracks(src, src_start, dur, keep, speech_out, cdir, v.loudness_lufs, tp, plan, voice_only,
                                         speeds))
            if chosen.get("song_first"):         # music promo: the song is the point; keep it at the reduced level
                res["intelligibility"]["kept_for_promo"] = True
                return res
            res.pop("mixed", None)
            res["music_dropped"] = f"speech intelligibility dropped {base_acc - acc:.0%} even at reduced music level"
        except Exception as e:  # never block the clip on the check itself
            self._log(f"intelligibility check skipped: {e}")
        return res

    def _word_accuracy(self, wav: Path, ref_words: list[dict]) -> float:
        import difflib
        import re as _re

        from engine.transcription.whisper import load_model
        import librosa
        model, _ = load_model(self.s.transcription, self._log)
        audio, _ = librosa.load(str(wav), sr=16000, mono=True)   # arrays, not paths (PyAV-version independent)
        segs, _ = model.transcribe(audio, language=self.s.transcription.language, beam_size=1, vad_filter=False)
        hyp = _re.findall(r"[a-z']+", " ".join(s.text for s in segs).lower())
        ref = _re.findall(r"[a-z']+", " ".join(w["word"] for w in ref_words).lower())
        return difflib.SequenceMatcher(None, ref, hyp).ratio() if ref else 1.0

    def song_metadata(self, meta: dict, chosen: dict) -> dict:
        meta = dict(meta)
        credit = f"Song: {chosen['title']} – {chosen['artist']}"
        artist = self.db.one("SELECT * FROM artists WHERE name=?", (chosen["artist"],))
        ctas = loads(artist["cta_options_json"], []) if artist else []
        cta = ctas[0] if ctas else "🎧 Full song in bio"
        tag_a = "#" + "".join(ch for ch in chosen["artist"] if ch.isalnum())
        tag_s = "#" + "".join(ch for ch in chosen["title"] if ch.isalnum())
        meta["song"] = {"credit": credit, "cta": cta, "hashtags": [tag_a, tag_s],
                        "pinned_comment": f"the {chosen['title']} drop at the end 🔥 ({chosen['artist']})"}
        if credit not in meta.get("description", ""):
            meta["description"] = f"{meta.get('description', '')}\n{credit}\n{cta}".strip()
        for p, v in meta.get("platform_variants", {}).items():
            v["caption"] = f"{v['caption']} {tag_a}".strip()
        return meta

    def save_music(self, clip_id: int, stream_id: int, music: dict, chosen: Optional[dict], clean_path) -> None:
        with self.db.tx() as con:
            con.execute("DELETE FROM music_conflict_regions WHERE clip_id=?", (clip_id,))
            con.execute("DELETE FROM music_placements WHERE clip_id=?", (clip_id,))
            for r in music.get("conflicts", []):
                con.execute("INSERT INTO music_conflict_regions(clip_id,stream_id,start,end,type,confidence) VALUES (?,?,?,?,?,?)",
                            (clip_id, stream_id, r["start"], r["end"], r["type"], r["confidence"]))
            for alt in music.get("alternatives", []):
                song = con.execute("SELECT id FROM songs WHERE slug=?", (alt["slug"],)).fetchone()
                is_chosen = bool(chosen and chosen["slug"] == alt["slug"])
                p = chosen["placement"] if is_chosen else {}
                con.execute("INSERT INTO music_placements(clip_id,song_id,mode,song_in,song_out,clip_in,clip_out,payoff_sync_json,"
                            "uses_stem,fit_score,reasoning,chosen) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (clip_id, song["id"] if song else None, p.get("mode"), p.get("song_in"), p.get("song_out"),
                             p.get("clip_in"), p.get("clip_out"), json.dumps(p.get("payoff_sync")),
                             "voice_stem" if (is_chosen and music.get("source_removed")) else "full_mix",
                             alt["fit"], alt["why"], int(is_chosen)))
        if not chosen:                                   # re-rendered without a song: forget the old usage
            self.db.execute("DELETE FROM song_usage WHERE clip_id=?", (clip_id,))
        if chosen:
            prev = self.db.one("SELECT id FROM song_usage WHERE clip_id=?", (clip_id,))
            if prev:
                self.db.execute("UPDATE song_usage SET song_id=? WHERE clip_id=?", (chosen["song_id"], clip_id))
            else:
                music_match.record_usage(self.db, chosen["song_id"], clip_id)
        self.db.execute("UPDATE candidates SET music_eligible=?, music_reason=? WHERE id=(SELECT candidate_id FROM clips WHERE id=?)",
                        (int(bool(chosen)), music.get("blocked") or (chosen["why"] if chosen else None), clip_id))

    def quality_check(self, out: Path, expected: float, words: list[dict], cand, streamer, score: float) -> dict:
        checks: dict[str, Any] = {}
        blocks: list[str] = []
        err = ffmpeg.decode_check(out)
        checks["decodes_cleanly"] = err is None
        if err:
            blocks.append(f"decode errors: {err[:200]}")
        p = ffmpeg.probe(out)
        checks["resolution"] = f"{p.width}x{p.height}"
        if (p.width, p.height) != (self.s.video.width, self.s.video.height):
            blocks.append("wrong resolution")
        checks["duration_delta_s"] = round(p.duration - expected, 3)
        if abs(p.duration - expected) > 0.35:
            blocks.append("duration mismatch (possible sync issue)")
        # picture and sound must both cover the whole clip (format duration alone can hide a short video track)
        checks["video_s"], checks["audio_s"] = p.video_duration, p.audio_duration
        for kind, d in (("video", p.video_duration), ("audio", p.audio_duration)):
            if d is not None and abs(d - expected) > 0.5:
                blocks.append(f"{kind} track is {d:.1f}s but the clip is {expected:.1f}s")
        if p.has_audio:
            loud = ffmpeg.measure_loudness(out)
            checks.update(loud)
            if "integrated_lufs" in loud and abs(loud["integrated_lufs"] - self.s.video.loudness_lufs) > 1.0:
                blocks.append(f"loudness {loud['integrated_lufs']} LUFS off target")
            if loud.get("true_peak_db", -99) > self.s.video.true_peak_db + 0.5:
                blocks.append("true peak too high")
        else:
            blocks.append("no audio")
        min_len = self.s.clips.min_length
        if min_len and expected < min_len - 1.0:
            blocks.append(f"shorter than your minimum clip length ({expected:.0f}s < {min_len:.0f}s)")
        checks["caption_words"] = len(words)
        sig = loads(cand["signals_json"], {}) if cand else {}
        checks["silence_ratio"] = sig.get("silence_ratio")
        checks["score"] = score
        if score is None or score < self.s.clips.quality_threshold:
            blocks.append(f"score below quality threshold {self.s.clips.quality_threshold:.0f}")
        perm = streamer["permission_status"] if streamer else "unknown"
        checks["permission_status"] = perm
        if perm not in PUBLISHABLE_PERMISSIONS:
            blocks.append(f"creator permission is '{perm}' (needs explicit authorization or official clipping program)")
        return {"checks": checks, "blocks": blocks, "publish_ready": not blocks}

    # ------------------------------------------------------------ export
    def export(self, clip_ids: list[int], include_unverified: bool = False) -> list[Path]:
        out_paths = []
        day = time.strftime("%Y-%m-%d")
        for cid in clip_ids:
            clip = self.db.one("SELECT * FROM clips WHERE id=?", (cid,))
            if not clip or clip["status"] != "rendered":
                self._log(f"clip {cid}: not rendered, skipped")
                continue
            qc = loads(clip["qc_json"], {})
            perm_ok = qc.get("checks", {}).get("permission_status") in PUBLISHABLE_PERMISSIONS
            if not perm_ok and not include_unverified:
                self._log(f"clip {cid}: creator permission not verified, skipped (use --include-unverified to export for review only)")
                continue
            from engine import naming
            dest = self.s.exports / day / (naming.safe(naming.clip_label(self.db, cid), 120) + ("" if perm_ok else " - UNVERIFIED RIGHTS"))
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(clip["video_path"], dest / "video.mp4")
            if clip["thumb_path"] and Path(clip["thumb_path"]).exists():
                shutil.copy2(clip["thumb_path"], dest / "thumbnail.jpg")
            stream = self._stream(clip["stream_id"])
            streamer = self.db.one("SELECT * FROM streamers WHERE id=?", (stream["streamer_id"],))
            meta = loads(clip["metadata_json"], {})
            hooks = loads(clip["hooks_json"], [])
            record = {
                "clip_id": cid, "score": clip["score"], "hook": hooks[clip["hook_choice"]] if hooks else None,
                "metadata": meta, "source": {"url": stream["source_url"], "platform": stream["platform"],
                                             "title": stream["title"], "start": clip["start"], "end": clip["end"]},
                "creator": {"name": streamer["name"], "permission_status": streamer["permission_status"],
                            "allows_monetization": streamer["allows_monetization"]},
                "notice": "Permission to clip is not permission to monetize. Automated editing does not guarantee "
                          "platform monetization eligibility.",
            }
            (dest / "metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            lines = [f"HOOK: {record['hook']}", f"TITLE: {meta.get('title', '')}", "", meta.get("description", ""), ""]
            for p, v in meta.get("platform_variants", {}).items():
                lines.append(f"[{p}] {v['caption']}")
            if meta.get("song"):
                lines += ["", f"PINNED COMMENT: {meta['song']['pinned_comment']}"]
            (dest / "post.txt").write_text("\n".join(lines), encoding="utf-8")
            # music: mixed version is video.mp4; clean version + "add sound at" sheet for in-app official sounds
            clean = Path(clip["video_path"]).with_name("clip_clean.mp4")
            placement = self.db.one("SELECT mp.*, s.title, s.official_sounds_json, a.name artist FROM music_placements mp "
                                    "JOIN songs s ON s.id=mp.song_id LEFT JOIN artists a ON a.id=s.artist_id "
                                    "WHERE mp.clip_id=? AND mp.chosen=1", (cid,))
            if placement and clean.exists():
                shutil.copy2(clean, dest / "video_clean_no_song.mp4")
                sounds = loads(placement["official_sounds_json"], {}) or {}
                sheet = [f"SOUND SHEET - clip {cid}", f"Song: {placement['title']} - {placement['artist']}",
                         f"Placement: {placement['mode']}  (song {placement['song_in']:.2f}s -> {placement['song_out']:.2f}s, "
                         f"enters the clip at {placement['clip_in']:.2f}s)", ""]
                for plat in ("tiktok", "instagram", "youtube_shorts"):
                    if sounds.get(plat):
                        sheet.append(f"[{plat}] RECOMMENDED: upload video_clean_no_song.mp4, add official sound "
                                     f"'{sounds[plat]}', start offset {placement['song_in']:.1f}s, original audio 100%, "
                                     f"added sound ~25-35% (matches the mixed version). Tappable sound page = song discovery.")
                    else:
                        sheet.append(f"[{plat}] Song not available as an official sound here: upload video.mp4 (mixed).")
                sheet += ["", "YouTube: if this song is in Content ID via your distributor, whitelist your channel "
                              "before posting at volume, or your own Shorts may be claimed."]
                (dest / "sound_sheet.txt").write_text("\n".join(sheet), encoding="utf-8")
            self.db.update("clips", cid, {"status": "rendered", "updated_at": time.time()})
            self.db.log("info", f"exported clip {cid} to {dest}", stream["id"])
            out_paths.append(dest)
        return out_paths

    # ------------------------------------------------------------ overview
    def stats(self) -> dict:
        q = lambda sql: self.db.one(sql)[0]  # noqa: E731
        storage = sum(f.stat().st_size for f in self.s.storage.rglob("*") if f.is_file()) if self.s.storage.exists() else 0
        return {
            "streams_discovered": q("SELECT COUNT(*) FROM streams"),
            "streams_processed": q("SELECT COUNT(*) FROM streams WHERE status='processed'"),
            "hours_analyzed": round((q("SELECT COALESCE(SUM(duration),0) FROM streams WHERE status='processed'") or 0) / 3600, 2),
            "candidates_found": q("SELECT COUNT(*) FROM candidates WHERE status='selected'"),
            "clips_rendered": q("SELECT COUNT(*) FROM clips WHERE status='rendered'"),
            "clips_approved": q("SELECT COUNT(*) FROM clips WHERE approval='approved'"),
            "clips_rejected": q("SELECT COUNT(*) FROM clips WHERE approval='rejected'"),
            "processing_time_s": round(q("SELECT COALESCE(SUM(duration_s),0) FROM stream_steps") or 0, 1),
            "clips_with_music": q("SELECT COUNT(*) FROM music_placements WHERE chosen=1"),
            "music_ineligible": q("SELECT COUNT(*) FROM candidates c JOIN clips k ON k.candidate_id=c.id WHERE c.music_eligible=0"),
            "songs": q("SELECT COUNT(*) FROM songs"),
            "storage_mb": round(storage / 1e6, 1),
        }


# ------------------------------------------------------------ job queue
def enqueue(db: Database, stream_id: int, payload: dict) -> int:
    existing = db.one("SELECT id FROM jobs WHERE stream_id=? AND status IN ('queued','running')", (stream_id,))
    if existing:
        return existing["id"]
    return db.insert("jobs", {"kind": "process_stream", "stream_id": stream_id, "payload_json": payload,
                              "status": "queued", "created_at": time.time()})


def _claim(db: Database) -> Optional[dict]:
    with db.tx() as con:
        row = con.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
        if not row:
            return None
        con.execute("UPDATE jobs SET status='running', attempts=attempts+1, started_at=? WHERE id=?", (time.time(), row["id"]))
        return dict(row)


def run_queue(settings: Settings, workers: Optional[int] = None, log: Callable[[str], None] = print) -> dict:
    """Process queued jobs with bounded concurrency. Crashed 'running' jobs are resumed."""
    from concurrent.futures import ThreadPoolExecutor

    db = Database(settings.db_path)
    db.execute("UPDATE jobs SET status='queued' WHERE status='running'")
    n = workers or settings.max_concurrent_jobs
    results = {"done": 0, "failed": 0, "requeued": 0}

    def worker():
        eng = Engine(settings, log)
        while True:
            job = _claim(db)
            if not job:
                return
            payload = loads(job["payload_json"], {})
            try:
                eng.process_stream(job["stream_id"], focus=payload.get("focus"), max_clips=payload.get("max_clips"),
                                   force_from=payload.get("force_from"), song=payload.get("song"))
                db.execute("UPDATE jobs SET status='done', error=NULL, finished_at=? WHERE id=?", (time.time(), job["id"]))
                results["done"] += 1
            except Exception as e:
                from engine import errors
                errors.log_exception(f"job {job['id']} (stream {job['stream_id']})", e)
                msg = errors.friendly(e)
                db.log("error", msg + "\n" + traceback.format_exc()[-2000:], job["stream_id"], job["id"])
                retry = job["attempts"] < settings.max_job_attempts and not isinstance(e, (SourceError, KeyError))
                db.execute("UPDATE jobs SET status=?, error=?, finished_at=? WHERE id=?",
                           ("queued" if retry else "failed", msg[:1000], time.time(), job["id"]))
                db.execute("UPDATE streams SET status=?, error=? WHERE id=?", ("failed" if not retry else "queued", msg[:1000], job["stream_id"]))
                log(f"[stream {job['stream_id']}] FAILED: {msg}" + (" (will retry)" if retry else ""))
                results["requeued" if retry else "failed"] += 1

    with ThreadPoolExecutor(max_workers=n) as pool:
        for f in [pool.submit(worker) for _ in range(n)]:
            f.result()
    return results
