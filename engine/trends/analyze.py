"""Per-reference-clip visual + audio breakdown (PRD M15). Everything here is measured from the pixels
and samples; anything we cannot measure without OCR / a classifier is recorded as None ("unavailable"),
never guessed."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from engine.media import ffmpeg
from engine.video import cuts as cuts_mod
from engine.video import webcam

BANDS = ("top", "upper", "middle", "lower", "bottom")


def _frames(path: Path, times: list[float]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            if ok:
                out.append(f)
    finally:
        cap.release()
    return out


def _caption_band(frames: list[np.ndarray]) -> dict:
    """Outlined subtitle text = near-white pixels hugging near-black pixels. Density per vertical band."""
    if not frames:
        return {"present": None, "position": None}
    dens = np.zeros(len(BANDS))
    hits = 0
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, (270, 480)), cv2.COLOR_BGR2GRAY)
        white = (g > 235).astype(np.uint8)
        dark = (g < 40).astype(np.uint8)
        near_dark = cv2.dilate(dark, np.ones((5, 5), np.uint8))
        text = white & near_dark
        row = text.sum(axis=1).astype(float)
        bands = np.array([b.sum() for b in np.array_split(row, len(BANDS))]) / (270 * 480 / len(BANDS))
        dens += bands
        hits += int(bands.max() > 0.004)
    dens /= len(frames)
    present = hits >= max(2, len(frames) // 3)
    return {"present": bool(present), "position": BANDS[int(dens.argmax())] if present else None,
            "density": [round(float(d), 4) for d in dens]}


def visual(path: Path, info: ffmpeg.ProbeInfo) -> dict:
    dur = float(info.duration or 0)
    out: dict = {"duration_s": round(dur, 2), "width": info.width, "height": info.height, "fps": round(info.fps or 0, 1)}
    cut_list = cuts_mod.detect_cuts_scored(path, 0.0, dur)
    out["cuts"] = len(cut_list)
    out["cuts_per_10s"] = round(len(cut_list) / max(dur, 1e-6) * 10, 2)
    edges = [0.0] + [t for t, _ in cut_list] + [dur]
    out["avg_shot_s"] = round(float(np.mean(np.diff(edges))), 2) if len(edges) > 1 else None
    # layout over the first 20 s (face cam / split / full-screen)
    try:
        lay = webcam.detect_layout(path, 0.0, min(dur, 20.0))
        cam_ratio = None
        if lay.cam_box and lay.frame_size[1]:
            cam_ratio = round(lay.cam_box[3] / lay.frame_size[1], 3)
        cam_pos = None
        if lay.cam_box and lay.frame_size[0]:
            cx = (lay.cam_box[0] + lay.cam_box[2] / 2) / lay.frame_size[0]
            cy = (lay.cam_box[1] + lay.cam_box[3] / 2) / lay.frame_size[1]
            cam_pos = ("top" if cy < 0.4 else "bottom" if cy > 0.6 else "middle") + ("-left" if cx < 0.4 else "-right" if cx > 0.6 else "")
        out["layout"] = {"kind": lay.layout, "cam_ratio": cam_ratio, "cam_position": cam_pos, "confidence": round(lay.confidence, 2)}
    except Exception as e:
        out["layout"] = {"kind": None, "error": str(e)[:120]}
    # hook frame: first 1.5 s
    first = _frames(path, [0.0, 0.5, 1.0, 1.4])
    faces = sum(1 for f in first if webcam.detect_faces(f)) if first else 0
    motion = None
    if len(first) >= 2:
        d = [float(np.mean(cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)))) for a, b in zip(first, first[1:])]
        motion = round(float(np.mean(d)), 2)
    cap = _caption_band(first)
    out["hook_frame"] = {"face_visible": faces >= len(first) // 2 + 1 if first else None, "motion": motion,
                         "has_text": cap["present"], "text_words": None}   # word count needs OCR
    # captions across the whole clip
    n = 16
    mid = _frames(path, [dur * (i + 0.5) / n for i in range(n)]) if dur else []
    cb = _caption_band(mid)
    out["captions"] = {"present": cb["present"], "position": cb["position"], "words_per_line": None,
                       "animation": None, "emoji": None}   # need OCR / frame-sequence tracking
    # structure: does the end look like the start (clean loop)?
    ends = _frames(path, [0.0, max(0.0, dur - 0.1)])
    if len(ends) == 2:
        h = [cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1], None, [24, 16], [0, 180, 0, 256]) for f in ends]
        for x in h:
            cv2.normalize(x, x)
        out["loops_cleanly"] = bool(cv2.compareHist(h[0], h[1], cv2.HISTCMP_CORREL) > 0.9)
    return out


def audio(path: Path, info: ffmpeg.ProbeInfo, music_threshold: float = 0.55) -> dict:
    if not info.has_audio:
        return {"has_audio": False}
    out: dict = {"has_audio": True}
    out.update({k: v for k, v in ffmpeg.measure_loudness(path).items()})
    with tempfile.TemporaryDirectory(prefix="trend_") as tmp:
        wav = Path(tmp) / "a.wav"
        ffmpeg.extract_audio(path, wav)
        import soundfile as sf
        x, sr = sf.read(str(wav), dtype="float32")
        hop = int(sr * 0.05)
        n = len(x) // hop
        rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
        db = 20 * np.log10(rms)
        t = np.arange(n) * 0.05
        # peak moment, deliberate silence right before it, transient (SFX-like) hits
        pk = int(db.argmax()) if n else 0
        out["peak_at_s"] = round(float(t[pk]), 2) if n else None
        if n and pk > 30:
            before = db[max(0, pk - 30): max(0, pk - 2)]
            out["silence_before_peak"] = bool(before.size and before.min() < db.max() - 35 and db[max(0, pk - 40): pk - 2].mean() < db.max() - 18)
        else:
            out["silence_before_peak"] = None
        d = np.diff(db, prepend=db[:1])
        hits = np.flatnonzero((d > 12) & (db > db.max() - 20))
        out["transient_hits"] = int(np.sum(np.diff(hits, prepend=-99) > 6)) if hits.size else 0
        out["transient_hits_per_10s"] = round(out["transient_hits"] / max(info.duration, 1e-6) * 10, 2)
        # added music: presence, entry point
        try:
            from engine.music import detect as md
            tl = md.music_timeline(wav)
            p = tl["prob"]
            on = np.flatnonzero(p > music_threshold)
            frac = float((p > music_threshold).mean()) if p.size else 0.0
            entry = float(tl["t"][on[0]]) if on.size else None
            dur = float(info.duration or 0)
            kind = "none"
            if frac > 0.05:
                kind = ("cold_open" if entry is not None and entry < 1.0 else
                        "bed" if frac > 0.7 else
                        "late_entry" if entry is not None and entry > dur * 0.5 else "mid_entry")
            out["music"] = {"fraction": round(frac, 3), "entry_s": round(entry, 2) if entry is not None else None, "kind": kind}
        except Exception as e:
            out["music"] = {"fraction": None, "entry_s": None, "kind": None, "error": str(e)[:100]}
        try:
            import librosa
            y = x[: sr * 90]
            bpm = float(np.atleast_1d(librosa.feature.rhythm.tempo(y=y, sr=sr))[0])
            out["bpm"] = round(bpm, 1) if out.get("music", {}).get("kind") not in (None, "none") else None
        except Exception:
            out["bpm"] = None
    return out


def analyze_file(path: Path) -> tuple[dict, dict]:
    info = ffmpeg.probe(path)
    if not info.has_video:
        raise ValueError("reference clip has no video track")
    return visual(path, info), audio(path, info)
