"""Check the face-cam finder and the source-music detector on YOUR real footage.

    clipper validate facecam  stream.mp4              -> annotated frames you can eyeball + per-window result
    clipper validate music    stream.mp4 --truth "0-30:music,30-95:none,95-120:music"

The sample clips shipped with the project have no webcam and only synthetic music, so these two checks
need real material. `--truth` lists the time ranges you know have music (by ear); the command reports
precision / recall so the threshold in Settings can be tuned from evidence instead of guesses.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from engine.media import ffmpeg


def facecam(video: Path, out_dir: Path, windows: int = 6, win_s: float = 8.0) -> dict:
    from engine.video import webcam

    info = ffmpeg.probe(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = float(info.duration or 0)
    results = []
    for i in range(windows):
        t0 = max(0.0, (dur - win_s) * (i + 0.5) / windows)
        lay = webcam.detect_layout(video, t0, min(dur, t0 + win_s))
        frame = webcam.sample_frames(video, t0, t0 + win_s, 1)
        png = None
        if frame:
            f = frame[0].copy()
            for box, colour in ((lay.cam_box, (0, 255, 0)), (lay.face_box, (0, 200, 255))):
                if box:
                    x, y, w, h = box
                    cv2.rectangle(f, (x, y), (x + w, y + h), colour, 3)
            png = out_dir / f"window_{i + 1}_{t0:.0f}s.png"
            cv2.imwrite(str(png), f)
        results.append({"at_s": round(t0, 1), "layout": lay.layout, "confidence": round(lay.confidence, 2), "method": lay.method,
                        "cam_box": lay.cam_box, "face_box": lay.face_box, "image": str(png) if png else None})
    kinds = {r["layout"] for r in results}
    return {"video": str(video), "frame": [info.width, info.height], "windows": results,
            "consistent": len(kinds) == 1,
            "how_to_read": "Green box = the crop used for the top panel, orange = the face. If the green box misses the webcam, "
                           "set it by hand with: clipper clips edit <id> --cam-box x,y,w,h  (or in the clip editor)."}


def parse_truth(spec: str) -> list[tuple[float, float, bool]]:
    out = []
    for part in spec.split(","):
        rng, label = part.strip().split(":")
        a, b = rng.split("-")
        out.append((float(a), float(b), label.strip().lower() in ("music", "yes", "1", "true")))
    return out


def music(video: Path, truth: Optional[str] = None, threshold: float = 0.55) -> dict:
    from engine.music import detect

    with tempfile.TemporaryDirectory(prefix="val_") as tmp:
        wav = Path(tmp) / "a.wav"
        ffmpeg.extract_audio(video, wav)
        tl = detect.music_timeline(wav)
    t, p = tl["t"], tl["prob"]
    regs = detect.regions(tl, 0.0, float(t[-1]) if t.size else 0.0, threshold)
    out: dict = {"video": str(video), "threshold": threshold, "music_fraction": round(float((p > threshold).mean()), 3),
                 "regions": [(round(r["start"], 1), round(r["end"], 1), r["type"], round(r["confidence"], 2)) for r in regs]}
    if truth:
        gt = np.zeros(t.size, bool)
        known = np.zeros(t.size, bool)
        for a, b, is_music in parse_truth(truth):
            m = (t >= a) & (t < b)
            gt[m], known[m] = is_music, True
        on = np.zeros(t.size, bool)
        for r in regs:
            on[(t >= r["start"]) & (t < r["end"])] = True
        tp = int((on & gt & known).sum()); fp = int((on & ~gt & known).sum()); fn = int((~on & gt & known).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        out["scored_seconds"] = round(float(known.sum()) * 0.5, 1)
        out["precision"], out["recall"] = round(prec, 3), round(rec, 3)
        out["f1"] = round(2 * prec * rec / max(1e-9, prec + rec), 3)
        out["labelled_music_s"], out["detector_flagged_s"] = round((tp + fn) * 0.5, 1), round((tp + fp) * 0.5, 1)
        if tp + fn == 0 or tp + fp == 0:
            out["advice"] = "Add both music and no-music ranges to --truth to get precision and recall."
        elif rec < 0.8:
            out["advice"] = "Detector misses real music: lower music.detect_threshold (currently %.2f)." % threshold
        elif prec < 0.8:
            out["advice"] = "Detector flags music that is not there: raise music.detect_threshold (currently %.2f)." % threshold
        else:
            out["advice"] = "Detector agrees with your labels on this footage."
    return out
