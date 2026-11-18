"""Shot-cut detection (cheap visual cue for where a moment really starts).

Trailers and cutscenes put the reveal on a CUT (the shot of the woman with the rocket
launcher), while the loudest audio comes a beat later (the blast). Snapping the payoff to
the cut just before the audio peak makes a song drop hit the reveal, like an editor would.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def detect_cuts(path: Path, start: float, end: float, fps: float = 8.0, threshold: float = 0.45) -> list[float]:
    """Return cut times (seconds in the FILE) between start and end."""
    return [t for t, _ in detect_cuts_scored(path, start, end, fps, threshold)]


def detect_cuts_scored(path: Path, start: float, end: float, fps: float = 8.0,
                       threshold: float = 0.45) -> list[tuple[float, float]]:
    """Cuts as (time in the FILE, strength 0..1+) where strength = how different the two shots are."""
    cap = cv2.VideoCapture(str(path))
    cuts: list[tuple[float, float]] = []
    prev = None
    try:
        t = start
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(src_fps / fps)))
        k = 0
        while t <= end:
            ok, frame = cap.read()
            if not ok:
                break
            if k % step == 0:
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                small = cv2.resize(frame, (160, 90))
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                if prev is not None:
                    corr = cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL)
                    if corr < 1 - threshold:
                        cuts.append((round(t, 3), round(1 - corr, 3)))
                prev = hist
            k += 1
    finally:
        cap.release()
    return cuts


def snap_to_cut(peak: float, cuts: list[float], before: float = 3.0, after: float = 0.3) -> float:
    """Latest cut in [peak - before, peak + after]; the peak itself if there is none."""
    near = [c for c in cuts if peak - before <= c <= peak + after]
    return max(near) if near else peak


def pick_reveal(cuts: list[tuple[float, float]], peak: float, rise_at, before: float = 1.0, after: float = 5.0) -> float:
    """For action/cinematic moments: the reveal is the cut near the audio peak with the biggest
    combination of picture change and the audio build that follows it. `rise_at(t)` -> audio rise (z)
    in the 1.5 s after t. Falls back to the peak when no cut is near. Heuristic: it cannot know WHAT
    is shown, only that the picture and sound change together - a user override always wins."""
    near = [(t, st) for t, st in cuts if peak - before <= t <= peak + after]
    if not near:
        return peak
    return max(near, key=lambda c: c[1] + 0.3 * max(0.0, min(4.0, rise_at(c[0]))))[0]
