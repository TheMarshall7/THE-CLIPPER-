"""Clip boundary intelligence (PRD §9).

Snaps rough [start, end] hints onto sentence boundaries and pauses using the
word-level transcript, then enforces the configured duration limits while
always keeping the peak and payoff inside the clip.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

TERMINAL = (".", "!", "?", "…")


@dataclass
class Window:
    start: float
    end: float
    starts_mid_sentence: bool
    ends_mid_sentence: bool

    @property
    def duration(self) -> float:
        return self.end - self.start


def _clean_start_index(segs: list[dict], i: int) -> bool:
    if i == 0:
        return True
    prev = segs[i - 1]
    gap = segs[i]["start"] - prev["end"]
    return gap >= 0.45 or prev["text"].rstrip().endswith(TERMINAL)


def _clean_end_index(segs: list[dict], i: int) -> bool:
    if segs[i]["text"].rstrip().endswith(TERMINAL):
        return True
    return i + 1 >= len(segs) or segs[i + 1]["start"] - segs[i]["end"] >= 0.6


def snap(segs: list[dict], start_hint: float, end_hint: float, must_include: tuple[float, float],
         min_dur: float, max_dur: float, media_duration: float) -> Window:
    """segs: transcript segments sorted by start (absolute seconds)."""
    keep_lo, keep_hi = min(must_include), max(must_include)
    start_hint = min(start_hint, keep_lo)
    end_hint = max(end_hint, keep_hi)
    if not segs:
        s = max(0.0, start_hint)
        e = min(media_duration, max(end_hint, s + min_dur))
        return Window(s, e, False, False)

    starts = [s["start"] for s in segs]
    # ---- start: last segment starting at/before the hint, walk back to a clean sentence start
    i = max(0, bisect.bisect_right(starts, start_hint) - 1)
    for _ in range(3):
        if _clean_start_index(segs, i) or i == 0:
            break
        if keep_hi - segs[i - 1]["start"] > max_dur:
            break
        i -= 1
    # ---- end: first segment ending at/after the hint, walk forward to a clean sentence end
    j = next((k for k, s in enumerate(segs) if s["end"] >= end_hint), len(segs) - 1)
    j = max(j, i)
    for _ in range(3):
        if _clean_end_index(segs, j) or j + 1 >= len(segs):
            break
        if segs[j + 1]["end"] - segs[i]["start"] > max_dur:
            break
        j += 1

    # ---- duration limits (extend/shrink by whole segments)
    def dur() -> float:
        return segs[j]["end"] - segs[i]["start"]

    while dur() < min_dur and (j + 1 < len(segs) or i > 0):
        nxt = segs[j + 1]["end"] - segs[i]["start"] if j + 1 < len(segs) else 1e9
        prv = segs[j]["end"] - segs[i - 1]["start"] if i > 0 else 1e9
        if nxt <= prv and j + 1 < len(segs):
            j += 1
        elif i > 0:
            i -= 1
        else:
            break
    while dur() > max_dur and i < j:
        if segs[i + 1]["start"] <= keep_lo:
            i += 1
        elif segs[j - 1]["end"] >= keep_hi:
            j -= 1
        else:
            break

    # pad into silence without clipping neighbouring words
    prev_end = segs[i - 1]["end"] if i > 0 else 0.0
    next_start = segs[j + 1]["start"] if j + 1 < len(segs) else media_duration
    start = max(prev_end + 0.05, segs[i]["start"] - 0.3, 0.0)
    end = min(next_start - 0.05, segs[j]["end"] + 0.6, media_duration)
    # a non-speech peak (e.g. explosion, scream) may sit outside the speech segments
    start = min(start, max(0.0, keep_lo - 1.0))
    end = max(end, min(media_duration, keep_hi + 1.0))
    # speech stopped well before the requested end (wordless gunfight/chase): let the action and the
    # reaction after the peak play out instead of ending 1 s after it
    if segs[j]["end"] < end_hint - 2.0 and segs[j]["end"] < keep_hi:
        end = max(end, min(media_duration, end_hint, start + max_dur))
    if end - start < min_dur:  # sparse speech: pad symmetrically
        extra = (min_dur - (end - start)) / 2
        start, end = max(0.0, start - extra), min(media_duration, end + extra)
    if end - start > max_dur:
        start = max(start, keep_hi + 1.0 - max_dur)
        end = start + max_dur
    # a forced edge must never land inside a sentence: grow to that sentence's boundary
    for s in segs:
        if s["start"] < end < s["end"] and s["end"] - start <= max_dur:
            end = min(media_duration, s["end"] + 0.3)
        if s["start"] < start < s["end"] and end - s["start"] <= max_dur:
            start = max(0.0, s["start"] - 0.2)
    return Window(round(start, 3), round(end, 3), not _clean_start_index(segs, i), not _clean_end_index(segs, j))


def sentences(segs: list[dict], max_gap: float = 0.7, max_len: float = 12.0) -> list[dict]:
    """Re-split Whisper segments into sentence units using word timings.
    Whisper/VAD segments can be 20-30 s long; clip boundaries need sentence precision."""
    units: list[dict] = []
    cur: list[dict] = []

    def flush():
        if cur:
            units.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                          "text": " ".join(w["word"] for w in cur), "words": list(cur)})
            cur.clear()

    words = [w for s in segs for w in s.get("words", [])]
    if not words:  # transcript without word timings: fall back to segments
        return [dict(s) for s in segs]
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > max_gap or w["end"] - cur[0]["start"] > max_len):
            flush()
        cur.append(w)
        if w["word"].rstrip().endswith(TERMINAL):
            flush()
    flush()
    return units


def keep_intervals(words: list[dict], start: float, end: float, max_pause: float = 0.6, keep_pause: float = 0.25,
                   protect: list[tuple[float, float]] = (), quiet=None) -> list[tuple[float, float]]:
    """Jump-cut plan: intervals of [start, end] to keep, removing dead air between words.
    Gaps overlapping a `protect` region (a loud non-speech reaction, the payoff) are kept.
    quiet(t0, t1) -> bool: if given, only gaps that are actually quiet get cut - a wordless
    gunfight or car chase is the content, not dead air."""
    ws = [w for w in words if w["end"] > start and w["start"] < end]
    if len(ws) < 2:
        return [(start, end)]
    cuts: list[tuple[float, float]] = []
    edges = [(start, ws[0]["start"])] + [(a["end"], b["start"]) for a, b in zip(ws, ws[1:])] + [(ws[-1]["end"], end)]
    for k, (g0, g1) in enumerate(edges):
        if g1 - g0 <= max_pause:
            continue
        if any(p0 < g1 and p1 > g0 for p0, p1 in protect):
            continue
        if quiet is not None and not quiet(g0, g1):
            continue
        first, last = k == 0, k == len(edges) - 1
        c0 = g0 + (0.0 if first else keep_pause / 2)
        c1 = g1 - (0.15 if first else keep_pause / 2)
        if last:                        # keep a short tail after the last word so the reaction breathes
            c0, c1 = g0 + 0.8, g1
        if c1 - c0 > 0.05:
            cuts.append((c0, c1))
    keep, cur = [], start
    for c0, c1 in cuts:
        if c0 > cur:
            keep.append((cur, c0))
        cur = max(cur, c1)
    if end > cur:
        keep.append((cur, end))
    return [(round(a, 3), round(b, 3)) for a, b in keep if b - a > 0.04]


def remap(t: float, keep: list[tuple[float, float]]) -> float | None:
    """Map a source time to the jump-cut timeline (None if it falls in a removed gap)."""
    acc = 0.0
    for a, b in keep:
        if t < a:
            return None
        if t <= b:
            return acc + (t - a)
        acc += b - a
    return None


def segments_in(segs: list[dict], start: float, end: float) -> list[dict]:
    return [s for s in segs if s["end"] > start and s["start"] < end]


def words_in(segs: list[dict], start: float, end: float) -> list[dict]:
    return [w for s in segments_in(segs, start, end) for w in s.get("words", []) if w["end"] > start and w["start"] < end]
