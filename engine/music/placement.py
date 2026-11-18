"""Placement intelligence (M6): where in the clip the song plays and which part of it.

All times here are on the OUTPUT timeline (after jump cuts), so a downbeat lands
on the payoff as the viewer hears it.

Modes (tried in a mode-preference order, first clean result wins):
  payoff_drop   - the hook's downbeat lands exactly on the payoff, 1-bar soft pre-roll
  reaction_tail - song fades in under the end of the reaction and takes the last 3-8 s
  underscore    - low bed under the whole clip, ducked under speech
  cold_open     - hook for the first 1.5-2 s under the text hook, then ducks out
Never inside a source-music region; entries/exits on bar boundaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Placement:
    mode: str
    song_in: float           # seconds into the song
    song_out: float
    clip_in: float           # seconds into the (cut) clip where the song starts
    clip_out: float
    fade_in: float
    fade_out: float
    payoff_sync: Optional[dict] = None
    uses_stem: str = "full_mix"
    window_quality: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _free(a: float, b: float, conflicts: list[tuple[float, float]]) -> bool:
    return all(b <= c0 or a >= c1 for c0, c1 in conflicts)


def _bar_at_or_before(bars: np.ndarray, t: float) -> float:
    c = bars[bars <= t + 1e-6]
    return float(c[-1]) if c.size else float(bars[0])


ACTION = ("crazy_gameplay", "dramatic", "surprising_event", "fail")
TALK_RATIO = 0.5          # share of the clip that is speech above which it counts as a "just talking" clip


def talk_ratio(speech: list[tuple[float, float]], clip_dur: float) -> float:
    return min(1.0, sum(max(0.0, b - a) for a, b in speech) / max(1e-6, clip_dur))


def preferred_modes(category: str, speech_rate: float, energy: float, talking: float = 0.0) -> list[str]:
    # style first: a clip that is mostly talking gets the song UNDER the voice (ducked bed);
    # an action scene gets the song OVER it (the hook lands on the payoff)
    if talking >= TALK_RATIO and not (category in ACTION and energy >= 0.8):
        return ["underscore", "reaction_tail"]
    if category in ACTION and talking < TALK_RATIO:
        return ["payoff_drop", "reaction_tail"]
    # the moment type decides first; speech rate only breaks ties (streamers talk ~2.5 w/s anyway)
    if category in ("funny", "memorable_dialogue", "controversial_opinion", "unexpected_interaction"):
        return ["reaction_tail", "payoff_drop"]          # music during the setup would kill the joke
    if category in ("crazy_gameplay", "dramatic", "surprising_event", "fail", "emotional_reaction") or \
            (energy >= 0.7 and speech_rate <= 3.2):
        return ["payoff_drop", "reaction_tail"]
    if speech_rate > 3.2:
        return ["reaction_tail", "payoff_drop"]
    if speech_rate < 1.0 and category in ("unexpected_discovery", "interesting_observation", "first_reaction"):
        return ["underscore", "reaction_tail", "payoff_drop"]
    return ["reaction_tail", "payoff_drop"]


def place(song: dict, clip_dur: float, payoff: float, conflicts: list[tuple[float, float]],
          speech: list[tuple[float, float]], modes: list[str], min_window: float = 6.0) -> Optional[Placement]:
    """song: effective analysis dict (bars, bar_s, hook{downbeat,start,end}, duration)."""
    bars = np.array(song["bars"])
    bar = float(song["bar_s"])
    hook_db = float(song["hook"]["downbeat"])
    sdur = float(song["duration"])
    if song["hook"].get("user") and song["hook"].get("end"):      # the user marked the part: never play past it
        sdur = min(sdur, float(song["hook"]["end"]))
        min_window = min(min_window, max(2.0, sdur - hook_db))
    for mode in modes:
        p = _try(mode, bars, bar, hook_db, sdur, clip_dur, payoff, conflicts, speech, min_window)
        if p:
            return p
    return None


def auto_entry(payoff: float, speech: list[tuple[float, float]], conflicts: list[tuple[float, float]],
               bar: float, max_lead: float = 32.0) -> Optional[float]:
    """Trailer-style entry point: right after the last line of setup dialogue, so the song takes
    over the action and builds into the payoff. Never inside/before source music that sits between
    the entry and the payoff. Returns clip time, or None."""
    min_lead = 2 * bar
    # ends of spoken lines (a pause >= 1.2 s after the word)
    ends = []
    for k, (a, b) in enumerate(speech):
        nxt = speech[k + 1][0] if k + 1 < len(speech) else 1e9
        if nxt - b >= 1.2:
            ends.append(b + 0.25)
    ends += [c1 + 0.3 for c0, c1 in conflicts]
    cands = sorted(e for e in ends if payoff - max_lead <= e <= payoff - min_lead)

    def ok(e: float, max_talk: float) -> bool:
        if any(c0 < payoff and c1 > e for c0, c1 in conflicts):
            return False
        talk = sum(max(0.0, min(b, payoff) - max(a, e)) for a, b in speech)
        return talk / max(1e-6, payoff - e) <= max_talk
    # Prefer a quiet entry after the last setup line. If that leaves less than ~12 s of build-up,
    # let the song start earlier UNDER the dialogue (it ducks under every word, like a trailer).
    best = next((e for e in cands if ok(e, 0.35)), None)
    if best is None or payoff - best < 12.0:
        early = next((e for e in cands if ok(e, 0.85) and payoff - e >= 12.0), None)
        if early is not None:
            return early
    if best is not None:
        return best
    e = payoff - 4 * bar
    return e if e >= 0 and ok(e, 0.85) else None


def place_custom(song: dict, clip_dur: float, enter: float, drop: float,
                 conflicts: list[tuple[float, float]], prefer_text: str = "", min_window: float = 6.0,
                 prefer_hook: bool = False) -> Optional[Placement]:
    """User-directed placement: the song comes in at `enter` (clip time) and a chorus downbeat
    lands exactly on `drop`. Prefers the chorus where the title line is actually sung (from the
    timed lyrics), then the earliest one with enough song before it for the lead-in. The lead-in
    is a whole number of bars (never earlier than `enter`) so the entry is on a bar too."""
    import math

    bars = np.array(song["bars"])
    bar = float(song["bar_s"])
    sdur = float(song["duration"])
    user_hook = song["hook"] if song.get("hook", {}).get("user") and song["hook"].get("end") else None
    lyrics = song.get("lyrics", [])
    key = prefer_text.lower().strip().rstrip("s")[:6]

    # candidate hit points: bar nearest each title-line lyric, then labelled choruses
    hits: list[float] = []
    if key:
        for l in lyrics:
            if key in l["text"].lower():
                hits.append(float(bars[np.argmin(np.abs(bars - l["start"]))]))
    for s in song.get("sections", []):
        if s.get("label") == "chorus" or s.get("is_hook"):
            hits.append(float(bars[np.argmin(np.abs(bars - s["start"]))]))
    if prefer_hook and song.get("hook", {}).get("downbeat") is not None:
        hits.insert(0, float(song["hook"]["downbeat"]))     # same hook every clip: viewers learn the sound
    seen: list[float] = []
    for h in hits:  # keep order: lyric-confirmed first
        if all(abs(h - x) > 0.05 for x in seen):
            seen.append(h)
    # song first: if the build-up is a hair short of one more whole bar, take that bar and start the song a moment into
    # it, so music plays from the very first frame instead of after a near-empty bar of silence
    slack = 0.25 if prefer_hook else 1e-6
    lead = max(0, math.floor((drop - enter) / bar + slack)) * bar
    for db in seen:
        song_in = db - lead
        if song_in < -1e-3:
            continue
        clip_in = drop - lead
        if clip_in < enter:                            # trim the song's start so it begins exactly at `enter`
            song_in += enter - clip_in
            clip_in = enter
        limit = min(sdur, float(user_hook["end"])) if user_hook and user_hook["start"] - 0.7 <= db <= user_hook["end"] else sdur
        song_out = _bar_at_or_before(bars, min(limit, song_in + (clip_dur - clip_in)))
        clip_out = clip_in + (song_out - song_in)
        mw = min(min_window, max(2.0, limit - db)) if user_hook and limit < sdur else min_window
        if song_out - song_in < max(2 * bar, mw) or clip_out - drop < mw or                 not _free(clip_in, clip_out, conflicts):
            continue                                   # skip placements where the hook gets only a few seconds
        return Placement("payoff_drop", round(song_in, 4), round(song_out, 4), round(clip_in, 4), round(clip_out, 4),
                         fade_in=round(min(bar, 1.5), 3), fade_out=round(min(bar, 2.0), 3),
                         payoff_sync={"clip_time": round(drop, 4), "song_downbeat": round(db, 4)},
                         window_quality=1.0, notes=[f"user-directed: enters {enter:.1f}s, chorus at {drop:.1f}s"])
    return None


def _try(mode, bars, bar, hook_db, sdur, clip_dur, payoff, conflicts, speech, min_window) -> Optional[Placement]:
    if mode == "payoff_drop":
        clip_in = payoff - bar                        # one-bar soft pre-roll
        if clip_in < 0.5 or clip_dur - payoff < min_window:
            return None
        if any(c0 < payoff < c1 or c0 < clip_in < c1 for c0, c1 in conflicts):
            return None
        song_in = hook_db - bar
        if song_in < 0:
            return None
        # play until the clip ends OR the next source-music cue starts (song bows out before it)
        nxt = min((c0 for c0, c1 in conflicts if c0 > payoff), default=clip_dur)
        avail = min(clip_dur, nxt - 0.3) - clip_in
        # end on a bar boundary at/before the clip end (and inside the song)
        song_out = _bar_at_or_before(bars, min(sdur, song_in + avail))
        if song_out - song_in < max(min_window, bar * 2):
            return None
        clip_out = clip_in + (song_out - song_in)
        if not _free(clip_in, clip_out, conflicts) or clip_out - payoff < min_window:
            return None                                # the hook itself must get a usable stretch after the drop
        q = min(1.0, (clip_out - payoff) / 12.0)
        return Placement("payoff_drop", round(song_in, 4), round(song_out, 4), round(clip_in, 4), round(clip_out, 4),
                         fade_in=round(bar * 0.9, 3), fade_out=round(min(bar, 2.0), 3),
                         payoff_sync={"clip_time": round(payoff, 4), "song_downbeat": round(hook_db, 4)},
                         window_quality=round(0.6 + 0.4 * q, 3))
    if mode == "reaction_tail":
        last_speech = max((b for a, b in speech if a < clip_dur), default=0.0)
        conflict_end = max((c1 for c0, c1 in conflicts if c0 < clip_dur), default=0.0)
        start = max(last_speech - 1.0, conflict_end + 0.2, payoff + 0.5, clip_dur - 8.0)
        if clip_dur - start < min_window:
            start = max(conflict_end + 0.2, payoff + 0.5, clip_dur - min_window - 0.5)
        avail = clip_dur - start
        if avail < min_window or not _free(start, clip_dur, conflicts):
            return None
        song_in = hook_db                              # the tail plays the hook itself
        song_out = _bar_at_or_before(bars, min(sdur, song_in + avail))
        if song_out - song_in < min_window:
            song_out = min(sdur, song_in + avail)      # bar snapping cost too much: use the full window
        if song_out - song_in < min_window:
            return None                                # a few seconds of song is never usable
        clip_out = start + (song_out - song_in)
        over_speech = sum(max(0.0, min(b, clip_out) - max(a, start)) for a, b in speech)
        q = max(0.0, 1.0 - over_speech / max(1e-6, clip_out - start)) * min(1.0, avail / 6.0)
        return Placement("reaction_tail", round(song_in, 4), round(song_out, 4), round(start, 4), round(clip_out, 4),
                         fade_in=round(min(1.2, avail / 4), 3), fade_out=round(min(bar, 1.5), 3),
                         window_quality=round(q, 3))
    if mode == "underscore":
        if conflicts:
            return None
        song_in = _bar_at_or_before(bars, max(0.0, hook_db - 4 * bar))
        song_out = _bar_at_or_before(bars, min(sdur, song_in + clip_dur))
        if song_out - song_in < min_window:
            return None
        return Placement("underscore", round(song_in, 4), round(song_out, 4), 0.0, round(song_out - song_in, 4),
                         fade_in=0.6, fade_out=round(min(bar, 2.0), 3), window_quality=0.7)
    if mode == "cold_open":
        if not _free(0.0, 2.5, conflicts):
            return None
        return Placement("cold_open", round(hook_db, 4), round(hook_db + 2.0, 4), 0.0, 2.0,
                         fade_in=0.02, fade_out=0.6, window_quality=0.5)
    return None
