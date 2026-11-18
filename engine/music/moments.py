"""Aim the song at the big moments: a crash, an explosion, a scream, a reveal - somewhere nobody is saying anything the
viewer has to hear. The song's drop lands exactly on that moment.

Two ways to choose the spot, both on the clip's OUTPUT timeline:
  AI judgment  - the AI sees the timed transcript + the loud events and picks the best spot (any time, not only events)
  rules        - fallback: the loudest event whose surrounding seconds hold the least meaningful speech
Yelling does not count as "talking": shouts and exclamations ("NO WAY", "AHHH") are fine to cover with music.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import numpy as np

# exclamations: the viewer doesn't need to hear these clearly, so music may cover them
FILLER = {"ah", "aah", "ahh", "aaah", "oh", "ohh", "no", "nooo", "wait", "what", "yo", "bro", "hey", "wow", "whoa", "woah", "ooh",
          "oof", "yes", "yeah", "yep", "god", "damn", "dude", "man", "omg", "ha", "haha", "uh", "um", "huh", "hm", "hmm", "my",
          "oh,", "go", "come", "on", "let's", "lets", "crazy", "insane", "bruh", "ay", "ayy"}


def _norm(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())


def meaningful(words: list[dict]) -> list[tuple[float, float]]:
    """Time spans of words the viewer should actually hear (everything except shouts / exclamations)."""
    return [(w["start"], w["end"]) for w in words if _norm(w["word"]) and _norm(w["word"]) not in FILLER]


def talk_share(spans: list[tuple[float, float]], a: float, b: float) -> float:
    if b <= a:
        return 1.0
    return min(1.0, sum(max(0.0, min(y, b) - max(x, a)) for x, y in spans) / (b - a))


def loud_events(features_path, start: float, end: float, remap: Callable, keep_abs, top: int = 8) -> list[dict]:
    """Strongest sound events inside the clip (explosions, crashes, screams), on the output timeline."""
    from engine.audio import analysis as audio_an

    _, ex = audio_an.load(features_path)
    t, score, voiced = ex["t"], ex["score"], ex["voiced"]
    sel = np.flatnonzero((t >= start) & (t <= end) & (score > 0.4))
    cands = sorted(((float(score[i]), i) for i in sel), reverse=True)
    out: list[dict] = []
    for sc, i in cands:
        o = remap(float(t[i]), keep_abs)
        if o is None or any(abs(o - e["t"]) < 1.5 for e in out):
            continue
        out.append({"t": round(o, 2), "score": round(min(1.0, sc), 2), "voiced": bool(voiced[i] >= 0.5)})
        if len(out) >= top:
            break
    return sorted(out, key=lambda e: e["t"])


def rules_pick(events: list[dict], words_out: list[dict], conflicts: list[tuple[float, float]], out_dur: float,
               lead_in: float = 2.0, tail: float = 5.0) -> Optional[dict]:
    spans = meaningful(words_out)
    best = None
    for e in events:
        t = e["t"]
        if t < lead_in or out_dur - t < 3.0:                       # needs room before (build) and after (the drop plays)
            continue
        if any(c0 < t < c1 for c0, c1 in conflicts):               # never on top of the game's own music
            continue
        talk = talk_share(spans, t - 0.4, min(out_dur, t + tail))
        if talk > 0.45:
            continue
        val = e["score"] * (1.0 - talk) * (0.85 if e["voiced"] and talk > 0.2 else 1.0)
        if best is None or val > best[0]:
            best = (val, e, talk)
    if not best:
        return None
    _, e, talk = best
    return {"drop": max(0.5, e["t"] - 0.15), "reason": f"loud moment at {e['t']:.1f}s with little to hear around it "
            f"({talk:.0%} speech)", "by": "rules"}


def ai_pick(provider, category: str, words_out: list[dict], events: list[dict], conflicts, out_dur: float,
            log: Callable[[str], None] = lambda m: None) -> Optional[dict]:
    lines, cur, t0 = [], [], 0.0
    for w in words_out:
        if not cur:
            t0 = w["start"]
        cur.append(w["word"])
        if len(cur) >= 8 or w["word"].rstrip().endswith((".", "?", "!")):
            lines.append(f"[{t0:.1f}s] {' '.join(cur)}")
            cur = []
    if cur:
        lines.append(f"[{t0:.1f}s] {' '.join(cur)}")
    ev = ", ".join(f"{e['t']:.1f}s (loudness {e['score']:.2f}, {'voice' if e['voiced'] else 'sound effect'})" for e in events) or "none detected"
    prompt = (
        f"A {out_dur:.0f}-second vertical clip (type: {category}).\nTimed transcript:\n" + ("\n".join(lines) or "(no speech)") +
        f"\n\nLoud events: {ev}\nGame/source music at: {', '.join(f'{a:.1f}-{b:.1f}s' for a, b in conflicts) or 'none'}\n\n"
        "Pick the ONE best second to land a song's big drop. It should be a big moment on screen or in the audio "
        "(a crash, explosion, scream, shocking reveal, big action) where no one is saying something the viewer must hear. "
        "Shouting or exclamations may be covered by music; real sentences, jokes and punchlines may not. "
        f"Use your judgment: it can be any time between 2 and {out_dur - 3:.0f} seconds, not only a listed event. "
        'Reply as JSON: {"drop_at": <seconds>, "reason": "<one short line>", "confidence": <0-1>}')
    try:
        r = provider.complete_json("You are an expert short-form video editor who times music to the action.", prompt)
        drop = float(r.get("drop_at"))
        conf = float(r.get("confidence", 0.6))
    except Exception as e:
        log(f"AI music spot failed: {e}")
        return None
    if not (2.0 <= drop <= out_dur - 3.0) or conf < 0.3:
        return None
    if any(c0 < drop < c1 for c0, c1 in conflicts):
        return None
    if talk_share(meaningful(words_out), drop - 0.4, min(out_dur, drop + 5.0)) > 0.6:     # the AI misjudged: don't cover speech
        return None
    return {"drop": drop, "reason": str(r.get("reason", ""))[:160], "by": "ai"}


def pick_spot(provider, category: str, words_out: list[dict], events: list[dict], conflicts, out_dur: float,
              use_ai: bool = True, log: Callable[[str], None] = lambda m: None) -> Optional[dict]:
    if use_ai and provider is not None:
        spot = ai_pick(provider, category, words_out, events, conflicts, out_dur, log)
        if spot:
            return spot
    return rules_pick(events, words_out, conflicts, out_dur)
