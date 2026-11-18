"""Viral moment detection (PRD §7) - "the brain".

Pipeline
1. propose():  fuse audio excitement (A), chat activity (B) and transcript cues into one
               curve, pick well-separated peaks. Proposals only; loud != interesting.
2. evaluate(): an AI model (Claude or local Qwen) reads the transcript around each
               proposal and scores it against a viral rubric (hook, payoff, surprise,
               clarity, shareability...). It also places setup/payoff/end so the clip
               gets to the point fast. Without a model, a rule-based evaluator scores
               the same rubric from signals and is labelled as such.
3. rank():     editor's final cut - the model compares the surviving moments side by
               side and re-orders them (absolute scores from one-at-a-time evaluation
               drift; comparisons are more reliable).
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from engine.audio.analysis import HOP_S, scream_or_laugh
from engine.clipping.boundaries import keep_intervals, segments_in, snap, words_in
from engine import profiles
from engine.config import Settings

SURPRISE = re.compile(r"\b(what+|wait|no way|oh my (god|gosh)|omg|holy|wtf|what the|are you (kidding|serious)|"
                      r"how (did|is)|bro+|dude|yo+|insane|crazy|unbelievable|no no|nah|oh no|look at (this|that)|"
                      r"did you see|i can'?t believe)\b", re.I)
HYPE = re.compile(r"\b(let'?s go+|yes+|we did it|clutch|get in|we (lost|got) (them|him|it)|perfect)\b", re.I)
LAUGH = re.compile(r"(\bha(ha)+\b|\blol\b|\blmao\b|\(laugh|\[laugh|\blaughing\b)", re.I)
CALLOUT = re.compile(r"\b(clip (that|it|this)|chat,? (look|did you|what)|did you guys see|somebody clip)\b", re.I)
QUESTION_HOOK = re.compile(r"^(wait|what|is that|did|how|why|no way|oh)\b", re.I)

CATEGORIES = ["first_reaction", "unexpected_discovery", "funny", "surprising_event", "crazy_gameplay",
              "interesting_observation", "emotional_reaction", "controversial_opinion", "unexpected_interaction",
              "story_development", "memorable_dialogue", "fail", "dramatic"]
RUBRIC = ("hook_strength", "entertainment", "payoff", "emotional_intensity", "unexpectedness",
          "contextual_clarity", "shareability")


def lexical_curve(units: list[dict], n: int) -> np.ndarray:
    curve = np.zeros(n)
    for s in units:
        t = s["text"]
        hits = (len(SURPRISE.findall(t)) + len(HYPE.findall(t)) + 1.5 * len(LAUGH.findall(t))
                + 2.0 * len(CALLOUT.findall(t)) + 0.5 * t.count("!"))
        if hits:
            i0, i1 = int(s["start"] / HOP_S), int(s["end"] / HOP_S) + 1
            curve[max(0, i0):min(n, i1)] += min(1.0, hits / 3)
    return curve


def propose(ex: dict, chat: Optional[dict], units: list[dict], settings: Settings, duration: float) -> list[dict]:
    audio = ex["score"]
    n = audio.size
    if n == 0:
        return []
    lex = lexical_curve(units, n)
    if chat is not None:
        ch = np.interp(ex["t"], chat["t"], chat["score"])
        fused = 0.50 * audio + 0.30 * ch + 0.20 * lex
    else:
        ch = None
        fused = 0.72 * audio + 0.28 * lex
    fused = uniform_filter1d(fused, size=max(1, int(3 / HOP_S)))
    cfg = settings.clips
    pct = 99.0 - 8.0 * cfg.sensitivity                    # sensitivity 0.5 -> 95th percentile
    thresh = max(float(np.percentile(fused, pct)), 0.08)
    peaks, props = find_peaks(fused, height=thresh, distance=max(1, int(30 / HOP_S)))
    order = np.argsort(props["peak_heights"])[::-1][: cfg.candidate_pool]
    out = []
    for idx in sorted(peaks[order]):
        t = float(ex["t"][idx])
        w = snap(units, t - cfg.max_setup_s, t + 8, (t, t), cfg.min_duration, cfg.target_max, duration)
        out.append({"peak": t, "start": w.start, "end": w.end, "starts_mid_sentence": w.starts_mid_sentence,
                    "fused": float(fused[idx])})
    jp = settings.video.max_pause if settings.video.jump_cuts else None
    for c in out:
        hook_first_start(c, units, cfg, duration)
        c["signals"] = signals_for(c["start"], c["end"], c["peak"], ex, ch, units, jp)
    return out


def hook_first_start(c: dict, units: list[dict], cfg, duration: float) -> None:
    """Start on the first line that hooks (a startled line/question/hype) before the peak,
    instead of on calm setup talk. The AI evaluator can override this with its own setup point."""
    for u in units:
        if u["start"] < c["start"] + 1.5 or u["start"] > c["peak"]:
            continue
        if SURPRISE.search(u["text"]) or QUESTION_HOOK.search(u["text"]) or HYPE.search(u["text"]):
            new_start = max(0.0, u["start"] - 0.2)
            new_end = c["end"]
            if new_end - new_start < cfg.min_duration:   # keep minimum length by extending the tail
                new_end = min(duration, new_start + cfg.min_duration)
            for v in units:                              # don't cut the tail mid-sentence
                if v["start"] < new_end < v["end"]:
                    new_end = min(duration, v["end"] + 0.4)
            c["start"], c["end"], c["starts_mid_sentence"] = new_start, new_end, False
            return


def signals_for(start: float, end: float, peak: float, ex: dict, chat_curve: Optional[np.ndarray],
                units: list[dict], jump_pause: Optional[float] = None) -> dict:
    i0, i1 = int(start / HOP_S), max(int(start / HOP_S) + 1, int(end / HOP_S))
    a = ex["score"][i0:i1]
    voiced = ex["voiced"][i0:i1] if "voiced" in ex else np.ones_like(a)
    words = words_in(units, start, end)
    text = " ".join(w["word"] for w in words)
    toks = [t.lower() for t in re.findall(r"[a-z']+", text, re.I)]
    rep = 0.0
    if len(toks) >= 12:
        grams = [" ".join(toks[k:k + 3]) for k in range(len(toks) - 2)]
        rep = 1 - len(set(grams)) / len(grams)
    # dead air: non-voiced AND low-excitement (an explosion is not dead air). When jump cuts are on,
    # only count what survives the cut plan, because pauses between words get removed.
    boring = (voiced < 0.5) & (a < 0.25)
    if jump_pause and boring.size:
        keep = keep_intervals(words, start, end, max_pause=jump_pause)
        tt = start + np.arange(boring.size) * HOP_S
        kept = np.zeros(boring.size, bool)
        for k0, k1 in keep:
            kept |= (tt >= k0) & (tt < k1)
        boring = boring[kept] if kept.any() else boring
    # hook window: first 2.5 s of the clip
    h1 = min(i1, i0 + int(2.5 / HOP_S))
    hook_words = " ".join(w["word"] for w in words if w["start"] < start + 2.5)
    pre0 = max(0, i0 - int(30 / HOP_S))
    contrast = float(a.max() - ex["score"][pre0:i0].mean()) if i0 > pre0 and a.size else 0.0
    # laughter cue only counts where no words were transcribed: speech itself has a 4-5 Hz
    # syllable rhythm in the same band, so voiced speech must not read as laughter
    sl = scream_or_laugh(ex, i0, i1)
    if "laugh" in ex and i1 > i0:
        tt = np.arange(i0, i1) * HOP_S
        wordy = np.zeros(i1 - i0, bool)
        for w in words:
            wordy |= (tt >= w["start"] - 0.2) & (tt <= w["end"] + 0.2)
        lg = ex["laugh"][i0:i1][: wordy.size]
        sl["laughter_like"] = float(np.mean((lg > 0.8) & ~wordy[: lg.size])) if lg.size else 0.0
    return {
        "audio_peak": round(float(a.max()) if a.size else 0.0, 3),
        "audio_mean": round(float(np.mean(np.sort(a)[-max(1, a.size // 4):])) if a.size else 0.0, 3),
        "hook_audio": round(float(ex["score"][i0:h1].max()) if h1 > i0 else 0.0, 3),
        "hook_text": hook_words[:120],
        "hook_question": bool(QUESTION_HOOK.search(hook_words.strip())),
        "contrast": round(contrast, 3),
        "chat_peak": None if chat_curve is None else round(float(chat_curve[i0:i1].max()) if i1 > i0 else 0.0, 3),
        "silence_ratio": round(float(boring.mean()) if boring.size else 1.0, 3),
        "raw_silence_ratio": round(1 - float(voiced.mean()) if voiced.size else 1.0, 3),
        "words": len(words),
        "speech_rate_wps": round(len(words) / max(1e-6, end - start), 2),
        "surprise_hits": len(SURPRISE.findall(text)),
        "hype_hits": len(HYPE.findall(text)),
        "laugh_hits": len(LAUGH.findall(text)),
        "callout_hits": len(CALLOUT.findall(text)),
        "repetition": round(rep, 3),
        **{k: round(v, 3) for k, v in sl.items()},
        "z_rise_max": round(float(ex["z_rise"][i0:i1].max()) if "z_rise" in ex and i1 > i0 else 0.0, 2),
        "peak_offset_s": round(peak - start, 1),
    }


# ---------------------------------------------------------------- evaluation
SYSTEM = """You are the head editor of a {vertical} clip channel that regularly hits millions of views on
TikTok, Reels and YouTube Shorts. You pick moments from videos. You are ruthless: most candidates
are NOT worth posting. Loud does not mean interesting.

VIDEO TYPE: {label}. {guidance}

What goes viral in short-form gaming clips:
- HOOK: the first 1-2 seconds stop the scroll (a startled line, a question, a visible weird thing,
  the streamer already mid-reaction). A slow intro kills the clip.
- PAYOFF: a clear, satisfying moment - a reveal, an escape, a fail, a punchline - with a genuine reaction.
- SURPRISE/CURIOSITY: something the viewer did not expect, or "wait for it" tension that resolves.
- STANDALONE: makes sense to someone who has never seen this streamer or game.
- SHAREABLE: quotable line, relatable feeling, or "you have to see this" energy.
- TIGHT: setup is short; the clip ends right after the reaction lands.
Reject: filler, long pauses, uninteresting gameplay, incomplete reactions, repetitive talk,
no payoff, needs prior context, or depends on visuals the transcript can't confirm.
Answer with a single JSON object only."""

PROMPT = """Source: {streamer}
Video title: {title}
Video type: {label}
{focus}
Candidate moment detected by audio/chat signals around t={peak:.1f}s.
Signal summary: {signals}
{evidence_note}
Transcript with absolute timestamps (seconds). Lines marked >> are inside the current window [{start:.1f}, {end:.1f}]:
{transcript}

Score each 0-100 (be harsh; a typical stream moment is 30-55, a post-worthy one 70+, a banger 85+):
- hook_strength: how hard the clip's first 2 seconds grab attention, given where you start it
- entertainment, payoff, emotional_intensity, unexpectedness, contextual_clarity, shareability

Return JSON:
{{
 "keep": true|false,
 "reject_reason": "short reason if keep is false, else empty",
 "category": one of {categories},
 "summary": "one factual sentence: what happens",
 "setup_start": seconds where the clip should START (as late as possible while still understandable; ideally on a line that hooks),
 "payoff_time": seconds where the payoff/peak reaction happens,
 "end_time": seconds where the reaction finishes (end soon after - don't let it drag),
 "hook_strength": 0-100, "entertainment": 0-100, "payoff": 0-100, "emotional_intensity": 0-100,
 "unexpectedness": 0-100, "contextual_clarity": 0-100, "shareability": 0-100,
 "why": "one sentence: why this would or would not perform"
}}
Use only what the transcript supports."""


def _transcript_block(units: list[dict], start: float, end: float, pad: float = 25.0) -> str:
    lines = []
    for s in segments_in(units, start - pad, end + pad):
        mark = ">>" if s["end"] > start and s["start"] < end else "  "
        lines.append(f"{mark} [{s['start']:.1f}-{s['end']:.1f}] {s['text']}")
    return "\n".join(lines) or "(no speech detected)"


def _sat(x: float, k: float) -> float:
    return min(1.0, max(0.0, x) / k)


def _spectacle_eval(c: dict, category: str) -> dict:
    """What is SHOWN and HEARD, not said: action spikes, audio intensity, the cut before the hit."""
    s = c["signals"]
    act = _sat(s["contrast"], 0.5)
    emo = min(1.0, max(s["audio_peak"], s["scream_like"] * 3))
    ent = 0.55 * s["audio_mean"] + 0.45 * act
    payoff = 0.55 * s["audio_peak"] + 0.45 * act
    hook = 0.6 * s["hook_audio"] + 0.25 * act + 0.15 * (1.0 if s["peak_offset_s"] < 10 else 0.3)
    unexp = 0.5 * _sat(s["z_rise_max"], 4) + 0.5 * act
    clarity = 0.65 + 0.35 * (1 - s["silence_ratio"])
    share = 0.45 * ent + 0.35 * emo + 0.2 * act
    return {
        "category": category, "summary": "Rule-based selection on footage (action and audio intensity).",
        "why": f"action spike (contrast {s['contrast']:.2f}), audio peak {s['audio_peak']:.2f}",
        "hook_strength": hook, "entertainment": ent, "payoff": payoff, "emotional_intensity": emo,
        "contextual_clarity": clarity, "unexpectedness": unexp, "shareability": share,
    }


def _speech_eval(c: dict) -> dict:
    """What is SAID: surprise words, laughter, 'clip that', clear sentences."""
    s = c["signals"]
    lex = _sat(s["surprise_hits"] + s["hype_hits"] + 1.5 * s["laugh_hits"] + 2 * s["callout_hits"], 4)
    emo = min(1.0, max(s["audio_peak"], s["laughter_like"] * 2, s["scream_like"] * 3))
    clarity = (1 - s["silence_ratio"]) * _sat(s["words"], 30) * (0.6 if c.get("starts_mid_sentence") else 1.0)
    hook = 0.45 * s["hook_audio"] + 0.35 * (1.0 if s["hook_question"] else _sat(len(SURPRISE.findall(s["hook_text"])), 1)) \
        + 0.20 * (1.0 if s["peak_offset_s"] < 8 else 0.4)
    payoff = 0.4 * s["audio_peak"] + 0.3 * _sat(s["contrast"], 0.5) + 0.3 * _sat(s["hype_hits"] + s["laugh_hits"] + s["callout_hits"], 2)
    ent = 0.40 * s["audio_mean"] + 0.25 * min(1.0, s["laughter_like"] * 3 + s["laugh_hits"] / 3) + 0.35 * lex
    unexp = 0.55 * _sat(s["surprise_hits"], 3) + 0.25 * _sat(s["z_rise_max"], 4) + 0.20 * _sat(s["contrast"], 0.5)
    share = 0.35 * ent + 0.25 * emo + 0.20 * clarity + 0.20 * _sat(s["callout_hits"] + s["surprise_hits"], 3)
    cat = "funny" if s["laugh_hits"] or s["laughter_like"] > 0.35 else (
        "surprising_event" if s["surprise_hits"] >= 2 else ("crazy_gameplay" if s["hype_hits"] else "emotional_reaction"))
    return {
        "category": cat, "summary": "Rule-based selection (no language model evaluated the context).",
        "why": "signal-based: " + ", ".join(k for k, v in (("strong audio peak", s["audio_peak"] > 0.6),
                                                          ("surprise words", s["surprise_hits"] > 0),
                                                          ("hype/laughter", s["hype_hits"] + s["laugh_hits"] > 0),
                                                          ("said 'clip that'", s["callout_hits"] > 0),
                                                          ("sharp contrast with prior 30 s", s["contrast"] > 0.3)) if v),
        "hook_strength": hook, "entertainment": ent, "payoff": payoff, "emotional_intensity": emo,
        "contextual_clarity": clarity, "unexpectedness": unexp, "shareability": share,
    }


def heuristic_eval(c: dict, profile: "profiles.Profile | str | None" = None) -> dict:
    """Rule-based rubric. Weaker than a model (it can't truly read context), but grounded in the
    signals a human editor reacts to. WHICH signals count depends on the video type: a streamer's words
    for a stream or podcast, the picture and sound for trailers and footage, both for movies and sports."""
    prof = profile if isinstance(profile, profiles.Profile) else profiles.get(profile)
    if prof.evidence == "spectacle":
        parts = [_spectacle_eval(c, prof.default_category)]
    elif prof.evidence == "mixed":
        parts = [_speech_eval(c), _spectacle_eval(c, prof.default_category)]
    else:
        parts = [_speech_eval(c)]
    out = {k: sum(p[k] for p in parts) / len(parts) for k in RUBRIC}
    first = parts[-1] if prof.evidence == "mixed" else parts[0]
    s = c["signals"]
    return {
        "mode": "heuristic", "keep": True, "reject_reason": "", "category": first["category"],
        "summary": first["summary"], "why": first["why"], "profile": prof.key,
        "setup_start": c["start"], "payoff_time": c["peak"], "end_time": c["end"],
        **{k: round(100 * v) for k, v in out.items()},
    }


def _num(v, lo, hi, default):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, x))


def evaluate(cands: list[dict], units: list[dict], ctx: dict, settings: Settings, provider,
             log: Callable[[str], None] = print) -> list[dict]:
    cfg = settings.clips
    focus = (f"The user's own instructions for choosing and cutting clips (follow them when judging, scoring and picking the setup "
             f"and payoff points; they outrank your defaults): \"{ctx['focus']}\"") if ctx.get("focus") else ""
    prof = ctx.get("profile") or profiles.get("stream")
    system = SYSTEM.format(vertical=settings.vertical, label=prof.label, guidance=prof.guidance)
    if ctx.get("music_brief"):
        system += "\n\n" + ctx["music_brief"]
    evidence_note = ("" if prof.evidence == "speech" else
                     "NOTE: the transcript only has spoken words. In this video type the picture and sound carry the moment, "
                     "so rely on the signal summary (audio peak, contrast with what came before) as well.\n")
    def ask(c: dict):
        if provider is None:
            return None, None
        prompt = PROMPT.format(streamer=ctx.get("streamer") or "unknown", title=ctx.get("title") or "",
                               label=prof.label, evidence_note=evidence_note, focus=focus, peak=c["peak"], start=c["start"], end=c["end"],
                               signals={k: v for k, v in c["signals"].items() if v is not None},
                               transcript=_transcript_block(units, c["start"], c["end"]), categories=CATEGORIES)
        try:
            ai = provider.complete_json(system, prompt)
            ai["mode"] = f"{provider.name}:{provider.model}"
            return ai, None
        except Exception as e:  # model failure must not kill the batch
            return None, e

    if provider is not None and len(cands) > 1:        # one request per moment: run several at once
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as pool:
            answers = list(pool.map(ask, cands))
    else:
        answers = [ask(c) for c in cands]
    for c, (ai, err) in zip(cands, answers):
        if err is not None:
            log(f"AI evaluation failed for t={c['peak']:.0f}s ({err}); using rule-based evaluation.")
        if ai is None:
            ai = heuristic_eval(c, prof)
            if provider is not None:
                ai["mode"] = "heuristic (AI call failed)"
        for k in RUBRIC:
            ai[k] = _num(ai.get(k), 0, 100, 0)
        lo, hi = c["start"] - 30, c["end"] + 30
        ai["setup_start"] = _num(ai.get("setup_start"), lo, hi, c["start"])
        ai["payoff_time"] = _num(ai.get("payoff_time"), lo, hi, c["peak"])
        ai["end_time"] = _num(ai.get("end_time"), lo, hi, c["end"])
        if ai.get("category") not in CATEGORIES:
            ai["category"] = "interesting_observation"
        c["ai"] = ai
        if ai.get("keep", True) and not str(ai.get("mode", "")).startswith("heuristic"):
            setup = max(ai["setup_start"], ai["payoff_time"] - cfg.max_setup_s)
            w = snap(units, setup, max(ai["end_time"], ai["payoff_time"] + 2),
                     (ai["payoff_time"], ai["payoff_time"]), cfg.min_duration, cfg.target_max, ctx["duration"])
            c["start"], c["end"], c["starts_mid_sentence"] = w.start, w.end, w.starts_mid_sentence
        c["payoff"] = ai["payoff_time"]
        c["category"] = ai["category"]
        c["summary"] = ai.get("summary", "")
    return cands


RANK_PROMPT = """You are making the final cut for today's {vertical} posts (video type: {label}). Below are moments that each
passed individual review. Rank them from most to least likely to perform on TikTok/Shorts, comparing
them against each other (hook, payoff, surprise, standalone clarity, shareability).

{items}

Return JSON: {{"order": [ids best first], "notes": {{"<id>": "one short reason"}}}}"""


_NEG = ("skip", "avoid", "no ", "without", "ignore", "not ", "don't", "dont", "less ", "never", "exclude")
_STOP = {"clips", "clip", "moments", "moment", "that", "this", "with", "from", "have", "make", "only", "more", "very", "when",
         "where", "what", "show", "want", "like", "just", "than", "them", "they", "their", "into", "about", "mostly", "fast",
         "short", "long", "keep", "get", "the", "and", "for", "are", "who", "all", "any", "big", "end", "point",
         "skip", "avoid", "ignore", "without", "never", "exclude", "pick", "find", "choose", "also", "some", "less", "setups", "setup"}
_SYN = {
    "funny": {"funny", "laugh", "laughing", "lol", "haha", "hilarious", "joke", "dying"},
    "laughs": {"laugh", "laughing", "lol", "haha", "hilarious"},
    "action": {"shoot", "shooting", "chase", "explosion", "crash", "fight", "gun", "police", "cops", "wanted"},
    "crashes": {"crash", "crashed", "wreck", "flip", "flipped", "collision"},
    "explosions": {"explosion", "explode", "exploded", "boom", "blow"},
    "yelling": {"yell", "yelling", "scream", "screaming", "shout", "ahh", "aah", "no way"},
    "raging": {"rage", "raging", "mad", "angry", "furious"},
    "rage": {"rage", "raging", "mad", "angry", "furious"},
    "reveals": {"reveal", "twist", "secret", "wait", "what", "plot"},
    "fails": {"fail", "failed", "oops", "dead", "died", "bug", "glitch"},
    "scary": {"scary", "scared", "jumpscare", "horror"},
}


def guidance_terms(text: str) -> tuple[set[str], set[str]]:
    """(wanted words, unwanted words) from the user's free-text instructions. Simple on purpose: it is only the fallback
    for when no AI is available to read the instructions properly."""
    import re

    pos: set[str] = set()
    neg: set[str] = set()
    for clause in re.split(r"[.;\n]+", (text or "").lower()):
        c = clause.strip()
        if not c:
            continue
        bad = any(c.startswith(n.strip()) or f" {n}" in f" {c}" for n in _NEG)
        for w in re.findall(r"[a-z']{4,}", c):
            if w in _STOP:
                continue
            extra: set[str] = set()
            for key, syns in _SYN.items():            # "funniest" / "funny" / "funnier" all mean the "funny" family
                if w.startswith(key[:4]) or key.startswith(w[:4]):
                    extra |= syns
            (neg if bad else pos).update({w} | extra | _SYN.get(w, set()))
    return pos, neg


def guidance_bias(text: str, haystack: str) -> float:
    """Score change (about -24..+18) for a candidate whose transcript/summary is `haystack`."""
    pos, neg = guidance_terms(text)
    h = (haystack or "").lower()
    hits_p = sum(1 for w in pos if w in h)
    hits_n = sum(1 for w in neg if w in h)
    return min(18.0, 6.0 * hits_p) - min(24.0, 12.0 * hits_n)


def _fit_note(c: dict) -> str:
    f = (c.get("ai") or {}).get("music_fit")
    if not f:
        return ""
    return (f" | song fit {f['score']:.0f} (hit {f['hit']:.0f}, clear of talking {f['clear']:.0f}, "
            f"no game music {f['no_music']:.0f})")


def rank(cands: list[dict], provider, settings: Settings, log: Callable[[str], None] = print, profile=None, focus: str = "",
         music_brief: str = "") -> None:
    """Blend a comparative ranking into each candidate's total (in place). Model-only."""
    if provider is None or len(cands) < 2 or not settings.clips.final_ranking:
        return
    items = "\n".join(f"id={i}: [{c['category']}] {c['summary']} | hook: \"{c['signals']['hook_text'][:80]}\" "
                      f"| score {c['total']:.0f}" + _fit_note(c) for i, c in enumerate(cands))
    try:
        prof = profile or profiles.get("stream")
        res = provider.complete_json(SYSTEM.format(vertical=settings.vertical, label=prof.label, guidance=prof.guidance)
                                     + ("\n\n" + music_brief if music_brief else ""),
                                     RANK_PROMPT.format(vertical=settings.vertical, label=prof.label, items=items)
                                     + (f"\nThe user's own instructions (they outrank your defaults): \"{focus}\"" if focus else ""))
        order = [int(x) for x in res.get("order", []) if str(x).lstrip("-").isdigit() and 0 <= int(x) < len(cands)]
    except Exception as e:
        log(f"final ranking skipped ({e})")
        return
    seen = list(dict.fromkeys(order)) + [i for i in range(len(cands)) if i not in order]
    n = len(seen)
    for pos, i in enumerate(seen):
        rank_score = 100 * (1 - pos / max(1, n - 1)) if n > 1 else 50
        c = cands[i]
        c["rank_note"] = (res.get("notes") or {}).get(str(i))
        c["total"] = round(0.8 * c["total"] + 0.2 * rank_score, 1)
