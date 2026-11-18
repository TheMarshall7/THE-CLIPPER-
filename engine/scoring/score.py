"""Viral score v2 (PRD §8, extended).

total = weighted rubric (0-100)  -  penalties  (+ optional comparative-rank blend in moments.rank)

Penalties encode things that measurably hurt short-form retention:
- duration outside the sweet spot (default 18-45 s)
- the peak arrives late (viewer waits > ~10 s for anything to happen)
- starting mid-sentence
Weights are configurable; chat weight is redistributed proportionally when chat is unavailable.
"""
from __future__ import annotations

from typing import Optional

from engine.config import ClipSettings, ScoringWeights

SUBSCORES = ("hook_strength", "entertainment", "payoff", "emotional_intensity", "unexpectedness",
             "contextual_clarity", "shareability", "audio_excitement", "chat_activity")


def effective_weights(weights: ScoringWeights, chat_available: bool) -> dict[str, float]:
    w = weights.model_dump()
    if not chat_available:
        w.pop("chat_activity")
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}  # proportional redistribution


def weighted(sub: dict[str, Optional[float]], weights: ScoringWeights) -> float:
    w = effective_weights(weights, sub.get("chat_activity") is not None)
    return sum(w[k] * float(sub.get(k) or 0.0) for k in w)


def penalties(duration: float, peak_offset: float, starts_mid: bool, cfg: ClipSettings) -> dict[str, float]:
    p: dict[str, float] = {}
    lo, hi = cfg.sweet_spot
    if duration > hi:
        p["too_long"] = round(min(10.0, (duration - hi) * 0.4), 1)
    elif duration < lo:
        p["too_short"] = round(min(6.0, (lo - duration) * 0.5), 1)
    if peak_offset > 10:
        p["slow_start"] = round(min(10.0, (peak_offset - 10) * 0.8), 1)
    if starts_mid:
        p["mid_sentence_start"] = 5.0
    return p


def total_score(sub: dict[str, Optional[float]], weights: ScoringWeights, pen: Optional[dict[str, float]] = None) -> float:
    return round(max(0.0, min(100.0, weighted(sub, weights) - sum((pen or {}).values()))), 1)


def rejection_reason(c: dict, cfg: ClipSettings, require_speech: bool = True) -> Optional[str]:
    """Deterministic rejection rules applied on top of the AI's own verdict."""
    sig = c["signals"]
    ai = c.get("ai") or {}
    if ai.get("keep") is False:
        return f"AI rejected: {ai.get('reject_reason') or 'no clear payoff'}"
    if sig.get("silence_ratio", 0) > cfg.max_silence_ratio:
        return f"too much dead air ({sig['silence_ratio']:.0%})"
    if require_speech and sig.get("words", 0) < 6 and sig.get("audio_peak", 0) < 0.6:
        return "almost no speech and no strong reaction"
    if c.get("starts_mid_sentence") and c["sub"].get("contextual_clarity", 100) < 50:
        return "starts mid-sentence and needs earlier context"
    if sig.get("repetition", 0) > 0.6:
        return "repetitive commentary"
    if c["sub"].get("payoff", 100) < 25:
        return "no clear payoff"
    if c["total"] < cfg.render_min_score:
        return f"score {c['total']:.0f} below render floor {cfg.render_min_score:.0f}"
    return None


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


def dedupe(cands: list[dict], thresh: float = 0.35) -> list[dict]:
    """Keep the best-scoring candidate among overlapping ones (non-maximum suppression)."""
    kept: list[dict] = []
    for c in sorted(cands, key=lambda c: c["total"], reverse=True):
        if not any(overlap((c["start"], c["end"]), (k["start"], k["end"])) > thresh for k in kept):
            kept.append(c)
    return kept
