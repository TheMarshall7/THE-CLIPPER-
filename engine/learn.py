"""Learning from your swipes in the Review tab.

Every clip you Keep / Post / Reject by hand is a lesson. From those decisions the engine fits a small model (logistic
regression, strongly regularised so a handful of swipes can't mislead it) over the things the engine already measures
for every moment: the sub-scores (hook, payoff, entertainment ...), the clip length and the moment type. The model's
opinion then nudges the score of every NEW candidate, so the engine picks more of what you like and fewer of what you
reject. It starts helping after ~12 decisions (with at least 3 keeps and 3 rejects) and keeps improving as you swipe.

Auto-rejects never teach it: only your own decisions count.
"""
from __future__ import annotations

import json
import math
import time
from typing import Optional

import numpy as np

from engine.db import Database, loads
from engine.intelligence.moments import CATEGORIES, RUBRIC

FEATS = RUBRIC + ("audio_excitement", "chat_activity")
NICE = {"hook_strength": "a strong hook", "entertainment": "entertainment", "payoff": "a clear payoff",
        "emotional_intensity": "emotional intensity", "unexpectedness": "surprise", "contextual_clarity": "clear context",
        "shareability": "shareability", "audio_excitement": "loud, exciting audio", "chat_activity": "chat going wild",
        "length": "longer clips"}
MIN_DECISIONS, MIN_EACH = 12, 3
KEY, SINCE_KEY = "learned_model", "learn_since"


def vector(sub: dict, category: Optional[str], duration: float) -> np.ndarray:
    x = [float(sub.get(k) or 0.0) / 100.0 for k in FEATS]
    x.append(min(max(duration, 0.0), 120.0) / 60.0)
    x += [1.0 if category == c else 0.0 for c in CATEGORIES]
    return np.array(x, dtype=float)


def names() -> list[str]:
    return list(FEATS) + ["length"] + [f"type:{c}" for c in CATEGORIES]


def _training_rows(db: Database) -> list[tuple[np.ndarray, float, float]]:
    since = db.one("SELECT value FROM meta WHERE key=?", (SINCE_KEY,))
    t0 = float(since["value"]) if since else 0.0
    rows = db.query("SELECT r.decision, c.start, c.end, k.category, k.scores_json FROM clip_review r "
                    "JOIN clips c ON c.id=r.clip_id LEFT JOIN candidates k ON k.id=c.candidate_id "
                    "WHERE r.auto=0 AND r.decided_at>=? AND k.scores_json IS NOT NULL", (t0,))
    out = []
    for r in rows:
        sub = {k: v for k, v in (loads(r["scores_json"], {}) or {}).items() if not str(k).startswith("_")}
        y = {"post": 1.0, "keep": 1.0, "reject": 0.0}[r["decision"]]
        w = 1.5 if r["decision"] == "post" else 1.0          # "I'm going to post it" is the strongest yes
        out.append((vector(sub, r["category"], (r["end"] or 0) - (r["start"] or 0)), y, w))
    return out


def fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 2.0, iters: int = 600, lr: float = 0.4) -> tuple[np.ndarray, float]:
    """Weighted, L2-regularised logistic regression by gradient descent (tiny data, so plain numpy is plenty)."""
    n, d = X.shape
    coef, b = np.zeros(d), 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ coef + b, -30, 30)))
        g = (p - y) * w
        coef -= lr * (X.T @ g / n + l2 * coef / n)
        b -= lr * g.mean()
    return coef, b


def retrain(db: Database) -> Optional[dict]:
    """Refit from all manual decisions and store the model. Returns it, or None while there isn't enough to learn from."""
    rows = _training_rows(db)
    y = np.array([r[1] for r in rows])
    if len(rows) < MIN_DECISIONS or (y == 1).sum() < MIN_EACH or (y == 0).sum() < MIN_EACH:
        db.execute("DELETE FROM meta WHERE key=?", (KEY,))
        return None
    X = np.vstack([r[0] for r in rows])
    w = np.array([r[2] for r in rows])
    coef, b = fit(X, y, w)
    model = {"coef": coef.tolist(), "bias": float(b), "n": len(rows), "keep": int((y == 1).sum()), "reject": int((y == 0).sum()),
             "base": float(np.average(y, weights=w)), "trained_at": time.time()}
    db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (KEY, json.dumps(model)))
    return model


def model(db: Database) -> Optional[dict]:
    r = db.one("SELECT value FROM meta WHERE key=?", (KEY,))
    return json.loads(r["value"]) if r else None


def predict(m: dict, sub: dict, category: Optional[str], duration: float) -> float:
    z = float(np.array(m["coef"]) @ vector(sub, category, duration) + m["bias"])
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def adjust(db: Database, sub: dict, category: Optional[str], duration: float, strength: float = 1.0) -> float:
    """Score points (about -15..+15) to add to a new candidate: positive when it resembles what you keep."""
    m = model(db)
    if not m or strength <= 0:
        return 0.0
    conf = min(1.0, m["n"] / 40.0) * strength
    adj = (predict(m, sub, category, duration) - m["base"]) * 60.0 * conf
    return round(max(-15.0 * strength, min(15.0 * strength, adj)), 1)


def forget(db: Database) -> None:
    """Start learning from scratch (your past decisions stay, they just stop teaching)."""
    db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (SINCE_KEY, str(time.time())))
    db.execute("DELETE FROM meta WHERE key=?", (KEY,))


def status(db: Database) -> dict:
    """Plain-language state for the Learning dialog."""
    rows = _training_rows(db)
    n = len(rows)
    keeps = sum(1 for r in rows if r[1] == 1.0)
    m = model(db)
    out = {"decisions": n, "keeps": keeps, "rejects": n - keeps, "active": m is not None, "likes": [], "dislikes": [],
           "need": max(0, MIN_DECISIONS - n)}
    if m:
        pairs = sorted(zip(names(), m["coef"]), key=lambda kv: kv[1], reverse=True)
        labels = lambda k: NICE.get(k) or k.replace("type:", "").replace("_", " ") + " moments"      # noqa: E731
        out["likes"] = [labels(k) for k, v in pairs if v > 0.15][:4]
        out["dislikes"] = [labels(k) for k, v in reversed(pairs) if v < -0.15][:4]
    return out
