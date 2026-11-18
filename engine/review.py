"""Clip review: swipe decisions, rejection categories, "post this" marks, auto-reject rules and learning.

Decisions live in `clip_review` (one row per clip). The clip's own `approval` column is kept in sync so the
dashboard and exporter keep working: keep/post -> approved, reject -> rejected.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Callable, Optional

from engine.db import Database, loads

# categories a rejected clip is filed under (number keys 1-9 in the Review tab)
REASONS = ["Weak start / boring", "No payoff", "Wrong part of song", "Face cut off / framing", "Bad captions",
           "Audio problem", "Too short / too long", "Not interesting", "Other"]
AUTO_PREFIX = "Auto: "

SCHEMA = """
CREATE TABLE IF NOT EXISTS clip_review (
    clip_id INTEGER PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
    decision TEXT NOT NULL,            -- keep | reject | post
    reason TEXT,
    auto INTEGER NOT NULL DEFAULT 0,
    posted INTEGER NOT NULL DEFAULT 0,
    posted_at REAL,
    posted_platform TEXT,
    ai_rating REAL,
    ai_note TEXT,
    decided_at REAL NOT NULL
);
"""


def ensure(db: Database) -> None:
    db.execute(SCHEMA)


def decide(db: Database, clip_id: int, decision: str, reason: Optional[str] = None, auto: bool = False,
           ai_rating: Optional[float] = None, ai_note: Optional[str] = None) -> None:
    if decision not in ("keep", "reject", "post"):
        raise ValueError(decision)
    now = time.time()
    prev = db.one("SELECT posted, posted_at, posted_platform FROM clip_review WHERE clip_id=?", (clip_id,))
    db.execute("INSERT OR REPLACE INTO clip_review(clip_id, decision, reason, auto, posted, posted_at, posted_platform, "
               "ai_rating, ai_note, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (clip_id, decision, reason, int(auto), prev["posted"] if prev else 0, prev["posted_at"] if prev else None,
                prev["posted_platform"] if prev else None, ai_rating, ai_note, now))
    db.execute("UPDATE clips SET approval=?, updated_at=? WHERE id=?",
               ("rejected" if decision == "reject" else "approved", now, clip_id))
    if not auto:
        _relearn(db)


def _relearn(db: Database) -> None:
    try:                                  # learning must never get in the way of swiping
        from engine import learn
        learn.retrain(db)
    except Exception:
        pass


def undo(db: Database, clip_id: int) -> None:
    db.execute("DELETE FROM clip_review WHERE clip_id=?", (clip_id,))
    db.execute("UPDATE clips SET approval='pending', updated_at=? WHERE id=?", (time.time(), clip_id))
    _relearn(db)


def last_manual(db: Database) -> Optional[int]:
    """The clip you decided on most recently (by hand), so Undo still works after leaving the tab or restarting."""
    r = db.one("SELECT clip_id FROM clip_review WHERE auto=0 ORDER BY decided_at DESC LIMIT 1")
    return r["clip_id"] if r else None


def mark_posted(db: Database, clip_id: int, platform: str = "") -> None:
    db.execute("UPDATE clip_review SET posted=1, posted_at=?, posted_platform=? WHERE clip_id=?", (time.time(), platform, clip_id))


def to_review(db: Database, include_auto_kept: bool = True) -> list[dict]:
    """Rendered clips still waiting for a human decision, most promising first (learned taste, then score)."""
    rows = db.query("SELECT c.*, s.title AS stream_title FROM clips c JOIN streams s ON s.id=c.stream_id "
                    "WHERE c.status='rendered' AND c.video_path IS NOT NULL "
                    "AND c.id NOT IN (SELECT clip_id FROM clip_review WHERE auto=0) "
                    "AND c.id NOT IN (SELECT clip_id FROM clip_review WHERE auto=1 AND decision='reject')")
    taste = taste_model(db)
    out = [dict(r) for r in rows]
    for r in out:
        r["taste"] = taste_score(db, r, taste)
    out.sort(key=lambda r: (r["taste"], r["score"] or 0), reverse=True)
    return out


def decisions(db: Database, decision: Optional[str] = None) -> list[dict]:
    sql = ("SELECT c.id, c.score, c.video_path, c.thumb_path, r.decision, r.reason, r.auto, r.posted, r.ai_rating, r.ai_note "
           "FROM clip_review r JOIN clips c ON c.id=r.clip_id")
    args: tuple = ()
    if decision:
        sql += " WHERE r.decision=?"
        args = (decision,)
    return [dict(r) for r in db.query(sql + " ORDER BY r.decided_at DESC", args)]


def counts(db: Database) -> dict:
    waiting = db.one("SELECT COUNT(*) FROM clips c WHERE c.status='rendered' AND c.video_path IS NOT NULL "
                     "AND c.id NOT IN (SELECT clip_id FROM clip_review WHERE auto=0) "
                     "AND c.id NOT IN (SELECT clip_id FROM clip_review WHERE auto=1 AND decision='reject')")[0]
    out = {"keep": 0, "reject": 0, "post": 0, "waiting": waiting, "reasons": {}}
    for r in db.query("SELECT decision, reason, COUNT(*) n FROM clip_review WHERE NOT (auto=1 AND decision='keep') "
                      "GROUP BY decision, reason"):
        out[r["decision"]] = out.get(r["decision"], 0) + r["n"]
        if r["decision"] == "reject":
            out["reasons"][r["reason"] or "Other"] = r["n"]
    return out


# ------------------------------------------------------------------ learning from what you keep / post
def _features(db: Database, clip: dict) -> dict:
    cand = db.one("SELECT category FROM candidates WHERE id=?", (clip.get("candidate_id"),)) if clip.get("candidate_id") else None
    song = db.one("SELECT s.slug FROM music_placements m JOIN songs s ON s.id=m.song_id WHERE m.clip_id=? AND m.chosen=1", (clip["id"],))
    return {"category": cand["category"] if cand else None, "song": song["slug"] if song else None}


def taste_model(db: Database) -> dict:
    """Approval rate per category and per song among clips you manually reviewed; 'post' counts double."""
    rows = db.query("SELECT r.clip_id, r.decision, c.candidate_id, c.id FROM clip_review r JOIN clips c ON c.id=r.clip_id "
                    "WHERE r.auto=0")
    tot: dict = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        f = _features(db, dict(r))
        good = {"post": 2.0, "keep": 1.0, "reject": 0.0}[r["decision"]]
        for key in (("category", f["category"]), ("song", f["song"])):
            if key[1]:
                tot[key][0] += good
                tot[key][1] += 2.0
    return {k: v[0] / v[1] for k, v in tot.items() if v[1] >= 6}      # need at least 3 reviews to trust a rate


def taste_score(db: Database, clip: dict, model: Optional[dict] = None) -> float:
    model = taste_model(db) if model is None else model
    f = _features(db, clip)
    vals = [model[k] for k in (("category", f["category"]), ("song", f["song"])) if k in model]
    return sum(vals) / len(vals) if vals else 0.5


def insights(db: Database) -> list[str]:
    model = taste_model(db)
    cats = sorted(((v, k[1]) for k, v in model.items() if k[0] == "category"), reverse=True)
    songs = sorted(((v, k[1]) for k, v in model.items() if k[0] == "song"), reverse=True)
    lines = []
    if cats:
        lines.append(f"You keep most: {cats[0][1].replace('_', ' ')} ({cats[0][0]:.0%})" +
                     (f"; least: {cats[-1][1].replace('_', ' ')} ({cats[-1][0]:.0%})" if len(cats) > 1 else ""))
    if songs:
        lines.append(f"Best song so far: {songs[0][1]} ({songs[0][0]:.0%} kept)")
    top = db.one("SELECT reason, COUNT(*) n FROM clip_review WHERE decision='reject' AND auto=0 GROUP BY reason ORDER BY n DESC LIMIT 1")
    if top:
        lines.append(f"Most common reject reason: {top['reason']} ({top['n']})")
    try:
        from engine import learn
        s = learn.status(db)
        if s["active"]:
            lines.insert(0, f"Learning from {s['decisions']} of your swipes: it is already shaping new picks.")
        elif s["decisions"]:
            lines.insert(0, f"Learning: {s['need']} more swipes and it starts shaping picks.")
    except Exception:
        pass
    return lines or ["Review a few clips and the engine will start learning your taste."]


# ------------------------------------------------------------------ automatic rejection
def _qc_blocks(clip: dict) -> list[str]:
    qc = loads(clip.get("qc_json"), {}) or {}
    return [b for b in qc.get("blocks", []) if not b.startswith("creator permission")]


def auto_review(db: Database, settings, clip_ids: Optional[list[int]] = None, log: Callable[[str], None] = print) -> dict:
    """Reject rendered clips that break the user's rules. Never touches clips a human already decided on."""
    rs = settings.review
    where = "status='rendered' AND id NOT IN (SELECT clip_id FROM clip_review WHERE auto=0)"
    rows = [dict(r) for r in db.query(f"SELECT * FROM clips WHERE {where}")]
    if clip_ids is not None:
        want = set(clip_ids)
        rows = [r for r in rows if r["id"] in want]
    provider = None
    if rs.use_ai:
        from engine.ai.llm import get_provider
        provider, msg = get_provider(settings.ai)
        if provider is None:
            log(f"AI check skipped: {msg}")
    out = {"checked": len(rows), "rejected": 0, "by_rule": Counter()}
    for c in rows:
        reason = _rule_hit(db, c, rs)
        rating = note = None
        if reason is None and provider is not None:
            rating, note = _ai_rate(db, c, provider, log)
            if rating is not None and rating < rs.ai_min_rating:
                reason = f"AI rated {rating:.0f}/10"
        if reason:
            decide(db, c["id"], "reject", AUTO_PREFIX + reason, auto=True, ai_rating=rating, ai_note=note)
            out["rejected"] += 1
            out["by_rule"][reason.split(" (")[0]] += 1
        else:
            if rating is not None:        # keep the AI's opinion for the review screen; the clip stays "waiting" (auto=1 keep)
                db.execute("INSERT OR REPLACE INTO clip_review(clip_id, decision, reason, auto, ai_rating, ai_note, decided_at) "
                           "SELECT ?,?,?,?,?,?,? WHERE NOT EXISTS (SELECT 1 FROM clip_review WHERE clip_id=? AND auto=0)",
                           (c["id"], "keep", None, 1, rating, note, time.time(), c["id"]))
    out["by_rule"] = dict(out["by_rule"])
    return out


def _rule_hit(db: Database, c: dict, rs) -> Optional[str]:
    dur = (c["end"] or 0) - (c["start"] or 0)
    qc = loads(c.get("qc_json"), {}) or {}
    layout = loads(c.get("layout_json"), {}) or {}
    if rs.min_score and (c["score"] or 0) < rs.min_score:
        return f"score under {rs.min_score:g}"
    if rs.min_seconds and dur < rs.min_seconds:
        return f"shorter than {rs.min_seconds:g}s"
    if rs.max_seconds and dur > rs.max_seconds:
        return f"longer than {rs.max_seconds:g}s"
    if rs.reject_blocked and _qc_blocks(c):
        return "failed quality checks (" + _qc_blocks(c)[0][:40] + ")"
    if rs.reject_music_dropped:
        music = qc.get("music", {}) or {}
        if music.get("blocked") and "dropped" in str(music.get("blocked")).lower() or \
                "intelligibility" in str(music.get("blocked", "")).lower():
            return "song dropped (speech not clear)"
    if rs.reject_no_face and (layout.get("layout") == "gameplay_only" or float(layout.get("confidence") or 1) < 0.5):
        return "no clear face / camera"
    sil = (qc.get("checks", {}) or {}).get("silence_ratio")
    if rs.max_silence and sil is not None and float(sil) > rs.max_silence:
        return "too much dead air"
    return None


def _ai_rate(db: Database, c: dict, provider, log) -> tuple[Optional[float], Optional[str]]:
    cand = db.one("SELECT category, summary FROM candidates WHERE id=?", (c.get("candidate_id"),)) if c.get("candidate_id") else None
    hooks = loads(c.get("hooks_json"), []) or []
    meta = loads(c.get("metadata_json"), {}) or {}
    prompt = (f"Short-form clip for TikTok/Shorts.\nMoment type: {cand['category'] if cand else 'unknown'}\n"
              f"What happens: {(cand['summary'] if cand else '') or meta.get('description', '')}\n"
              f"Title text: {hooks[0] if hooks else meta.get('title', '')}\nLength: {(c['end'] - c['start']):.0f}s\n\n"
              'Rate how likely this is to hold viewers and be shared, 1-10, and give a one-line reason. '
              'Reply as JSON: {"rating": <number>, "note": "<short reason>"}')
    try:
        r = provider.complete_json("You are a ruthless short-form video editor. Be honest and brief.", prompt)
        return float(r.get("rating")), str(r.get("note", ""))[:200]
    except Exception as e:      # an AI hiccup must never reject (or block) a clip
        log(f"AI check failed for clip {c['id']}: {e}")
        return None, None
