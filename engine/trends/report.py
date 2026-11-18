"""Trend aggregation (PRD M15): patterns, performance-weighted, rising vs saturated, with sample sizes."""
from __future__ import annotations

import datetime as dt
import statistics
import time
from typing import Callable, Optional

from engine.db import Database
from engine.trends.store import all_references

WINDOWS = {"24h": 1, "7d": 7, "30d": 30}
EARLY_N = 8          # fewer clips than this -> "early signal"


def _bucket(v: Optional[float], edges: list[float], labels: list[str]) -> Optional[str]:
    if v is None:
        return None
    for e, l in zip(edges, labels):
        if v < e:
            return l
    return labels[-1]


# pattern name -> predicate(ref) -> bool | None (None = not measurable for this clip)
def _patterns(r: dict) -> dict[str, Optional[bool]]:
    v, a = r["visual"], r["audio"]
    lay = (v.get("layout") or {}).get("kind")
    cuts = v.get("cuts_per_10s")
    dur = v.get("duration_s")
    music = (a.get("music") or {}).get("kind")
    hook = v.get("hook_frame") or {}
    cap = v.get("captions") or {}
    out: dict[str, Optional[bool]] = {
        "layout: split (face cam + gameplay)": None if lay is None else lay == "split",
        "layout: full-screen gameplay": None if lay is None else lay in ("gameplay_only", "fullcam"),
        "pacing: fast (4+ cuts / 10 s)": None if cuts is None else cuts >= 4,
        "pacing: calm (under 2 cuts / 10 s)": None if cuts is None else cuts < 2,
        "length: under 20 s": None if dur is None else dur < 20,
        "length: 20-35 s": None if dur is None else 20 <= dur < 35,
        "length: over 35 s": None if dur is None else dur >= 35,
        "hook frame: face visible": hook.get("face_visible"),
        "hook frame: text on screen": hook.get("has_text"),
        "hook frame: high motion": None if hook.get("motion") is None else hook["motion"] > 12,
        "captions: lower third": None if cap.get("present") is None else (cap.get("position") in ("lower", "bottom")),
        "captions: centre": None if cap.get("present") is None else (cap.get("position") in ("middle", "upper")),
        "audio: no added music": None if music is None else music == "none",
        "audio: music from the first second": None if music is None else music == "cold_open",
        "audio: music enters late": None if music is None else music == "late_entry",
        "audio: music bed throughout": None if music is None else music == "bed",
        "audio: silence right before the payoff": a.get("silence_before_peak"),
        "audio: lots of sound-effect hits (3+ / 10 s)": None if a.get("transient_hits_per_10s") is None else a["transient_hits_per_10s"] >= 3,
        "structure: loops back to the start": v.get("loops_cleanly"),
    }
    return out


def _when(r: dict) -> float:
    d = r.get("publish_date")
    if d:
        try:
            return dt.datetime.fromisoformat(d[:10]).replace(tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            pass
    return r.get("created_at") or time.time()


def build_report(refs: list[dict], window: str = "30d", now: Optional[float] = None) -> dict:
    now = now or time.time()
    days = WINDOWS[window]
    cur = [r for r in refs if now - _when(r) <= days * 86400]
    prev = [r for r in refs if days * 86400 < now - _when(r) <= 2 * days * 86400]
    scored = [r for r in cur if r.get("perf") is not None]
    base_med = statistics.median([r["perf"] for r in scored]) if scored else None
    top_cut = sorted((r["perf"] for r in scored), reverse=True)[max(0, len(scored) // 4 - 1)] if scored else None

    def stats_for(rs: list[dict]) -> dict[str, dict]:
        agg: dict[str, dict] = {}
        for r in rs:
            for name, hit in _patterns(r).items():
                a = agg.setdefault(name, {"n": 0, "measured": 0, "perfs": [], "examples": []})
                if hit is None:
                    continue
                a["measured"] += 1
                if hit:
                    a["n"] += 1
                    if r.get("perf") is not None:
                        a["perfs"].append(r["perf"])
                    if len(a["examples"]) < 3:
                        a["examples"].append({"id": r["id"], "url": r["url"], "perf": r.get("perf")})
        return agg

    now_s, prev_s = stats_for(cur), stats_for(prev)
    items = []
    for name, a in now_s.items():
        if a["n"] == 0:
            continue
        med = round(statistics.median(a["perfs"]), 2) if a["perfs"] else None
        share = a["n"] / a["measured"] if a["measured"] else 0
        top_hits = sum(1 for r in scored if top_cut is not None and r["perf"] >= top_cut and _patterns(r).get(name))
        top_n = sum(1 for r in scored if top_cut is not None and r["perf"] >= top_cut)
        share_top = top_hits / top_n if top_n else None
        p = prev_s.get(name)
        prev_share_top = None
        if p and prev:
            ps = [r for r in prev if r.get("perf") is not None]
            if ps:
                pc = sorted((r["perf"] for r in ps), reverse=True)[max(0, len(ps) // 4 - 1)]
                tp = [r for r in ps if r["perf"] >= pc]
                prev_share_top = sum(1 for r in tp if _patterns(r).get(name)) / len(tp) if tp else None
        status = "steady"
        if share_top is not None and prev_share_top is not None and share_top - prev_share_top >= 0.15 and a["n"] >= 5:
            status = "rising"
        elif share >= 0.6 and med is not None and base_med and med <= base_med * 1.0 and a["n"] >= 8:
            status = "saturated"           # almost everyone does it and it no longer lifts performance
        items.append({"pattern": name, "n": a["n"], "of": a["measured"], "share": round(share, 2),
                      "median_perf": med, "vs_baseline": round(med / base_med, 2) if med and base_med else None,
                      "share_of_top_clips": round(share_top, 2) if share_top is not None else None,
                      "prev_share_of_top": round(prev_share_top, 2) if prev_share_top is not None else None,
                      "status": status, "early_signal": a["n"] < EARLY_N, "examples": a["examples"]})
    items.sort(key=lambda i: (i["median_perf"] is None, -(i["median_perf"] or 0), -i["n"]))
    sounds = _sounds(cur)
    return {"window": window, "generated_at": now, "clips_in_window": len(cur), "clips_with_performance": len(scored),
            "clips_in_previous_window": len(prev), "baseline_median_perf": base_med,
            "working_now": [i for i in items if i["status"] != "saturated" and (i["median_perf"] or 0) > (base_med or 0)][:8],
            "rising": [i for i in items if i["status"] == "rising"],
            "saturated": [i for i in items if i["status"] == "saturated"],
            "all_patterns": items, "sounds": sounds,
            "note": "Patterns with fewer than %d clips are labelled early signal. Performance is views/day relative to "
                    "account size, as a multiple of the median; unavailable metrics are excluded, not estimated." % EARLY_N}


def _sounds(cur: list[dict]) -> dict:
    """Music usage in recent top clips: kind/BPM only. Trending songs are analysed for style, never reused."""
    rows = [(r["audio"].get("bpm"), (r["audio"].get("music") or {}).get("kind"), r.get("perf")) for r in cur
            if (r["audio"].get("music") or {}).get("kind") not in (None, "none")]
    bpms = [b for b, _, _ in rows if b]
    kinds: dict[str, int] = {}
    for _, k, _ in rows:
        kinds[k] = kinds.get(k, 0) + 1
    return {"clips_with_music": len(rows), "median_bpm": round(statistics.median(bpms), 1) if bpms else None,
            "entry_styles": kinds}


def refresh(db: Database, log: Callable[[str], None] = print) -> dict:
    refs = all_references(db)
    out = {}
    for w in WINDOWS:
        rep = build_report(refs, w)
        db.insert("trend_reports", {"window": w, "report_json": rep, "created_at": time.time()})
        out[w] = rep
        log(f"trend report {w}: {rep['clips_in_window']} clips, {len(rep['rising'])} rising, {len(rep['saturated'])} saturated")
    return out


def latest(db: Database, window: str = "7d") -> Optional[dict]:
    from engine.db import loads

    row = db.one("SELECT report_json FROM trend_reports WHERE window=? ORDER BY id DESC LIMIT 1", (window,))
    return loads(row["report_json"]) if row else None
