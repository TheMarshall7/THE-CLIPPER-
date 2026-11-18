"""Style Recipes, "Our Twist" and the learning loop (PRD M15).

A recipe is only style PARAMETERS (layout, pacing, caption placement, music timing). It never contains
another account's assets, text, watermark or songs, which `validate()` enforces by whitelist.
"""
from __future__ import annotations

import statistics
import time
from typing import Optional

from engine.db import Database, loads

ALLOWED = {
    "recipe_id", "based_on", "layout", "cam_ratio", "hook_text_style", "captions", "pacing", "audio", "zoom", "moment_type",
}
ALLOWED_NESTED = {
    "captions": {"words_per_line", "highlight", "animation", "position"},
    "pacing": {"remove_pauses_over_s", "target_cuts_per_10s"},
    "audio": {"music_mode", "pre_payoff_dropout_s", "sfx"},
    "zoom": {"punch_in_on_reaction"},
}
MUSIC_MODES = {"payoff_drop", "reaction_tail", "underscore", "cold_open"}


def validate(recipe: dict) -> dict:
    extra = set(recipe) - ALLOWED
    if extra:
        raise ValueError(f"recipe has fields that are not style parameters: {sorted(extra)}")
    for k, allowed in ALLOWED_NESTED.items():
        bad = set(recipe.get(k) or {}) - allowed
        if bad:
            raise ValueError(f"recipe.{k} has unsupported fields: {sorted(bad)}")
    mode = (recipe.get("audio") or {}).get("music_mode")
    if mode and mode not in MUSIC_MODES:
        raise ValueError(f"unknown music_mode {mode!r}")
    return recipe


def _find(report: dict, prefix: str) -> Optional[dict]:
    good = [i for i in report["all_patterns"] if i["pattern"].startswith(prefix) and i["status"] != "saturated"]
    good = [i for i in good if (i["median_perf"] or 0) >= (report.get("baseline_median_perf") or 0)]
    return good[0] if good else None


def from_report(report: dict, refs: list[dict], moment_type: str = "any", n: int = 1) -> dict:
    """Build one recipe from the strongest non-saturated patterns in each dimension."""
    layout = _find(report, "layout:")
    pacing = _find(report, "pacing:")
    cap = _find(report, "captions:")
    mus = _find(report, "audio: music") or _find(report, "audio: no added")
    sil = next((i for i in report["all_patterns"] if i["pattern"].startswith("audio: silence") and i["status"] != "saturated"
                and (i["median_perf"] or 0) > (report.get("baseline_median_perf") or 0)), None)
    used = [i for i in (layout, pacing, cap, mus, sil) if i]
    recipe: dict = {"recipe_id": f"{moment_type}_v{n}", "moment_type": moment_type}
    cams = [r["visual"]["layout"].get("cam_ratio") for r in refs if (r["visual"].get("layout") or {}).get("cam_ratio")]
    if layout:
        recipe["layout"] = "split" if "split" in layout["pattern"] else "frame"
        if recipe["layout"] == "split" and cams:
            recipe["cam_ratio"] = round(statistics.median(cams), 2)
    if pacing:
        fast = "fast" in pacing["pattern"]
        recipe["pacing"] = {"remove_pauses_over_s": 0.4 if fast else 0.8, "target_cuts_per_10s": 4 if fast else 1.5}
    recipe["captions"] = {"words_per_line": 3, "animation": "pop", "highlight": "brand_color"}
    if cap:
        recipe["captions"]["position"] = "lower" if "lower" in cap["pattern"] else "seam"
    audio: dict = {}
    if mus:
        p = mus["pattern"]
        audio["music_mode"] = ("cold_open" if "first second" in p else "payoff_drop" if "late" in p else
                               "underscore" if "bed" in p else None)
        if audio["music_mode"] is None:
            audio.pop("music_mode")
    if sil:
        audio["pre_payoff_dropout_s"] = 0.4
    if audio:
        recipe["audio"] = audio
    recipe["zoom"] = {"punch_in_on_reaction": True}
    if used:
        best = max(used, key=lambda i: (i["median_perf"] or 0))
        recipe["based_on"] = (f"{best['pattern']} (n={best['n']}/{best['of']}, "
                              f"{best['vs_baseline'] or '?'}x median"
                              f"{', early signal' if best['early_signal'] else ''}); {len(used)} patterns combined")
    else:
        recipe["based_on"] = "no pattern beat the baseline yet; defaults"
    return validate(recipe)


def twists(recipe: dict, report: dict, artist: Optional[dict] = None) -> list[dict]:
    """2-3 differentiation ideas grounded in the brand profile. Rules, not copying: nothing from other accounts."""
    out = [{"kind": "signature_sonic_moment",
            "text": "Land the song's hook on the payoff every time, so viewers learn that sound means the moment hits."}]
    colour = ((artist or {}).get("visual_style") or {}).get("accent")
    out.append({"kind": "signature_visual",
                "text": f"Keep one brand caption highlight{(' (' + colour + ')') if colour else ''}, the same sound-tag style and "
                        "the same divider on every clip so the account is recognisable."})
    sat = [i for i in report.get("saturated", [])]
    if sat:
        s = sat[0]
        out.append({"kind": "counter_trend",
                    "text": f"'{s['pattern']}' is saturated (n={s['n']}, no lift over baseline): do the opposite, e.g. clean audio "
                            "and real silence before the payoff." if s["pattern"].startswith("audio") else
                            f"'{s['pattern']}' is saturated (n={s['n']}, no lift over baseline): try its opposite on a test batch."})
    else:
        out.append({"kind": "smarter_hooks",
                    "text": "Write hooks about the specific thing that happens in the clip instead of generic ones."})
    return out[:3]


def save(db: Database, recipe: dict, twist_list: list[dict]) -> str:
    validate(recipe)
    rid = recipe["recipe_id"]
    n = 1
    base = rid
    while db.one("SELECT id FROM style_recipes WHERE recipe_id=?", (rid,)):
        n += 1
        rid = f"{base.rsplit('_v', 1)[0]}_v{int(base.rsplit('_v', 1)[1]) + n - 1}" if "_v" in base else f"{base}_{n}"
    recipe["recipe_id"] = rid
    db.insert("style_recipes", {"recipe_id": rid, "recipe_json": recipe, "status": "candidate", "created_at": time.time()})
    for t in twist_list:
        db.insert("twists", {"recipe_id": rid, "twist_json": t, "created_at": time.time()})
    return rid


def get(db: Database, recipe_id: str) -> Optional[dict]:
    row = db.one("SELECT * FROM style_recipes WHERE recipe_id=?", (recipe_id,))
    return {**loads(row["recipe_json"], {}), "status": row["status"]} if row else None


def list_all(db: Database) -> list[dict]:
    out = []
    for r in db.query("SELECT * FROM style_recipes ORDER BY id DESC"):
        rec = loads(r["recipe_json"], {})
        tw = [loads(t["twist_json"], {}) for t in db.query("SELECT twist_json FROM twists WHERE recipe_id=?", (r["recipe_id"],))]
        out.append({**rec, "status": r["status"], "twists": tw, "score": score(db, r["recipe_id"])})
    return out


def set_status(db: Database, recipe_id: str, status: str) -> None:
    if status not in ("candidate", "default", "retired"):
        raise ValueError("status must be candidate, default or retired")
    if status == "default":      # one default at a time
        db.execute("UPDATE style_recipes SET status='candidate' WHERE status='default'")
    db.execute("UPDATE style_recipes SET status=? WHERE recipe_id=?", (status, recipe_id))


def default_recipe(db: Database) -> Optional[dict]:
    row = db.one("SELECT recipe_id FROM style_recipes WHERE status='default' ORDER BY id DESC LIMIT 1")
    return get(db, row["recipe_id"]) if row else None


# ---------------------------------------------------------------- applying a recipe
def apply(settings, recipe: dict):
    """Return a copy of Settings with the recipe's style parameters applied. Nothing is mutated."""
    s = settings.model_copy(deep=True)
    if recipe.get("cam_ratio") and recipe.get("layout") == "split":
        s.video.cam_ratio = float(recipe["cam_ratio"])
    p = recipe.get("pacing") or {}
    if p.get("remove_pauses_over_s"):
        s.video.max_pause = float(p["remove_pauses_over_s"])
    c = recipe.get("captions") or {}
    if c.get("words_per_line"):
        s.captions.max_words_per_line = int(c["words_per_line"])
    if c.get("animation") in ("highlight", "pop", "none"):
        s.captions.animation = c["animation"]
    if c.get("position") in ("seam", "lower", "upper"):
        s.captions.position = c["position"]
    z = recipe.get("zoom") or {}
    if "punch_in_on_reaction" in z:
        s.video.punch_in = bool(z["punch_in_on_reaction"])
    return s


def music_mode(recipe: Optional[dict]) -> Optional[str]:
    return ((recipe or {}).get("audio") or {}).get("music_mode")


# ---------------------------------------------------------------- learning loop
def record_performance(db: Database, clip_id: int, platform: str, views: int, likes: Optional[int] = None,
                       shares: Optional[int] = None, saves: Optional[int] = None, comments: Optional[int] = None) -> int:
    return db.insert("clip_performance", {"clip_id": clip_id, "platform": platform, "views": views, "likes": likes,
                                          "shares": shares, "saves": saves, "comments": comments,
                                          "song_results_json": {}, "recorded_at": time.time()})


def _clip_views(db: Database) -> list[tuple[int, int, Optional[str]]]:
    rows = db.query("SELECT c.id id, MAX(p.views) views, c.layout_json lj FROM clips c JOIN clip_performance p ON p.clip_id=c.id "
                    "GROUP BY c.id")
    return [(r["id"], r["views"] or 0, (loads(r["lj"], {}).get("recipe") or {}).get("recipe_id")) for r in rows]


def score(db: Database, recipe_id: str) -> dict:
    """Median views of clips made with the recipe vs the median of all other measured clips."""
    rows = _clip_views(db)
    mine = [v for _, v, rid in rows if rid == recipe_id]
    others = [v for _, v, rid in rows if rid != recipe_id]
    out = {"clips": len(mine), "median_views": statistics.median(mine) if mine else None,
           "baseline_clips": len(others), "baseline_median_views": statistics.median(others) if others else None,
           "vs_baseline": None, "verdict": "no data yet"}
    if mine and others and statistics.median(others) > 0:
        out["vs_baseline"] = round(statistics.median(mine) / statistics.median(others), 2)
        enough = len(mine) >= 5
        out["verdict"] = ("beats the baseline" if out["vs_baseline"] >= 1.2 else
                          "below the baseline" if out["vs_baseline"] <= 0.8 else "about the same") + ("" if enough else " (early signal)")
    return out
