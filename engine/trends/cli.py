"""`clipper trends ...` commands (PRD M15)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from engine.config import load_settings
from engine.db import Database, loads
from engine.trends import recipes, report, store

trends_app = typer.Typer(help="Trend intelligence: reference clips, trend report, style recipes, learning loop", no_args_is_help=True)
console = Console()


def _db() -> Database:
    return Database(load_settings().db_path)


@trends_app.command("import")
def trends_import(sources: list[str] = typer.Argument(..., help="Reference clip URLs or saved files you may analyze"),
                  streamer: Optional[str] = None, moment: Optional[str] = typer.Option(None, help="moment type, e.g. crazy_gameplay")):
    """Measure reference clips: layout, pacing, hook frame, captions, music, loudness, sound-effect hits."""
    s, db = load_settings(), _db()
    for src in sources:
        try:
            store.import_reference(db, s, src, streamer, moment, log=console.print)
        except Exception as e:
            console.print(f"[red]{src}: {e}[/]")
    report.refresh(db, console.print)


@trends_app.command("search")
def trends_search(query: str = typer.Option("GTA 6 funny moments", help="YouTube Shorts search"),
                  limit: int = 20, days: int = 30, do_import: bool = typer.Option(False, "--import")):
    """Official YouTube Data API search (needs YOUTUBE_API_KEY). Lists URLs, optionally imports them."""
    s = load_settings()
    urls = store.youtube_search(query, limit, days)
    for u in urls:
        console.print(u)
    if do_import:
        db = _db()
        for u in urls:
            try:
                store.import_reference(db, s, u, log=console.print)
            except Exception as e:
                console.print(f"[red]{u}: {e}[/]")
        report.refresh(db, console.print)


@trends_app.command("metrics")
def trends_metrics(reference_id: int, views: Optional[int] = None, likes: Optional[int] = None, comments: Optional[int] = None,
                   shares: Optional[int] = None, saves: Optional[int] = None, followers: Optional[int] = None):
    """Enter metrics you read from an analytics page (shares/saves are rarely public)."""
    store.set_metrics(_db(), reference_id, views=views, likes=likes, comments=comments, shares=shares, saves=saves,
                      followers=followers)
    console.print("saved")


@trends_app.command("list")
def trends_list():
    t = Table("id", "account", "published", "len", "cuts/10s", "layout", "music", "perf x median")
    for r in store.all_references(_db()):
        t.add_row(str(r["id"]), str(r["account"] or r["streamer"] or "-"), str(r["publish_date"] or "-"),
                  str(r["duration"]), str(r["visual"].get("cuts_per_10s")), str((r["visual"].get("layout") or {}).get("kind")),
                  str((r["audio"].get("music") or {}).get("kind")), str(r["perf"] if r["perf"] is not None else "n/a"))
    console.print(t)


@trends_app.command("report")
def trends_report(window: str = typer.Option("7d", help="24h | 7d | 30d"), refresh: bool = False):
    """What's working now, rising, saturated - every line with its sample size."""
    db = _db()
    rep = report.refresh(db, lambda m: None)[window] if refresh or not report.latest(db, window) else report.latest(db, window)
    console.print(f"[bold]{window}[/]: {rep['clips_in_window']} clips ({rep['clips_with_performance']} with performance data)")
    for title, key in (("What's working now", "working_now"), ("Rising", "rising"), ("Saturated", "saturated")):
        t = Table(title=title)
        for c in ("pattern", "n / measured", "x median", "note"):
            t.add_column(c)
        for i in rep[key]:
            t.add_row(i["pattern"], f"{i['n']}/{i['of']}", str(i["vs_baseline"] or "n/a"), "early signal" if i["early_signal"] else "")
        console.print(t)
    console.print(f"sounds: {rep['sounds']}")
    console.print(f"[dim]{rep['note']}[/]")


@trends_app.command("recipe")
def trends_recipe(moment: str = typer.Option("any", help="moment type this recipe is for"), window: str = "30d"):
    """Create a Style Recipe (and 2-3 twists) from the current trend report."""
    db = _db()
    rep = report.latest(db, window) or report.refresh(db, lambda m: None)[window]
    refs = store.all_references(db)
    rec = recipes.from_report(rep, refs, moment, n=len(recipes.list_all(db)) + 1)
    art = db.one("SELECT * FROM artists ORDER BY id LIMIT 1")
    tw = recipes.twists(rec, rep, {"visual_style": loads(art["visual_style_json"], {})} if art else None)
    rid = recipes.save(db, rec, tw)
    console.print(json.dumps(recipes.get(db, rid), indent=2))
    for t in tw:
        console.print(f"twist [{t['kind']}]: {t['text']}")


@trends_app.command("recipes")
def trends_recipes():
    t = Table("recipe", "status", "based on", "clips", "vs baseline", "verdict")
    for r in recipes.list_all(_db()):
        sc = r["score"]
        t.add_row(r["recipe_id"], r["status"], r.get("based_on", "")[:60], str(sc["clips"]), str(sc["vs_baseline"] or "n/a"), sc["verdict"])
    console.print(t)


@trends_app.command("promote")
def trends_promote(recipe_id: str):
    """Make a recipe the default for all new renders."""
    recipes.set_status(_db(), recipe_id, "default")
    console.print(f"{recipe_id} is now the default recipe")


@trends_app.command("retire")
def trends_retire(recipe_id: str):
    recipes.set_status(_db(), recipe_id, "retired")


@trends_app.command("use")
def trends_use(clip_id: int, recipe_id: Optional[str] = typer.Argument(None), off: bool = False, render: bool = True):
    """Apply a recipe to one clip (or --off for none) and re-render it."""
    from engine.pipeline import Engine

    eng = Engine(load_settings(), log=console.print)
    c = eng.db.one("SELECT layout_json FROM clips WHERE id=?", (clip_id,))
    lay = loads(c["layout_json"], {})
    if off:
        lay["recipe"] = {"off": True}
    else:
        if not recipe_id or not recipes.get(eng.db, recipe_id):
            raise typer.Exit("unknown recipe id")
        lay["recipe"] = {"recipe_id": recipe_id}
    eng.db.update("clips", clip_id, {"layout_json": lay})
    if render:
        eng.render_clip(clip_id)


@trends_app.command("performance")
def trends_performance(clip_id: int, platform: str, views: int, likes: Optional[int] = None, shares: Optional[int] = None,
                       saves: Optional[int] = None, comments: Optional[int] = None):
    """Record how a posted clip did; recipe scores update from this."""
    recipes.record_performance(_db(), clip_id, platform, views, likes, shares, saves, comments)
    console.print("recorded")
