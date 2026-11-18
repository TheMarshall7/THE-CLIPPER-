"""Hooks (PRD §12) and social metadata (PRD §13).

The model sees the clip's actual transcript and must stay factual. Attribution
(streamer + source URL) is appended deterministically, never left to the model.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

SYSTEM = """You write short-form video copy for {vertical} clips. Be accurate: never claim anything the
transcript does not support, never invent numbers, never use clickbait that misrepresents the moment.
Hooks are about the moment itself. Answer with one JSON object only."""

PROMPT = """Source: {streamer} ({platform}) | Video type: {profile_label}
Tone: {copy_note}
Moment category: {category}
What happens: {summary}
Clip transcript (what the streamer actually says):
\"\"\"{transcript}\"\"\"

Return JSON:
{{
 "hooks": [three different opening hooks, each <= 9 words, curiosity-driven, specific to THIS clip],
 "title": "social title <= 70 chars",
 "description": "1-2 sentence description",
 "caption": "post caption <= 150 chars",
 "hashtags": {{"tiktok": [3-6 tags], "instagram": [3-8 tags], "youtube_shorts": [3-5 tags]}},
 "suggested_platform": "tiktok" | "instagram" | "youtube_shorts",
 "category": "short posting category, e.g. Gaming / Comedy"
}}
Hashtags must be relevant (game, streamer, moment type). No hashtag stuffing."""


def _clean_tags(tags, vertical_tag: str) -> list[str]:
    out = []
    for t in tags or []:
        t = "#" + re.sub(r"[^A-Za-z0-9_]", "", str(t).lstrip("#"))
        if len(t) > 1 and t.lower() not in (x.lower() for x in out):
            out.append(t)
    if vertical_tag.lower() not in (x.lower() for x in out):
        out.insert(0, vertical_tag)
    return out[:8]


def attribution_line(streamer: str, platform: str, url: str, fmt: Optional[str] = None) -> str:
    if fmt:
        return fmt.format(streamer=streamer, platform=platform, url=url)
    plat = {"youtube": "YouTube", "twitch": "Twitch", "kick": "Kick", "file": "original stream"}.get(platform, platform)
    return f"Clip from {streamer} on {plat}" + (f": {url}" if platform != "file" else "")


def heuristic(ctx: dict) -> dict:
    """No-model fallback: copy built from the clip's own words, labelled as such.
    Reaction videos get reaction hooks; trailers and footage get scene hooks (there is no reaction to mention)."""
    text = ctx["transcript"]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 3 <= len(s.split()) <= 12]
    quote = max(sentences, key=lambda s: s.count("!") * 2 + s.count("?") + len(s) / 50, default=None)
    who = ctx["streamer"] or "This streamer"
    game = ctx["vertical"]
    style = ctx.get("hook_style", "reaction")
    word = ctx.get("clip_word", "moment")
    hooks = []
    if quote and style in ("reaction", "quote"):
        hooks.append(f"\u201c{quote.rstrip('.')}\u201d")
    if style == "reaction":
        hooks += [f"{who}'s reaction to this {game} moment", f"Watch what happens in {game} here",
                  f"{who} was not ready for this"]
    else:
        hooks += [f"Watch this {game} {word}", f"The {game} {word} that stood out", f"Wait for the {game} {word} to land"]
        if quote and style == "scene":
            hooks.append(f"\u201c{quote.rstrip('.')}\u201d")
    tag = "#" + re.sub(r"\W", "", game)
    st = "#" + re.sub(r"\W", "", who) if ctx["streamer"] else None
    tags = [tag, "#gaming", "#clips"] + ([st] if st else [])
    if ctx.get("profile_label") and style != "reaction":
        tags = [tag, "#trailer" if "railer" in ctx["profile_label"] else "#clips"] + ([st] if st else [])
    return {
        "mode": "heuristic",
        "hooks": hooks[:3],
        "title": f"{who} | {game} {word}" if style == "reaction" else f"{game} {word}",
        "description": f"{who} reacts during a {game} stream." if style == "reaction" else f"A {game} {word}.",
        "caption": (quote or f"{who} in {game}")[:150],
        "hashtags": {"tiktok": tags, "instagram": tags + ["#reels"], "youtube_shorts": tags[:3] + ["#shorts"]},
        "suggested_platform": "tiktok",
        "category": "Gaming",
    }


def generate(ctx: dict, provider, log: Callable[[str], None] = print) -> dict:
    """ctx: streamer, platform, url, category, summary, transcript, vertical, attribution_format"""
    meta = None
    if provider is not None and ctx["transcript"].strip():
        try:
            defaults = {"profile_label": "Streamer stream", "copy_note": "Hooks are about the moment."}
            meta = provider.complete_json(SYSTEM.format(vertical=ctx["vertical"]), PROMPT.format(**{**defaults, **ctx}))
            meta["mode"] = f"{provider.name}:{provider.model}"
        except Exception as e:
            log(f"Metadata generation via AI failed ({e}); using transcript-based fallback.")
    if meta is None:
        meta = heuristic(ctx)
    hooks = [str(h).strip() for h in (meta.get("hooks") or []) if str(h).strip()][:3]
    fallback = [h for h in heuristic(ctx)["hooks"] if h not in hooks]
    while len(hooks) < 3:                          # always three options, whatever the AI / heuristic returned
        hooks.append(fallback.pop(0) if fallback else f"Wait for it ({len(hooks) + 1})")
    meta["hooks"] = hooks
    vtag = "#" + re.sub(r"\W", "", ctx["vertical"])
    tags = meta.get("hashtags") or {}
    meta["hashtags"] = {p: _clean_tags(tags.get(p), vtag) for p in ("tiktok", "instagram", "youtube_shorts")}
    attr = attribution_line(ctx["streamer"] or "the streamer", ctx["platform"], ctx["url"], ctx.get("attribution_format"))
    meta["attribution"] = attr
    meta["description"] = f"{meta.get('description', '').strip()}\n\n{attr}".strip()
    meta["platform_variants"] = {
        p: {"caption": f"{meta.get('caption', '')} {' '.join(meta['hashtags'][p])}".strip(), "title": meta.get("title", "")}
        for p in ("tiktok", "instagram", "youtube_shorts")
    }
    return meta
