"""AI suggestions for the New job page: ready-made instructions the user can give the AI brain about how to choose and
cut clips. Works without an AI key (a fixed list); with one, the AI writes fresh ideas for the chosen video type."""
from __future__ import annotations

PRESETS = [
    "Funny reactions and big laughs",
    "Big action, crashes and explosions",
    "Streamer yelling or raging",
    "Skip long setups, get to the point fast",
    "Surprising plot reveals",
    "Fails and unexpected moments",
    "Keep clips short and punchy",
    "Avoid clips that are mostly talking about chat",
]


def ideas(provider, type_label: str = "streamer video", already: str = "", n: int = 6) -> list[str]:
    """Up to n short instructions. Falls back to PRESETS when there is no AI or it answers badly."""
    if provider is None:
        return PRESETS[:n]
    prompt = (f"The user is making short vertical clips (TikTok/Shorts) from a {type_label} about GTA 6. They give an AI video editor "
              f"instructions about WHICH moments to pick and HOW to cut them. Suggest {n} different, specific instructions, each "
              f"under 12 words, mixing what to look for and what to avoid. Already written: \"{already[:200]}\".\n"
              'Reply as JSON: {"ideas": ["...", "..."]}')
    try:
        res = provider.complete_json("You are a sharp short-form video editor. Be concrete and brief.", prompt)
        out = [str(x).strip().strip('"') for x in (res.get("ideas") or []) if str(x).strip()]
        out = [x for x in out if 3 <= len(x) <= 110][:n]
        return out or PRESETS[:n]
    except Exception:
        return PRESETS[:n]
