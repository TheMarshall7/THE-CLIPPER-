"""Signal B: chat activity (PRD §7). Optional — every caller must handle `None`.

Supported inputs:
- yt-dlp YouTube `*.live_chat.json` (one JSON action per line)
- TwitchDownloader chat JSON ({"comments": [{"content_offset_seconds", "message": {"body"}}]})
- generic JSON list [{"t"|"time"|"offset": seconds, "message"|"text": "..."}]
- CSV with columns seconds,message
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d


def parse_chat(path: Path) -> list[tuple[float, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    msgs: list[tuple[float, str]] = []
    if path.suffix.lower() == ".csv":
        for row in csv.reader(text.splitlines()):
            try:
                msgs.append((float(row[0]), row[1] if len(row) > 1 else ""))
            except (ValueError, IndexError):
                continue
        return sorted(msgs)
    stripped = text.lstrip()
    if stripped.startswith("{") and "\n{" in stripped:  # JSON lines (yt-dlp live_chat)
        for line in text.splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            m = _yt_chat_line(obj)
            if m:
                msgs.append(m)
        return sorted(msgs)
    data = json.loads(text)
    if isinstance(data, dict) and "comments" in data:
        for c in data["comments"]:
            body = (c.get("message") or {}).get("body") or ""
            msgs.append((float(c.get("content_offset_seconds", 0)), body))
    elif isinstance(data, list):
        for c in data:
            t = c.get("t", c.get("time", c.get("offset")))
            if t is not None:
                msgs.append((float(t), str(c.get("message", c.get("text", "")))))
    return sorted(msgs)


def _yt_chat_line(obj: dict) -> Optional[tuple[float, str]]:
    try:
        replay = obj["replayChatItemAction"]
        offset = int(replay["videoOffsetTimeMsec"]) / 1000.0
        for act in replay["actions"]:
            item = act.get("addChatItemAction", {}).get("item", {})
            r = item.get("liveChatTextMessageRenderer")
            if r:
                runs = r.get("message", {}).get("runs", [])
                text = "".join(x.get("text") or (x.get("emoji", {}).get("shortcuts") or [""])[0] for x in runs)
                return offset, text
    except (KeyError, ValueError, TypeError):
        return None
    return None


_TOKEN = re.compile(r"[A-Za-z0-9:_']+")


def activity(msgs: list[tuple[float, str]], duration: float, hop: float = 0.5) -> Optional[dict]:
    """Per-hop chat score in [0,1] relative to the stream's own chat baseline."""
    if len(msgs) < 30:
        return None
    n = int(duration / hop) + 1
    counts = np.zeros(n)
    for t, _ in msgs:
        i = int(t / hop)
        if 0 <= i < n:
            counts[i] += 1
    vel = uniform_filter1d(counts, size=int(10 / hop))  # 10 s velocity
    base = median_filter(vel, size=min(n | 1, int(600 / hop) | 1), mode="nearest")
    spread = max(np.median(np.abs(vel - np.median(vel))) * 1.4826, 1e-3)
    z = (vel - base) / spread
    score = 1 - np.exp(-np.clip(z, 0, None) / 3)
    return {"t": np.arange(n) * hop, "score": score, "velocity_per_min": vel / hop * 60}


def window_summary(msgs: list[tuple[float, str]], start: float, end: float, top: int = 5) -> dict:
    inside = [m for t, m in msgs if start <= t <= end]
    tokens = Counter(tok.lower() for m in inside for tok in _TOKEN.findall(m))
    repeated = Counter(m.strip().lower() for m in inside if m.strip())
    return {
        "messages": len(inside),
        "per_min": round(len(inside) / max(1e-6, (end - start) / 60), 1),
        "top_tokens": tokens.most_common(top),
        "top_phrases": [p for p in repeated.most_common(top) if p[1] > 1],
    }
