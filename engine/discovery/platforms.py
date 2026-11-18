"""Stream discovery via official APIs (PRD §4).

- Twitch Helix  : needs TWITCH_CLIENT_ID + TWITCH_CLIENT_SECRET (free app at dev.twitch.tv/console)
- YouTube Data v3: needs YOUTUBE_API_KEY (Google Cloud console, "YouTube Data API v3")
- Kick          : no public discovery endpoint is configured; add channels manually.

Metrics that an API does not return are stored as None, never estimated.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import requests

TIMEOUT = 20


class DiscoveryError(RuntimeError):
    pass


@dataclass
class Found:
    streamer: str
    platform: str
    title: str
    url: str
    channel_url: Optional[str]
    viewer_count: Optional[int]
    total_views: Optional[int]
    duration_s: Optional[float]
    started_at: Optional[str]
    is_live: bool
    recording_available: bool
    discovery_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_twitch_duration(d: str) -> float:
    total, num = 0.0, ""
    for ch in d:
        if ch.isdigit():
            num += ch
        else:
            total += float(num or 0) * {"h": 3600, "m": 60, "s": 1}.get(ch, 0)
            num = ""
    return total


class Twitch:
    def __init__(self):
        self.cid = os.environ.get("TWITCH_CLIENT_ID")
        self.secret = os.environ.get("TWITCH_CLIENT_SECRET")
        if not (self.cid and self.secret):
            raise DiscoveryError("Twitch discovery needs TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in config/secrets.env "
                                 "(create a free application at https://dev.twitch.tv/console/apps).")
        r = requests.post("https://id.twitch.tv/oauth2/token", timeout=TIMEOUT, params={
            "client_id": self.cid, "client_secret": self.secret, "grant_type": "client_credentials"})
        if r.status_code != 200:
            raise DiscoveryError(f"Twitch auth failed ({r.status_code}): {r.text[:200]}")
        self.headers = {"Client-Id": self.cid, "Authorization": f"Bearer {r.json()['access_token']}"}

    def _get(self, path: str, params: dict) -> dict:
        for attempt in range(3):
            r = requests.get(f"https://api.twitch.tv/helix/{path}", headers=self.headers, params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise DiscoveryError("Twitch rate limit exceeded; try again shortly.")

    def game_id(self, name: str = "Grand Theft Auto VI") -> Optional[str]:
        data = self._get("games", {"name": name}).get("data", [])
        return data[0]["id"] if data else None

    def live(self, game_name: str, limit: int) -> list[Found]:
        gid = self.game_id(game_name)
        if not gid:
            raise DiscoveryError(f"Twitch has no category named '{game_name}'.")
        out = []
        for s in self._get("streams", {"game_id": gid, "first": min(100, limit)}).get("data", []):
            out.append(Found(s["user_name"], "twitch", s["title"], f"https://www.twitch.tv/{s['user_login']}",
                             f"https://www.twitch.tv/{s['user_login']}", s.get("viewer_count"), None, None,
                             s.get("started_at"), True, False))
        return out

    def recent_vods(self, login: str, limit: int = 3) -> list[Found]:
        users = self._get("users", {"login": login}).get("data", [])
        if not users:
            return []
        vids = self._get("videos", {"user_id": users[0]["id"], "type": "archive", "first": limit}).get("data", [])
        return [Found(v["user_name"], "twitch", v["title"], v["url"], f"https://www.twitch.tv/{login}", None,
                      v.get("view_count"), _parse_twitch_duration(v.get("duration", "")), v.get("created_at"), False, True)
                for v in vids]


class YouTube:
    def __init__(self):
        self.key = os.environ.get("YOUTUBE_API_KEY")
        if not self.key:
            raise DiscoveryError("YouTube discovery needs YOUTUBE_API_KEY in config/secrets.env "
                                 "(Google Cloud console -> enable 'YouTube Data API v3' -> create API key).")

    def _get(self, path: str, params: dict) -> dict:
        r = requests.get(f"https://www.googleapis.com/youtube/v3/{path}", params={**params, "key": self.key}, timeout=TIMEOUT)
        if r.status_code == 403 and "quota" in r.text.lower():
            raise DiscoveryError("YouTube API daily quota exhausted.")
        r.raise_for_status()
        return r.json()

    def search(self, query: str, live: bool, limit: int) -> list[Found]:
        params = {"part": "snippet", "q": query, "type": "video", "maxResults": min(50, limit), "order": "viewCount"}
        if live:
            params["eventType"] = "live"
        else:
            params["publishedAfter"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3 * 86400))
        items = self._get("search", params).get("items", [])
        ids = [i["id"]["videoId"] for i in items]
        if not ids:
            return []
        details = {v["id"]: v for v in self._get("videos", {"part": "statistics,liveStreamingDetails,contentDetails,snippet",
                                                           "id": ",".join(ids)}).get("items", [])}
        out = []
        for vid in ids:
            v = details.get(vid)
            if not v:
                continue
            sn, st, ls = v["snippet"], v.get("statistics", {}), v.get("liveStreamingDetails", {})
            cv = ls.get("concurrentViewers")
            out.append(Found(sn["channelTitle"], "youtube", sn["title"], f"https://www.youtube.com/watch?v={vid}",
                             f"https://www.youtube.com/channel/{sn['channelId']}", int(cv) if cv else None,
                             int(st["viewCount"]) if "viewCount" in st else None, _iso_dur(v.get("contentDetails", {}).get("duration")),
                             ls.get("actualStartTime") or sn.get("publishedAt"), sn.get("liveBroadcastContent") == "live",
                             sn.get("liveBroadcastContent") == "none"))
        return out


def _iso_dur(d: Optional[str]) -> Optional[float]:
    if not d or not d.startswith("PT"):
        return None
    return _parse_twitch_duration(d[2:].lower())


def score(f: Found, keywords: list[str], permission: str = "unknown", growth: Optional[float] = None) -> float:
    """0-100 discovery score. Missing metrics contribute nothing rather than a guess."""
    pop = 0.0
    if f.viewer_count:
        pop = min(1.0, math.log10(f.viewer_count + 1) / 5)       # 100k concurrent -> 1.0
    elif f.total_views:
        pop = min(1.0, math.log10(f.total_views + 1) / 6.5)
    relevance = 1.0 if any(k in f.title.lower() for k in keywords) else 0.4
    avail = 1.0 if f.recording_available else (0.5 if f.is_live else 0.2)
    perm = {"explicitly_authorized": 1.0, "official_clipping_program": 1.0, "permission_requested": 0.5,
            "unknown": 0.3, "not_authorized": 0.0}.get(permission, 0.3)
    g = 0.5 if growth is None else max(0.0, min(1.0, growth))
    return round(100 * (0.40 * pop + 0.20 * relevance + 0.15 * avail + 0.15 * perm + 0.10 * g), 1)


def discover(query: str, keywords: list[str], live: bool, limit: int, platforms: list[str],
             permissions: dict[str, str]) -> tuple[list[Found], list[str]]:
    found: list[Found] = []
    notes: list[str] = []
    for p in platforms:
        try:
            if p == "twitch":
                found += Twitch().live("Grand Theft Auto VI", limit) if live else []
                if not live:
                    notes.append("Twitch: VODs are listed per streamer (`clipper discover vods <login>`).")
            elif p == "youtube":
                found += YouTube().search(query, live, limit)
            elif p == "kick":
                notes.append("Kick: no discovery API configured; add Kick channels manually with `clipper streamer add`.")
        except DiscoveryError as e:
            notes.append(str(e))
        except requests.RequestException as e:
            notes.append(f"{p}: request failed ({e}).")
    for f in found:
        f.discovery_score = score(f, keywords, permissions.get(f"{f.platform}:{f.streamer.lower()}", "unknown"))
    found.sort(key=lambda f: f.discovery_score, reverse=True)
    return found[:limit], notes
