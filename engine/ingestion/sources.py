"""Source handling: local files, YouTube, Twitch, Kick (PRD §3, §5).

Remote media is accessed through yt-dlp, which uses each platform's public
playback endpoints. It does not bypass DRM, logins or private/subscriber-only
restrictions; such content fails with a clear error instead.

Audio-first: `download_audio` fetches the audio-only rendition. Video is only
fetched later for the selected clip windows via `download_section`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PLATFORM_PATTERNS = {
    "youtube": re.compile(r"^https?://(www\.|m\.)?(youtube\.com|youtu\.be)/", re.I),
    "twitch": re.compile(r"^https?://(www\.|m\.)?twitch\.tv/", re.I),
    "kick": re.compile(r"^https?://(www\.)?kick\.com/", re.I),
}
MEDIA_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".flv", ".ts", ".m4v", ".avi", ".mp3", ".wav", ".m4a", ".aac", ".flac"}


class SourceError(RuntimeError):
    """User-facing ingestion failure with a clear message."""


@dataclass
class SourceInfo:
    platform: str
    url: str
    title: Optional[str] = None
    uploader: Optional[str] = None
    channel_url: Optional[str] = None
    duration: Optional[float] = None
    view_count: Optional[int] = None
    concurrent_view_count: Optional[int] = None
    is_live: bool = False
    started_at: Optional[str] = None
    has_chat_replay: bool = False
    raw: dict = field(default_factory=dict)


def classify(source: str) -> str:
    p = Path(source)
    if p.exists():
        if p.suffix.lower() not in MEDIA_EXTS:
            raise SourceError(f"Unsupported file type: {p.suffix}")
        return "file"
    for name, pat in PLATFORM_PATTERNS.items():
        if pat.match(source):
            return name
    if re.match(r"^https?://", source):
        raise SourceError(f"Unsupported URL (supported: YouTube, Twitch, Kick, local files): {source}")
    raise SourceError(f"File not found and not a URL: {source}")


def _friendly(err: Exception) -> str:
    msg = str(err)
    low = msg.lower()
    if "private" in low:
        return "This recording is private. Only content you can legitimately access can be processed."
    if "members-only" in low or "subscriber" in low or "join this channel" in low:
        return "This recording is restricted to members/subscribers; it will not be accessed."
    if "sign in" in low or "login" in low or "age" in low and "confirm" in low:
        return "The platform requires sign-in for this recording; the engine does not bypass authentication."
    if "removed" in low or "deleted" in low or "not exist" in low or "404" in low or "unavailable" in low:
        return "The recording is unavailable or has been deleted."
    if "429" in low or "rate" in low and "limit" in low:
        return "The platform is rate-limiting requests. The job will be retried later."
    if "drm" in low:
        return "This content is DRM-protected and cannot be processed."
    return msg.splitlines()[-1][:400]


def _ydl(opts: dict):
    import yt_dlp

    base = {"quiet": True, "no_warnings": True, "noprogress": True, "retries": 5,
            "fragment_retries": 5, "socket_timeout": 30}
    base.update(opts)
    return yt_dlp.YoutubeDL(base)


def fetch_info(source: str) -> SourceInfo:
    platform = classify(source)
    if platform == "file":
        from engine.media.ffmpeg import probe

        p = Path(source).resolve()
        info = probe(p)
        return SourceInfo(platform="file", url=str(p), title=p.stem, duration=info.duration)
    try:
        with _ydl({"skip_download": True}) as ydl:
            raw = ydl.extract_info(source, download=False)
    except Exception as e:  # yt_dlp.utils.DownloadError and friends
        raise SourceError(_friendly(e)) from e
    if raw.get("_type") == "playlist":
        raise SourceError("That URL is a playlist/channel. Paste a single video or VOD URL.")
    subs = raw.get("subtitles") or {}
    ts = raw.get("release_timestamp") or raw.get("timestamp")
    started = None
    if ts:
        import datetime as dt
        started = dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat()
    return SourceInfo(
        platform=platform,
        url=raw.get("webpage_url") or source,
        title=raw.get("title"),
        uploader=raw.get("uploader") or raw.get("channel") or raw.get("uploader_id"),
        channel_url=raw.get("channel_url") or raw.get("uploader_url"),
        duration=raw.get("duration"),
        view_count=raw.get("view_count"),
        concurrent_view_count=raw.get("concurrent_view_count"),
        is_live=bool(raw.get("is_live")),
        started_at=started,
        has_chat_replay="live_chat" in subs,
        raw={k: raw.get(k) for k in ("id", "extractor", "live_status", "was_live", "upload_date", "fps", "width", "height")},
    )


def download_audio(info: SourceInfo, out_dir: Path) -> Path:
    """Fetch the audio-only rendition (much smaller than video)."""
    if info.platform == "file":
        return Path(info.url)
    if info.is_live:
        raise SourceError("This stream is still live. Process it once the VOD/recording is available.")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpl = str(out_dir / "source_audio.%(ext)s")
    try:
        with _ydl({"format": "bestaudio/bestaudio*/best[height<=480]/best", "outtmpl": tmpl, "continuedl": True}) as ydl:
            raw = ydl.extract_info(info.url, download=True)
            path = Path(ydl.prepare_filename(raw))
    except Exception as e:
        raise SourceError(_friendly(e)) from e
    if not path.exists():
        found = sorted(out_dir.glob("source_audio.*"))
        if not found:
            raise SourceError("Audio download finished but no file was written.")
        path = found[0]
    return path


def download_chat(info: SourceInfo, out_dir: Path) -> Optional[Path]:
    """YouTube live-chat replay via yt-dlp. Twitch/Kick chat: import a file with `clipper chat import`."""
    if info.platform != "youtube" or not info.has_chat_replay:
        return None
    tmpl = str(out_dir / "chat.%(ext)s")
    try:
        with _ydl({"skip_download": True, "writesubtitles": True, "subtitleslangs": ["live_chat"], "outtmpl": tmpl}) as ydl:
            ydl.extract_info(info.url, download=True)
    except Exception:
        return None
    found = sorted(out_dir.glob("chat*.live_chat.json"))
    return found[0] if found else None


def download_section(info: SourceInfo, start: float, end: float, dst: Path, max_height: int = 1080,
                     audio_file: Optional[Path] = None) -> Path:
    """Fetch only [start, end] of the video (PRD §5 step 9).

    YouTube's separate audio streams come back truncated in ranged downloads, so the VIDEO is
    fetched as a section and the AUDIO is cut from the full audio track already downloaded for
    analysis (audio_file), then both are muxed. Lengths are verified."""
    from engine.media.ffmpeg import cut_av, probe, run

    if info.platform == "file":
        return cut_av(info.url, start, end, dst)
    from yt_dlp.utils import download_range_func

    dst.parent.mkdir(parents=True, exist_ok=True)
    stem = dst.with_name(dst.stem + "_v")
    for old in stem.parent.glob(stem.name + ".*"):
        old.unlink()
    opts = {
        "format": f"bestvideo[height<={max_height}]/best[height<={max_height}]/best",
        "outtmpl": str(stem) + ".%(ext)s",
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "overwrites": True,
    }
    try:
        with _ydl(opts) as ydl:
            ydl.extract_info(info.url, download=True)
    except Exception as e:
        raise SourceError(_friendly(e)) from e
    found = [p for p in stem.parent.glob(stem.name + ".*") if p.suffix in (".mp4", ".mkv", ".webm")]
    if not found:
        raise SourceError("Segment download produced no file.")
    vid = found[0]
    dur = end - start
    if audio_file is None or not Path(audio_file).exists():
        audio_file = download_audio(info, dst.parent / "_audio")
    run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-i", str(vid), "-ss", f"{start:.3f}",
         "-t", f"{dur:.3f}", "-i", str(audio_file), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "256k", "-t", f"{dur:.3f}", str(dst)])
    vid.unlink(missing_ok=True)
    p = probe(dst)
    if p.video_duration and p.video_duration < dur - 1.5:
        raise SourceError(f"Platform returned only {p.video_duration:.1f}s of video for a {dur:.1f}s section.")
    return dst
