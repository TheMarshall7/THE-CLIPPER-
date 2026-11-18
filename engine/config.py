"""Settings: one JSON file (config/settings.json) validated by Pydantic.

Every value has a sane default so the engine runs before `setup` is executed.
Secrets (API keys) live in config/secrets.env or environment variables, never in settings.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


def _data_root() -> Path:
    """Where config, database, storage and library live. GTA6_DATA_DIR overrides; a packaged exe keeps
    its data next to itself (so the folder is portable); a source checkout uses the project folder."""
    env = os.environ.get("GTA6_DATA_DIR")
    if env:
        return Path(env).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ROOT = _data_root()
CONFIG_DIR = ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "settings.json"
SECRETS_PATH = CONFIG_DIR / "secrets.env"


class ScoringWeights(BaseModel):
    """Viral score v2. PRD §8 weights, plus hook strength and payoff, which are the two
    strongest predictors of short-form retention. Weights are renormalised to sum to 1."""
    hook_strength: float = 0.14      # does the first ~2 s stop the scroll?
    entertainment: float = 0.15
    payoff: float = 0.12             # is there a clear, satisfying payoff/reaction?
    emotional_intensity: float = 0.10
    unexpectedness: float = 0.12
    contextual_clarity: float = 0.10
    shareability: float = 0.11       # quotable / relatable / "send this to a friend"
    audio_excitement: float = 0.08
    chat_activity: float = 0.08


class MusicSettings(BaseModel):
    enabled: bool = True
    viral_gate: float = 70.0          # only clips at/above this viral score get a song (M12)
    min_window_s: float = 6.0
    duck_db: float = 15.0
    attack_ms: int = 120
    release_ms: int = 400
    volume_preset: Optional[Literal["subtle", "balanced", "featured", "trailer"]] = None   # None = the video type's default
    rotation_window: int = 10
    rotation_max: int = 0             # most times one song may appear in the last `rotation_window` clips (0 = no limit)
    campaign_song: Optional[str] = None
    intelligibility_max_drop: float = 0.10
    sound_tag: bool = True
    style: Literal["auto", "under", "over", "moment"] = "auto"
    ai_placement: bool = True          # "moment" style: let the AI judge where the song should land (falls back to rules)   # auto: talking clips -> under the voice, action -> over the scene
    detect_threshold: float = 0.55    # source-music probability above which a region counts as music
    end_card: bool = False            # M8: song card after the clip (only when a song was mixed in)
    end_card_seconds: float = 2.0
    remove_source_music: bool = False  # M4 option 2: Demucs strips radio/BGM so a song can be placed (needs `pip install demucs`)
    beat_sync_cuts: bool = True        # jump cuts inside the song land on its beat grid (eighth notes)
    pre_drop_dropout: Literal["auto", "on", "off"] = "auto"   # silence for one beat before the chorus hits (auto: music promo only)
    promo_build_bars: int = 4          # music promo: bars of song before the chorus lands on the big moment
    pre_drop_seconds: float = 0.0      # length of that silence; 0 = one beat of the song (max 0.5 s)
    speed_ramps: Literal["auto", "on", "off"] = "auto"   # faster into the hit, slow-mo after it (auto: music promo only)
    beat_fx: Literal["auto", "on", "off"] = "auto"       # zoom pulse on the beat, flash/shake/colour on the chorus hit
    beat_fx_strength: float = 1.0      # 0.3 subtle .. 2.0 wild
    ducking: Literal["smooth", "tight"] = "smooth"   # smooth: steady level while people talk on and off; tight: dips under every phrase
    duck_hold_s: float = 2.5           # smooth ducking: pauses shorter than this still count as talking
    talk_duck_db: Optional[float] = None   # how far the song dips under talking (dB); None = the volume preset's amount


class CaptionStyle(BaseModel):
    font: str = "Arial Black"
    font_size: int = 76
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFD400"
    outline_color: str = "#000000"
    outline: int = 6
    shadow: int = 2
    emphasis_color: str = "#3DFF6E"   # hype/surprise words ("NO WAY", "INSANE")
    max_words_per_line: int = 3
    position: Literal["seam", "lower", "upper"] = "lower"
    animation: Literal["highlight", "pop", "none"] = "pop"
    uppercase: bool = True
    show_hook: bool = True
    hook_seconds: float = 2.8
    show_attribution: bool = True     # creator name + platform at the top of the clip
    credit_in_caption: bool = True    # creator name + link inside the post caption / description
    title_style: Literal["box", "yellow", "dark", "red", "outline"] = "box"   # look of the title card at the start
    lyrics: Literal["auto", "on", "off"] = "auto"   # song lyrics on screen while the song plays (auto: music promo only)


# Every AI provider the engine can judge with. kind "anthropic" uses Anthropic's own API;
# kind "openai" uses the OpenAI chat format, which Gemini, OpenRouter and most other hosts also speak.
PROVIDERS: dict[str, dict] = {
    "gemini": {"label": "Google Gemini (cheapest to start)", "secret": "GEMINI_API_KEY", "kind": "openai",
               "base": "https://generativelanguage.googleapis.com/v1beta/openai", "needs_url": False,
               "models": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"], "default_model": "gemini-2.5-flash",
               "key_prefix": "AIza"},
    "anthropic": {"label": "Anthropic (Claude)", "secret": "ANTHROPIC_API_KEY", "kind": "anthropic",
                  "base": "https://api.anthropic.com", "needs_url": False,
                  "models": ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"], "default_model": "claude-sonnet-5-5",
                  "key_prefix": "sk-ant-"},
    "openai": {"label": "OpenAI (GPT)", "secret": "OPENAI_API_KEY", "kind": "openai",
               "base": "https://api.openai.com/v1", "needs_url": True,
               "models": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"], "default_model": "gpt-4.1-mini",
               "key_prefix": "sk-"},
    "openrouter": {"label": "OpenRouter (Claude, GPT, Gemini, Llama... one key)", "secret": "OPENROUTER_API_KEY", "kind": "openai",
                   "base": "https://openrouter.ai/api/v1", "needs_url": False,
                   "models": ["google/gemini-2.5-flash", "anthropic/claude-sonnet-5-5", "openai/gpt-4.1-mini", "meta-llama/llama-4-maverick"],
                   "default_model": "google/gemini-2.5-flash", "key_prefix": "sk-or-"},
    "custom": {"label": "Other OpenAI-compatible endpoint", "secret": "CUSTOM_API_KEY", "kind": "openai",
               "base": "", "needs_url": True, "models": [], "default_model": "", "key_prefix": None},
}
# Keys that start with a known prefix are matched to that provider. Order matters: "sk-or-" before "sk-".
_PREFIX_ORDER = ["openrouter", "anthropic", "gemini", "openai"]
SECRET_NAMES = {name: spec["secret"] for name, spec in PROVIDERS.items()}


def detect_provider(key: str) -> Optional[str]:
    """Which provider a pasted key belongs to, from its prefix. None when it is not recognised."""
    key = key.strip()
    for name in _PREFIX_ORDER:
        prefix = PROVIDERS[name]["key_prefix"]
        if prefix and key.startswith(prefix):
            return name
    return None


class AISettings(BaseModel):
    """AI is optional. `heuristic` = rules only (no AI, no network). `api` = your own key for any provider in
    PROVIDERS (Gemini, OpenAI, Anthropic, OpenRouter, or a custom OpenAI-compatible address). Keys live in
    config/secrets.env, never here."""
    mode: Literal["heuristic", "api"] = "heuristic"
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    openai_base_url: str = "https://api.openai.com/v1"   # used only by the "openai" and "custom" providers
    timeout_s: int = 90

    @field_validator("mode", mode="before")
    @classmethod
    def _migrate_mode(cls, v):
        # older settings files used "local" (Ollama) / "claude" (CLI): both are gone -> rules only
        return v if v in ("heuristic", "api") else "heuristic"

    @field_validator("provider", mode="before")
    @classmethod
    def _known_provider(cls, v):
        return v if v in PROVIDERS else "gemini"

    def base_url(self) -> str:
        spec = PROVIDERS[self.provider]
        return (self.openai_base_url if spec["needs_url"] else spec["base"]).rstrip("/")


class TranscriptionSettings(BaseModel):
    model: str = "small"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str = "auto"
    language: Optional[str] = "en"
    chunk_seconds: int = 600
    beam_size: int = 5
    vad_filter: bool = True


class ClipSettings(BaseModel):
    allowed_durations: list[int] = [15, 30, 45, 60, 90]
    min_duration: float = 15.0
    min_length: float = 30.0          # YOUR minimum clip length (s), after dead air is cut. Wins over the video type's own minimum.
    max_duration: float = 90.0
    target_min: float = 30.0
    target_max: float = 60.0
    sensitivity: float = Field(0.5, ge=0.0, le=1.0)
    candidate_pool: int = 25          # moments sent to AI evaluation per stream
    max_clips_per_stream: int = 6     # minimum cap on clips rendered per video (long videos get more, see below)
    clips_per_hour: float = 20.0      # long videos: cap = max(max_clips_per_stream, hours * this)  (3 h -> 60)
    pool_per_hour: float = 40.0       # candidates proposed per hour of video (never fewer than candidate_pool)
    render_workers: int = 0           # clips rendered at the same time; 0 = automatic from this PC
    render_min_score: float = 55.0    # below this a candidate is not rendered at all
    quality_threshold: float = 70.0   # below this a rendered clip is never publish-ready
    max_silence_ratio: float = 0.45   # measured AFTER jump cuts would remove dead air
    max_setup_s: float = 14.0         # get to the point: setup before the payoff is capped
    sweet_spot: list[float] = [18.0, 45.0]  # durations that retain best; outside gets a small penalty
    final_ranking: bool = True        # AI compares survivors side by side (editor's final cut)


class VideoSettings(BaseModel):
    width: int = 1080
    height: int = 1920
    cam_ratio: float = 0.38           # share of height used by the face cam in split layout
    max_fps: int = 60
    gameplay_mode: Literal["fill", "fit_blur"] = "fill"
    encoder: Literal["auto", "libx264", "h264_nvenc"] = "auto"
    crf: int = 18                     # lower = sharper + bigger files (16 maximum, 18 high, 20 standard)
    max_cam_zoom: float = 2.5         # a camera blown up more than this looks blurry: auto layout drops a too-small cam panel
    full_view_mode: Literal["blur", "fill"] = "blur"   # whole-frame clips: full picture over a blurred backdrop, or zoom to fill
    preset: str = "medium"
    audio_bitrate: str = "192k"
    loudness_lufs: float = -14.0
    true_peak_db: float = -1.0
    max_download_height: int = 1080
    jump_cuts: bool = True            # remove pauses > max_pause between words
    max_pause: float = 0.6
    punch_in: bool = True             # subtle face/action zoom on the payoff (hard cut, 1.6 s)
    punch_in_zoom: float = 1.18
    divider: bool = True              # thin brand-coloured line between cam and gameplay
    divider_color: str = "#FFD400"
    progress_bar: bool = False
    lead_with_best: bool = False      # the best part plays first (lead_seconds), then the whole clip from its start
    lead_seconds: float = 2.0
    loop_ending: bool = False         # the end dissolves into the first frame so a looping player feels seamless
    face_tracking: bool = True        # split layout: the camera panel follows the face as it moves
    layout_style: Literal["auto", "double", "single"] = "auto"   # double: cam close-up on top + whole scene below


class ReviewSettings(BaseModel):
    """Rules for rejecting clips automatically (0 / off = rule not used)."""
    learn: bool = True                # your Keep / Post / Reject swipes teach the engine what to pick (needs ~12 decisions)
    learn_strength: float = 1.0       # 0.5 gentle, 1 normal, 1.5 strong
    auto_run: bool = False            # run the rules on every new batch as soon as it is rendered
    min_score: float = 0.0
    min_seconds: float = 0.0
    max_seconds: float = 0.0
    reject_blocked: bool = False      # failed a quality check (audio, sync, loudness...)
    reject_music_dropped: bool = False
    reject_no_face: bool = False
    max_silence: float = 0.0          # share of dead air, 0-1
    use_ai: bool = False              # let the AI rate each clip (needs an API key; costs a little per clip)
    ai_min_rating: float = 6.0


class WatchSettings(BaseModel):
    enabled: bool = False
    folder: str = "Watch"             # videos (and a links.txt) dropped here are queued automatically
    max_clips: int = 0                # 0 = engine default
    song: str = ""                    # "" = automatic, "__none__" = no music, or a song slug


class Settings(BaseModel):
    storage_dir: str = "storage"
    folder_names: Literal["video", "number"] = "video"    # Clips/<video title> - song - Clip 01   or   Clips/Video 3 - song - Clip 01
    library_dir: str = "library"
    max_concurrent_jobs: int = 1
    max_job_attempts: int = 3
    processing_priority: Literal["low", "normal"] = "normal"
    cleanup_temp: bool = True
    keep_source_audio: bool = False
    ai: AISettings = AISettings()
    transcription: TranscriptionSettings = TranscriptionSettings()
    clips: ClipSettings = ClipSettings()
    video: VideoSettings = VideoSettings()
    captions: CaptionStyle = CaptionStyle()
    weights: ScoringWeights = ScoringWeights()
    music: MusicSettings = MusicSettings()
    review: ReviewSettings = ReviewSettings()
    watch: WatchSettings = WatchSettings()
    # per-video-type tweaks, e.g. {"movie_trailer": {"payoff": 0.3}} (see engine/profiles.py)
    profile_overrides: dict[str, dict[str, float]] = {}
    vertical: str = "GTA 6"
    vertical_keywords: list[str] = ["gta 6", "gta vi", "gta6", "grand theft auto vi", "grand theft auto 6"]

    # ---- paths -------------------------------------------------------
    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path

    @property
    def storage(self) -> Path:
        return self._resolve(self.storage_dir)

    @property
    def library(self) -> Path:
        return self._resolve(self.library_dir)

    @property
    def db_path(self) -> Path:
        return ROOT / "database" / "clipper.db"

    def stream_dir(self, stream_id: int) -> Path:
        d = self.storage / "streams" / f"{stream_id:05d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def clip_dir(self, clip_id: int) -> Path:
        d = self.storage / "clips" / f"{clip_id:05d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def temp(self) -> Path:
        d = self.storage / "tmp"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def exports(self) -> Path:
        d = self.storage / "exports"
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    load_secrets()
    if path.exists():
        return Settings.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return Settings()


def save_settings(settings: Settings, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")


def load_secrets(path: Path = SECRETS_PATH) -> None:
    """Load KEY=VALUE lines into the environment without overriding real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def connect_with_key(key: str) -> str:
    """Save an API key for the provider its prefix identifies, switch AI on with that provider's default
    model, and return the provider name. Raises ValueError when the key is not recognised."""
    name = detect_provider(key)
    if name is None:
        raise ValueError("Could not tell which provider this key is for. Use: clipper ai key <key> --provider <name>.")
    return connect_provider(name, key)


def connect_provider(name: str, key: str) -> str:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {', '.join(PROVIDERS)}.")
    # pass the paths explicitly: default arguments are bound at import time and would ignore a patched path
    st = load_settings(CONFIG_PATH)
    st.ai.mode = "api"
    st.ai.provider = name
    st.ai.model = PROVIDERS[name]["default_model"] or st.ai.model
    save_settings(st, CONFIG_PATH)
    set_secret(PROVIDERS[name]["secret"], key, SECRETS_PATH)
    return name


def set_secret(name: str, value: str, path: Path = SECRETS_PATH) -> None:
    """Write/replace NAME=value in secrets.env and in this process's environment (takes effect immediately)."""
    value = value.strip()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out, done = [], False
    for line in lines:
        key = line.split("=", 1)[0].strip().lstrip("#").strip()
        if key == name:
            if not done:
                out.append(f"{name}={value}" if value else f"# {name}=")
                done = True
            continue
        out.append(line)
    if not done and value:
        out.append(f"{name}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)


def secret_hint(name: str) -> Optional[str]:
    """Masked view of a stored key for the UI: '...abcd'. Never returns the key itself."""
    v = os.environ.get(name, "")
    return ("..." + v[-4:]) if len(v) >= 8 else None
