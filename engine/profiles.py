"""Content profiles: what kind of video this is decides HOW its moments are ranked and cut.

A streamer reacting, a game trailer, a movie trailer, a podcast and a sports highlight reel
are won by different things, so each profile sets:
  - ranking weights (what matters most to the viral score),
  - what counts as evidence (what is said vs. what is shown/heard),
  - clip length and pacing, jump cuts, layout,
  - how music is placed and how loud it is,
  - the guidance given to the AI judge, and the tone of rules-only hooks.

Everything here is plain data so it is easy to tune. Per-profile weight overrides can also be put in
settings.json under "profile_overrides": {"movie_trailer": {"payoff": 0.3}}.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from engine.config import ScoringWeights, Settings

LEGACY = {"cinematic": "game_trailer"}          # older streams stored this value


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    short: str
    ranks: tuple[str, ...]                      # "how it ranks" bullets shown on the page
    evidence: str                               # speech | spectacle | mixed
    weights: dict
    require_speech: bool
    min_duration: float
    target_max: float
    sweet_spot: tuple[float, float]
    max_setup_s: float
    jump_cuts: bool
    layout: str                                 # auto (face cam / split) | frame (whole shot, blurred fill)
    trailer_music: bool                         # song takes over after the setup and the chorus hits the reveal
    volume: str                                 # music volume preset
    default_category: str
    guidance: str                               # extra instructions for the AI judge
    copy_note: str                              # tone for AI-written hooks/captions
    hook_style: str                             # reaction | scene | quote
    clip_word: str = "moment"


def _w(**kw) -> dict:
    base = dict(hook_strength=0.0, entertainment=0.0, payoff=0.0, emotional_intensity=0.0, unexpectedness=0.0,
                contextual_clarity=0.0, shareability=0.0, audio_excitement=0.0, chat_activity=0.02)
    base.update(kw)
    return base


PROFILES: dict[str, Profile] = {p.key: p for p in [
    Profile(
        "stream", "Streamer stream / VOD", "A streamer playing or reacting live",
        ("The streamer's reaction is the star: surprise, laughter, shouting, 'clip that'",
         "Needs a clear setup, a payoff and a reaction; filler talk and dead air are rejected",
         "Chat spikes count when a chat log is available", "Pauses are cut out; face cam on top, game below"),
        "speech", _w(hook_strength=.14, entertainment=.15, payoff=.12, emotional_intensity=.10, unexpectedness=.12,
                     contextual_clarity=.10, shareability=.11, audio_excitement=.08, chat_activity=.08),
        True, 15, 60, (18, 45), 14, True, "auto", False, "featured", "funny",
        "The streamer is the content. Their reaction, commentary and chemistry with chat decide whether a moment works.",
        "Hooks are about the streamer's reaction or the surprising moment. Never invent things they did not say.",
        "reaction", "moment"),
    Profile(
        "gameplay", "Gameplay footage (no commentary)", "Raw gameplay, nobody talking over it",
        ("The action itself is ranked: sudden spikes in sound and picture, escapes, crashes, big hits",
         "Strong opening beat and a clear payoff matter most; talk is not required",
         "Whole shot is kept (HUD stays visible); the song takes over and the drop lands on the payoff",
         "Short clips: 12-35 s sweet spot"),
        "spectacle", _w(hook_strength=.18, entertainment=.10, payoff=.18, emotional_intensity=.06, unexpectedness=.16,
                        contextual_clarity=.06, shareability=.10, audio_excitement=.14),
        False, 10, 45, (12, 35), 8, False, "frame", True, "trailer", "crazy_gameplay",
        "There is no commentary. Judge the gameplay itself: action, skill, chaos, funny physics, near misses, payoffs.",
        "Hooks describe what happens on screen. There is no streamer reaction to mention.",
        "scene", "moment"),
    Profile(
        "game_trailer", "Game trailer / cutscene", "An official game trailer or cinematic",
        ("Iconic, shareable shots and reveals rank highest, then hook and payoff",
         "Memorable lines and character moments count even without action",
         "Trailer-style music: the song comes in after the setup and the chorus hits the reveal",
         "Whole frame is kept; 20-50 s sweet spot"),
        "spectacle", _w(hook_strength=.18, entertainment=.08, payoff=.14, emotional_intensity=.10, unexpectedness=.12,
                        contextual_clarity=.06, shareability=.20, audio_excitement=.10),
        False, 15, 60, (20, 50), 20, False, "frame", True, "trailer", "crazy_gameplay",
        "This is a game trailer or cinematic, not a streamer. The best moments are reveals, iconic shots, big action beats "
        "and quotable lines that a fan would send to a friend. Do not penalise the lack of commentary.",
        "Hooks are about the scene or reveal itself. Never claim anything about the game that the clip does not show.",
        "scene", "scene"),
    Profile(
        "movie_trailer", "Movie / TV trailer", "A film or series trailer",
        ("Reveals, star moments and the line everyone will quote rank highest",
         "Dialogue and score both count; the cleanest hook beats raw loudness",
         "Trailer-style music; whole frame kept; 15-40 s sweet spot"),
        "mixed", _w(hook_strength=.18, entertainment=.06, payoff=.12, emotional_intensity=.12, unexpectedness=.14,
                    contextual_clarity=.10, shareability=.20, audio_excitement=.06),
        False, 12, 50, (15, 40), 12, False, "frame", True, "trailer", "dramatic",
        "This is a movie or TV trailer. The best moments are twists, reveals, a big star shot, a quotable line, or an emotional "
        "beat. A great clip stands alone without knowing the film.",
        "Hooks tease the moment without spoilers or invented plot details. Use only what the clip shows or says.",
        "quote", "scene"),
    Profile(
        "podcast", "Podcast / interview", "People talking: podcasts, interviews, panels",
        ("Quotable lines, strong opinions and stories with a clear payoff rank highest",
         "Needs to make sense on its own: clarity and shareability weigh most",
         "Longer clips (30-75 s), pauses trimmed, face shown full; music stays subtle"),
        "speech", _w(hook_strength=.12, entertainment=.14, payoff=.10, emotional_intensity=.08, unexpectedness=.10,
                     contextual_clarity=.20, shareability=.20, audio_excitement=.04),
        True, 25, 90, (30, 75), 20, True, "auto", False, "subtle", "memorable_dialogue",
        "This is a conversation. The best moments are a surprising claim, a strong opinion, a funny story or a confession that "
        "makes sense without the rest of the episode. Reject small talk, tangents and anything that needs earlier context.",
        "Hooks quote or tease the actual claim. Never put words in anyone's mouth.",
        "quote", "moment"),
    Profile(
        "sports", "Sports / highlights", "Matches and highlight reels",
        ("The play itself: payoff, emotion and crowd or commentator excitement rank highest",
         "Short punchy clips (12-35 s); whole frame kept",
         "Sound intensity counts more than talk"),
        "mixed", _w(hook_strength=.10, entertainment=.10, payoff=.20, emotional_intensity=.14, unexpectedness=.14,
                    contextual_clarity=.04, shareability=.10, audio_excitement=.16),
        False, 10, 45, (12, 35), 8, False, "frame", False, "featured", "crazy_gameplay",
        "This is sports. The best moments are decisive plays, upsets, near misses and big reactions from commentators or the crowd.",
        "Hooks describe the play. Never invent scores, names or outcomes that the clip does not show.",
        "scene", "play"),
]}

ORDER = list(PROFILES)


def get(key: Optional[str]) -> Profile:
    key = LEGACY.get(key or "", key or "")
    return PROFILES.get(key, PROFILES["stream"])


def effective_settings(settings: Settings, profile: Profile) -> Settings:
    """A copy of the settings with this profile's ranking, pacing and music defaults applied."""
    s = settings.model_copy(deep=True)
    w = {**profile.weights, **{k: float(v) for k, v in (settings.profile_overrides.get(profile.key) or {}).items()
                               if k in ScoringWeights.model_fields}}
    s.weights = ScoringWeights(**w)
    c = s.clips
    c.min_duration = max(profile.min_duration, settings.clips.min_length)       # the user's minimum always wins
    c.target_max = max(profile.target_max, c.min_duration + 10.0)
    c.sweet_spot, c.max_setup_s = list(profile.sweet_spot), profile.max_setup_s
    s.video.jump_cuts = settings.video.jump_cuts and profile.jump_cuts
    if s.music.volume_preset is None:
        s.music.volume_preset = profile.volume  # type: ignore[assignment]
    return s


_STREAM = re.compile(r"react|stream|\blive\b|\bvod\b|first time|let'?s play|playthrough|blind run", re.I)
_PODCAST = re.compile(r"podcast|interview|\bep\.? ?\d+|episode|conversation|panel|sits down|q&a|\btalks\b", re.I)
_SPORTS = re.compile(r"highlights?|\bnba\b|\bnfl\b|\bnhl\b|\bmlb\b|\bufc\b|\bfifa\b|premier league|champions league|match recap|full game", re.I)
_TRAILER = re.compile(r"trailer|teaser|cinematic|cut ?scene|official (reveal|clip)|launch trailer", re.I)
_MOVIE = re.compile(r"movie|\bfilm\b|\bseries\b|season \d|in theaters|in cinemas|netflix|disney|marvel|hbo|a24|official teaser", re.I)
_GAME = re.compile(r"\bgame\b|gaming|playstation|\bps[45]\b|xbox|nintendo|switch|steam|rockstar|gta|trailer \d|gameplay", re.I)
_GAMEPLAY = re.compile(r"gameplay|walkthrough|montage|no commentary|longplay", re.I)


def detect(title: str, music_fraction: float, vertical_keywords: list[str], words_per_min: float | None = None) -> tuple[str, str]:
    """Best-guess profile key + the reason. A heuristic from the title and how much of the audio is music;
    the New job dropdown (or the Streams table) always overrides it."""
    t = (title or "").lower()
    game = bool(_GAME.search(t)) or any(k in t for k in vertical_keywords)
    if _STREAM.search(t):
        return "stream", "the title looks like a streamer video"
    if _PODCAST.search(t):
        return "podcast", "the title looks like a podcast or interview"
    if _SPORTS.search(t):
        return "sports", "the title looks like sports or highlights"
    is_game = game and not _MOVIE.search(t)
    if _TRAILER.search(t):
        # no game cue = use the broader movie/TV rubric (it weighs speech AND spectacle), which is the safer default
        if is_game:
            return "game_trailer", "the title looks like a game trailer"
        return "movie_trailer", "the title looks like a trailer (no game cue, so the broader movie/TV ranking is used)"
    if _GAMEPLAY.search(t):
        return "gameplay", "the title looks like gameplay footage"
    if words_per_min is not None and words_per_min < 30:
        # a "streamer video" needs a streamer talking. Hardly any speech = compilation / montage / raw gameplay
        return "gameplay", f"very little talking ({words_per_min:.0f} words a minute), so it is treated as gameplay footage"
    if music_fraction >= 0.4:
        key = "game_trailer" if is_game else "movie_trailer"
        return key, f"music plays {music_fraction:.0%} of the time, so it is more likely a trailer than a streamer"
    return "stream", "no sign of a trailer or other type: treating it as a streamer video"


def describe() -> list[dict]:
    return [{"key": p.key, "label": p.label, "short": p.short, "ranks": list(p.ranks), "evidence": p.evidence,
             "sweet_spot": list(p.sweet_spot), "trailer_music": p.trailer_music, "layout": p.layout} for p in PROFILES.values()]
