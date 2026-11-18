"""Video types: the same moments must rank differently depending on what the video is."""
from engine import profiles
from engine.config import ScoringWeights, Settings
from engine.intelligence import moments
from engine.metadata.generate import heuristic
from engine.scoring.score import rejection_reason, total_score

KAI = "GTA 6 MARATHON | Official Trailer"


def test_auto_detect_from_title_and_music():
    d = lambda t, m=0.1: profiles.detect(t, m, ["gta 6", "gta vi"])[0]  # noqa: E731
    assert d(KAI) == "game_trailer"
    assert d("Avengers: Doomsday | Official Trailer") == "movie_trailer"
    assert d("Stranger Things Season 6 Official Teaser Netflix") == "movie_trailer"
    assert d("The Joe Rogan Experience #2100 - interview with a chef") == "podcast"
    assert d("NBA Finals Game 7 Highlights") == "sports"
    assert d("KAI REACTS to the GTA 6 trailer") == "stream"          # 'reacts' beats 'trailer'
    assert d("GTA 6 gameplay no commentary") == "gameplay"
    assert d("untitled 4", 0.55) == "movie_trailer"                    # mostly music, no game cue: the broader rubric
    assert d("GTA 6 untitled", 0.55) == "game_trailer"
    assert d("untitled 5", 0.05) == "stream"
    assert profiles.get("cinematic").key == "game_trailer"             # older streams keep working
    assert profiles.get("nonsense").key == "stream"


def test_every_profile_is_complete_and_sane():
    assert set(profiles.ORDER) == {"stream", "gameplay", "game_trailer", "movie_trailer", "podcast", "sports"}
    for p in profiles.PROFILES.values():
        assert set(p.weights) == set(ScoringWeights.model_fields), p.key
        assert abs(sum(p.weights.values()) - 1.0) < 0.06, (p.key, sum(p.weights.values()))
        assert p.min_duration < p.sweet_spot[0] < p.sweet_spot[1] <= p.target_max
        assert p.evidence in ("speech", "spectacle", "mixed") and p.layout in ("auto", "frame")
        assert p.volume in ("subtle", "balanced", "featured", "trailer") and p.ranks and p.guidance


def test_effective_settings_apply_profile_and_respect_user_choices():
    base = Settings()
    stream = profiles.effective_settings(base, profiles.get("stream"))
    trailer = profiles.effective_settings(base, profiles.get("game_trailer"))
    pod = profiles.effective_settings(base, profiles.get("podcast"))
    assert trailer.weights.shareability > stream.weights.shareability > 0
    assert stream.weights.chat_activity > trailer.weights.chat_activity
    assert stream.video.jump_cuts and not trailer.video.jump_cuts and pod.video.jump_cuts
    assert pod.clips.sweet_spot[1] > stream.clips.sweet_spot[1] > trailer.clips.sweet_spot[0] - 1
    assert (stream.music.volume_preset, trailer.music.volume_preset, pod.music.volume_preset) == ("featured", "trailer", "subtle")
    assert base.music.volume_preset is None                            # the original is not modified
    base.music.volume_preset = "balanced"                              # an explicit user choice wins everywhere
    assert profiles.effective_settings(base, profiles.get("game_trailer")).music.volume_preset == "balanced"
    base.profile_overrides = {"movie_trailer": {"payoff": 0.5, "not_a_weight": 9}}
    assert profiles.effective_settings(base, profiles.get("movie_trailer")).weights.payoff == 0.5


SIG = dict(audio_peak=0.5, audio_mean=0.4, hook_audio=0.4, hook_text="", hook_question=False, contrast=0.1, chat_peak=None,
           silence_ratio=0.1, words=0, surprise_hits=0, hype_hits=0, laugh_hits=0, callout_hits=0, repetition=0.0,
           laughter_like=0.0, scream_like=0.0, z_rise_max=0.5, peak_offset_s=5.0)


def _score(sig, prof_key):
    prof = profiles.get(prof_key)
    S = profiles.effective_settings(Settings(), prof)
    c = {"signals": sig, "start": 0, "end": 30, "peak": 5}
    ai = moments.heuristic_eval(c, prof)
    c["ai"], c["sub"] = ai, {**{k: ai[k] for k in moments.RUBRIC}, "audio_excitement": 100 * sig["audio_peak"], "chat_activity": None}
    c["total"] = total_score(c["sub"], S.weights)
    c["reject"] = rejection_reason(c, S.clips, prof.require_speech)
    return c


def test_same_moments_rank_differently_by_video_type():
    # A: a streamer losing it - lots of surprised talk, laughter, 'clip that', modest picture change
    talk = {**SIG, "words": 45, "surprise_hits": 4, "hype_hits": 1, "laugh_hits": 2, "callout_hits": 1, "laughter_like": 0.4,
            "audio_peak": 0.75, "audio_mean": 0.6, "hook_audio": 0.7, "hook_question": True, "hook_text": "wait what is that", "contrast": 0.25}
    # B: a silent spectacle - huge picture/sound change, nobody speaking
    show = {**SIG, "words": 0, "audio_peak": 0.95, "audio_mean": 0.8, "hook_audio": 0.8, "contrast": 0.9, "z_rise_max": 5.0, "scream_like": 0.1}
    assert _score(talk, "stream")["total"] > _score(show, "stream")["total"]
    for spectacle_type in ("game_trailer", "gameplay"):
        assert _score(show, spectacle_type)["total"] > _score(talk, spectacle_type)["total"], spectacle_type
    # a movie trailer values both, so it sits between the two extremes for the speech-led moment
    assert _score(talk, "stream")["total"] > _score(talk, "movie_trailer")["total"] > _score(talk, "game_trailer")["total"] - 20


def test_speech_requirement_depends_on_type():
    quiet_footage = {**SIG, "words": 0, "audio_peak": 0.3, "contrast": 0.7}
    assert "almost no speech" in _score(quiet_footage, "stream")["reject"]
    assert _score(quiet_footage, "podcast")["reject"] and "almost no speech" in _score(quiet_footage, "podcast")["reject"]
    assert "almost no speech" not in str(_score(quiet_footage, "game_trailer")["reject"])
    assert "almost no speech" not in str(_score(quiet_footage, "gameplay")["reject"])


def test_hooks_fit_the_video_type():
    ctx = {"streamer": "Rockstar Games", "transcript": "Get in, get the ticket, and get out.", "vertical": "GTA 6",
           "hook_style": "scene", "clip_word": "scene", "profile_label": "Game trailer / cutscene"}
    hooks = heuristic(ctx)["hooks"]
    assert hooks and not any("reaction" in h.lower() for h in hooks)         # nobody is reacting in a trailer
    react = heuristic({**ctx, "hook_style": "reaction", "streamer": "Kai", "profile_label": "Streamer stream"})["hooks"]
    assert any("reaction" in h.lower() for h in react)
