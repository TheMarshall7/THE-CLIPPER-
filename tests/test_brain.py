"""Scoring, boundaries, jump cuts, captions (PRD §8, §9, §11)."""
import numpy as np

from engine.captions.ass import build_ass
from engine.clipping.boundaries import keep_intervals, remap, sentences, snap
from engine.config import CaptionStyle, ClipSettings, ScoringWeights
from engine.intelligence import moments
from engine.scoring.score import dedupe, effective_weights, penalties, rejection_reason, total_score


def W(word, s, e):
    return {"word": word, "start": s, "end": e, "prob": 0.9}


SEGS = [  # one long Whisper/VAD segment containing three sentences and a pause
    {"start": 0.0, "end": 14.0, "text": "", "words": [
        W("Okay", 0.0, 0.3), W("let's", 0.4, 0.6), W("drive.", 0.7, 1.0),
        W("Wait.", 4.0, 4.4), W("What", 4.6, 4.8), W("is", 4.85, 4.95), W("that?", 5.0, 5.4),
        W("No", 9.0, 9.2), W("way!", 9.3, 9.8), W("That", 10.0, 10.2), W("is", 10.25, 10.4), W("insane!", 10.5, 11.2)]},
]


def test_weights_redistribute_without_chat():
    w = effective_weights(ScoringWeights(), chat_available=False)
    assert "chat_activity" not in w and abs(sum(w.values()) - 1) < 1e-9
    w2 = effective_weights(ScoringWeights(), chat_available=True)
    # proportional: ratios between remaining weights are preserved
    assert abs(w["hook_strength"] / w["entertainment"] - w2["hook_strength"] / w2["entertainment"]) < 1e-9


def test_total_and_penalties():
    sub = {k: 80 for k in ("hook_strength", "entertainment", "payoff", "emotional_intensity", "unexpectedness",
                           "contextual_clarity", "shareability", "audio_excitement")}
    sub["chat_activity"] = None
    assert total_score(sub, ScoringWeights()) == 80.0
    p = penalties(duration=75, peak_offset=20, starts_mid=True, cfg=ClipSettings())
    assert set(p) == {"too_long", "slow_start", "mid_sentence_start"}
    assert total_score(sub, ScoringWeights(), p) < 80 - 15
    assert penalties(30, 4, False, ClipSettings()) == {}


def test_rejections_and_dedupe():
    cfg = ClipSettings()
    base = {"signals": {"silence_ratio": 0.1, "words": 30, "audio_peak": 0.8, "repetition": 0},
            "sub": {"contextual_clarity": 80, "payoff": 70}, "total": 80, "ai": {"keep": True}}
    assert rejection_reason(base, cfg) is None
    assert "AI rejected" in rejection_reason({**base, "ai": {"keep": False, "reject_reason": "filler"}}, cfg)
    assert "dead air" in rejection_reason({**base, "signals": {**base["signals"], "silence_ratio": 0.9}}, cfg)
    assert "payoff" in rejection_reason({**base, "sub": {"contextual_clarity": 80, "payoff": 10}}, cfg)
    a = {"start": 10, "end": 40, "total": 90}
    b = {"start": 15, "end": 45, "total": 70}
    c = {"start": 100, "end": 130, "total": 60}
    assert dedupe([b, a, c]) == [a, c]


def test_sentences_split_long_segments():
    u = sentences(SEGS)
    assert [x["text"] for x in u] == ["Okay let's drive.", "Wait.", "What is that?", "No way!", "That is insane!"]


def test_snap_keeps_payoff_and_sentence_boundaries():
    u = sentences(SEGS)
    w = snap(u, start_hint=4.5, end_hint=9.5, must_include=(9.3, 9.3), min_dur=5, max_dur=60, media_duration=14)
    assert w.start <= 4.0 and w.end >= 9.8 and not w.starts_mid_sentence
    # no word may be cut by either edge
    for word in SEGS[0]["words"]:
        assert not (word["start"] < w.end < word["end"]) and not (word["start"] < w.start < word["end"])
        assert not (word["start"] < w.end and word["end"] > w.end and word["start"] >= 9.8 and word["start"] < w.end)


def test_hook_first_start_skips_calm_setup():
    u = sentences(SEGS)
    c = {"start": 0.0, "end": 11.5, "peak": 9.5}
    moments.hook_first_start(c, u, ClipSettings(min_duration=5), 14.0)
    assert 3.7 <= c["start"] <= 4.0           # starts on "Wait." not on "Okay let's drive."


def test_jump_cuts_and_remap():
    words = SEGS[0]["words"]
    keep = keep_intervals(words, 0.0, 12.0, max_pause=0.6)
    out_dur = sum(b - a for a, b in keep)
    assert out_dur < 12.0 - 5.0               # the 3 s and 3.6 s pauses are mostly gone
    # every word survives and stays in order on the new timeline
    mapped = [remap(w["start"], keep) for w in words]
    assert all(m is not None for m in mapped) and mapped == sorted(mapped)
    # protected region (e.g. a scream with no words) is not cut
    keep2 = keep_intervals(words, 0.0, 12.0, max_pause=0.6, protect=[(6.0, 8.0)])
    assert remap(7.0, keep2) is not None


def test_loud_gaps_are_content_not_dead_air():
    words = SEGS[0]["words"]
    # the 5.4-9.0 s gap is a loud gunfight (not quiet); the 1.0-4.0 s gap is silence
    loud = lambda a, b: a >= 5.0 and b <= 9.5  # noqa: E731
    keep = keep_intervals(words, 0.0, 12.0, max_pause=0.6, quiet=lambda a, b: not loud(a, b))
    assert remap(7.0, keep) is not None          # gunfight kept
    assert remap(2.5, keep) is None              # silence removed


def test_captions_follow_word_timing():
    words = [{"start": 0.5, "end": 0.9, "word": "No"}, {"start": 1.0, "end": 1.4, "word": "way!"}]
    ass = build_ass(words, 5.0, CaptionStyle(), 1080, 1920, "split", 0.38, hook="He was not ready",
                    sound_tag=("♪ A – B", 2.0, 4.0))
    lines = [l for l in ass.splitlines() if l.startswith("Dialogue: 1")]
    assert lines[0].split(",")[1] == "0:00:00.50" and lines[1].split(",")[1] == "0:00:01.00"
    assert "NO" in lines[0] and "WAY!" in lines[1]
    assert "Hook" in ass and "♪ A – B" in ass


def test_heuristic_rubric_rewards_callouts():
    sig = {"surprise_hits": 3, "hype_hits": 1, "laugh_hits": 0, "callout_hits": 1, "audio_peak": 0.9, "audio_mean": 0.7,
           "laughter_like": 0.0, "scream_like": 0.2, "silence_ratio": 0.1, "words": 40, "hook_audio": 0.8,
           "hook_question": True, "hook_text": "Wait what is that", "peak_offset_s": 5, "contrast": 0.6, "z_rise_max": 4}
    hi = moments.heuristic_eval({"signals": sig, "start": 0, "end": 30, "peak": 5})
    lo = moments.heuristic_eval({"signals": {**sig, "surprise_hits": 0, "hype_hits": 0, "callout_hits": 0, "audio_peak": 0.3,
                                             "audio_mean": 0.2, "hook_audio": 0.1, "hook_question": False, "hook_text": "so anyway",
                                             "contrast": 0.0, "z_rise_max": 0}, "start": 0, "end": 30, "peak": 5})
    for k in moments.RUBRIC:
        assert hi[k] >= lo[k]
    assert hi["hook_strength"] > 60 and lo["hook_strength"] < 40
