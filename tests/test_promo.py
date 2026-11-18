"""Music promo (song-first) mode: chorus on the biggest hit, beat-synced cuts, pre-drop dropout, lyric captions."""
import time

import numpy as np
import pytest

from engine import profiles
from engine.captions.ass import build_ass
from engine.clipping.boundaries import remap
from engine.config import CaptionStyle
from engine.db import Database
from engine.music import mixing, promo
from engine.music.library import analyze
from engine.music.placement import place_custom


@pytest.fixture(scope="module")
def song_analysis(our_song):
    return analyze(our_song)


def test_music_promo_profile():
    p = profiles.get("music_promo")
    assert p.song_first and p.trailer_music and not p.require_speech
    assert abs(sum(p.weights.values()) - 1.0) < 1e-6
    assert p.weights["audio_excitement"] == max(p.weights.values())
    assert any(d["key"] == "music_promo" and d["song_first"] for d in profiles.describe())


def test_window_is_cut_around_the_biggest_hit():
    t = np.arange(0, 120, 0.5)
    score = np.full(t.size, 0.1)
    score[(t >= 40) & (t < 42)] = 0.5          # a smaller bump
    score[(t >= 70) & (t < 71.5)] = 0.95       # the explosion
    hit, strength = promo.biggest_event(t, score, 30, 90)
    assert 69.0 <= hit <= 70.5 and strength > 0.9
    start, end = promo.reshape(hit, 7.0, 12.0, 120.0)
    assert start == pytest.approx(hit - 7.0) and end == pytest.approx(hit + 12.0)
    assert promo.reshape(3.0, 7.0, 12.0, 120.0)[0] == 0.0     # near the start: slides, never negative


def test_music_fit_prefers_quiet_big_hits():
    talky = [{"start": 50 + i * 0.4, "end": 50 + i * 0.4 + 0.35, "word": "something"} for i in range(30)]
    shouts = [{"start": 50 + i * 0.4, "end": 50 + i * 0.4 + 0.35, "word": "NOOO"} for i in range(30)]
    quiet, _ = promo.music_fit(50.0, 0.9, shouts, [], (43, 62), 7, 12)
    loud_talk, parts = promo.music_fit(50.0, 0.9, talky, [], (43, 62), 7, 12)
    with_radio, _ = promo.music_fit(50.0, 0.9, shouts, [(40, 65)], (43, 62), 7, 12)
    assert quiet > loud_talk and parts["clear"] < 50
    assert quiet > with_radio


def test_window_lengths_use_marked_hook(song_analysis):
    a = dict(song_analysis)
    a["hook"] = {**a["hook"], "user": True, "end": a["hook"]["downbeat"] + 13.0}
    build, after = promo.window_lengths(a, 4)
    assert build == pytest.approx(min(10.0, 4 * a["bar_s"]), abs=0.01)
    assert after == pytest.approx(13.0, abs=0.01)


def test_chorus_downbeat_lands_on_the_hit(song_analysis):
    a = song_analysis
    hit = 8.3
    p = place_custom(a, 22.0, 0.0, hit, [], prefer_text="", min_window=6.0, prefer_hook=True)
    assert p is not None
    assert p.payoff_sync["clip_time"] == pytest.approx(hit)
    assert p.payoff_sync["song_downbeat"] == pytest.approx(a["hook"]["downbeat"], abs=1e-3)
    # the song position under the hit is exactly the hook downbeat
    assert p.song_in + (hit - p.clip_in) == pytest.approx(a["hook"]["downbeat"], abs=1e-3)


def test_beat_snapped_cuts_move_nothing_else():
    # source timeline: 4 pieces of speech with removed gaps between them
    keep = [(10.0, 13.1), (14.0, 17.37), (18.5, 21.0), (22.0, 26.0)]
    words = [{"start": a + 0.2, "end": b - 0.2, "word": "word"} for a, b in keep]
    payoff = 19.5
    beats = np.arange(0, 20, 0.25)                         # 120 bpm eighth notes on the output timeline
    new, shifts = promo.snap_cuts_to_beats(keep, beats, words, (0, 20), fixed=[payoff])
    assert shifts, "at least one cut should move"
    total = lambda k: sum(b - a for a, b in k)            # noqa: E731
    assert total(new) == pytest.approx(total(keep), abs=1e-6)
    for w in words:                                        # every word plays at the same output time
        assert remap(w["start"], new) == pytest.approx(remap(w["start"], keep), abs=1e-6)
    assert remap(payoff, new) == pytest.approx(remap(payoff, keep), abs=1e-6)
    acc = 0.0
    for a, b in new[:-1]:                                  # moved cuts sit on the grid
        acc += b - a
        assert min(abs(beats - acc)) < 0.13
    assert sum(1 for k in range(len(new) - 1) if min(abs(beats - sum(b - a for a, b in new[:k + 1]))) < 1e-3) >= len(shifts)


def test_snap_never_clips_a_word():
    keep = [(0.0, 3.1), (4.0, 7.0)]
    words = [{"start": 4.0, "end": 4.5, "word": "hello"}, {"start": 2.0, "end": 3.1, "word": "yo"}]
    beats = np.array([3.2])                                 # would need piece 2 to start 0.1 s into "hello"
    new, shifts = promo.snap_cuts_to_beats(keep, beats, words, (0, 10))
    assert shifts == [] and new == keep


def test_dropout_silences_the_beat_before_the_chorus(our_song, song_analysis):
    sr = mixing.SR
    dur, hit = 20.0, 9.0
    voice = (np.random.default_rng(0).standard_normal((int(dur * sr), 2)) * 0.05).astype(np.float32)   # game noise
    p = place_custom(song_analysis, dur, 0.0, hit, [], prefer_hook=True).to_dict()
    win = promo.dropout_window(song_analysis, p)
    assert win is not None and win[1] == pytest.approx(hit)
    assert win[1] - win[0] == pytest.approx(min(0.5, 60 / song_analysis["bpm"]), abs=1e-3)
    plan = mixing.SongPlan(our_song, p["song_in"], p["song_out"], p["clip_in"], 0.0, p["fade_out"], 1.0, 7.0, 120, 400,
                           carve=False, bed_db=-8.0, dropout=win)
    mixed, _ = mixing.mix(voice, [], plan)
    rms = lambda a, b: float(np.sqrt(np.mean(mixed[int(a * sr):int(b * sr)] ** 2)))   # noqa: E731
    inside = rms(win[0] + 0.03, win[1] - 0.01)
    before = rms(win[0] - 0.6, win[0] - 0.1)
    after = rms(hit + 0.01, hit + 0.4)
    assert 20 * np.log10(before / inside) > 20          # the drop is real silence
    assert after > before * 0.5                          # and the chorus comes straight back


def test_no_dropout_without_a_synced_drop(song_analysis):
    assert promo.dropout_window(song_analysis, {"clip_in": 0, "song_in": 0, "song_out": 10, "payoff_sync": None}) is None


def test_lyrics_on_clip_timeline_and_after_hook_card():
    lyr = [{"start": 30.0, "end": 34.0, "text": "we ride all night under city lights again and again"},
           {"start": 50.0, "end": 52.0, "text": "outside the window"}]
    pl = {"song_in": 28.0, "song_out": 45.0, "clip_in": 1.0}
    lines = promo.lyric_lines(lyr, pl, max_words=6)
    assert [l[0] for l in lines] == ["we ride all night under city", "lights again and again"]
    assert lines[0][1] == pytest.approx(3.0) and lines[-1][2] == pytest.approx(7.0)
    ass = build_ass([], 20.0, CaptionStyle(hook_seconds=3.5), 1080, 1920, "gameplay_only", 0.38, hook="Big crash",
                    lyrics=lines)
    lyric_events = [l for l in ass.splitlines() if ",Lyric," in l]
    assert len(lyric_events) == 2
    assert lyric_events[0].split(",")[1] == "0:00:03.50"            # pushed past the hook card
    assert "\\an5\\pos(" in lyric_events[0] and "\a" not in ass and "\f" not in ass


def test_choose_song_prefers_job_then_campaign_then_priority(tmp_path, our_song, song_analysis):
    from engine.config import Settings
    db = Database(tmp_path / "p.db")
    aid = db.insert("artists", {"name": "A", "created_at": time.time()})
    ids = {}
    for slug, pr in (("low", 1), ("high", 5), ("camp", 2)):
        ids[slug] = db.insert("songs", {"slug": slug, "artist_id": aid, "title": slug, "file_path": str(our_song),
                                        "analysis_json": song_analysis, "campaign_priority": pr, "created_at": time.time()})
        db.insert("song_rights_confirmations", {"song_id": ids[slug], "statement": "mine", "confirmed_at": time.time()})
    s = Settings()
    assert promo.choose_song(db, s)["slug"] == "high"
    s.music.campaign_song = "camp"
    assert promo.choose_song(db, s)["slug"] == "camp"
    assert promo.choose_song(db, s, "low")["slug"] == "low"


def test_smooth_ducking_holds_a_steady_level_through_on_and_off_talking():
    sr = mixing.SR
    talk = [(1.0, 2.2), (3.0, 4.1), (5.0, 6.0), (6.9, 8.0)]          # talking with ~1 s pauses
    tight = mixing.duck_envelope(int(10 * sr), talk, 12.0, 120, 400)
    merged = mixing.merge_speech(talk, 2.5)
    assert merged == [(1.0, 8.0)]
    smooth = mixing.duck_envelope(int(10 * sr), merged, 12.0, 120, 900)
    seg = lambda e: 20 * np.log10(e[int(2.0 * sr):int(7.5 * sr)])     # noqa: E731  inside the talking stretch
    assert np.ptp(seg(tight)) > 6.0          # the old way pumps up in every pause
    assert np.ptp(seg(smooth)) < 0.5         # smooth: one steady level
    assert smooth[int(9.8 * sr)] > 0.85      # and the song comes back after the talking ends


def test_ai_brief_follows_the_live_settings(song_analysis):
    from engine.config import Settings
    s = Settings()
    b = promo.ai_brief("Tweakin", "Moqemae", song_analysis, 6.9, 13.0, s, song_first=True)
    assert "Tweakin" in b and "silence" in b.lower() and "lyrics" in b.lower() and "steady" in b
    s.music.pre_drop_dropout, s.captions.lyrics, s.music.ducking = "off", "off", "tight"
    b2 = promo.ai_brief("Tweakin", "Moqemae", song_analysis, 6.9, 13.0, s, song_first=True)
    assert "silence" not in b2.lower() and "lyrics" not in b2.lower() and "dips under each" in b2


# ---------------------------------------------------------------- speed ramps and beat effects
def test_ramps_shift_by_whole_beats_and_keep_the_hit_synced():
    beat = 60 / 140
    keep = [(20.0, 45.0)]
    hit = 31.0
    pieces, info = promo.ramp_plan(keep, hit, beat, [])
    assert set(info) == {"speed_up", "slow_mo"}
    assert promo.out_len(pieces) == pytest.approx(25.0, abs=1e-6)        # -1 beat before the hit, +1 beat after
    assert promo.src_to_out(hit, pieces) == pytest.approx(11.0 - beat, abs=1e-6)
    later = 40.0                                                          # after both ramps: back exactly where it was
    assert promo.src_to_out(later, pieces) == pytest.approx(later - 20.0, abs=1e-6)
    pl = {"song_in": 10.0, "song_out": 30.0, "clip_in": 0.0, "clip_out": 20.0, "mode": "moment_drop",
          "payoff_sync": {"clip_time": 11.0, "song_downbeat": 21.0}}
    new = promo.resync_placement(pl, keep, pieces, 25.0)
    hit_out = new["payoff_sync"]["clip_time"]
    assert hit_out == pytest.approx(11.0 - beat, abs=1e-4)
    # the song position under the hit is still the chorus downbeat
    assert new["song_in"] + (hit_out - new["clip_in"]) == pytest.approx(21.0, abs=1e-4)
    assert new["song_out"] <= 30.0 + 1e-6


def test_ramps_skip_talking_and_cuts():
    beat = 0.5
    words = [{"start": 30.0 + i * 0.3, "end": 30.25 + i * 0.3, "word": "something"} for i in range(4)]
    pieces, info = promo.ramp_plan([(20.0, 45.0)], 31.0, beat, words)
    assert "speed_up" not in info                                         # someone is talking right before the hit
    pieces, info = promo.ramp_plan([(20.0, 30.5), (31.5, 40.0)], 31.6, beat, [])
    assert "speed_up" not in info and "slow_mo" in info                   # would cross the jump cut


def test_next_hit_is_nudged_onto_a_bar_without_moving_anything_after():
    bar = 2.0
    pieces = promo.pieces_from_keep([(0.0, 30.0)])
    hit_out = 5.0
    pieces, done = promo.nudge_hits(pieces, [8.8], hit_out, bar, [], after_src=5.5)
    far, none = promo.nudge_hits(promo.pieces_from_keep([(0.0, 30.0)]), [8.5], hit_out, bar, [], after_src=5.5)
    assert none == []                                                     # would need >12% speed change: left alone
    assert len(done) == 1
    assert promo.src_to_out(8.8, pieces) == pytest.approx(9.0, abs=1e-4)          # landed on the bar (5 + 2*2)
    assert promo.src_to_out(20.0, pieces) == pytest.approx(20.0, abs=1e-4)         # everything later untouched
    assert all(0.88 <= s <= 1.12 for _, _, s in pieces)


def test_audio_follows_the_speed_of_each_piece():
    sr = mixing.SR
    a = np.zeros((int(4 * sr), 2), np.float32)
    out = mixing.apply_cuts(a, [(0.0, 2.0), (2.0, 3.0), (3.0, 4.0)], speeds=[1.0, 0.5, 2.0])
    assert out.shape[0] / sr == pytest.approx(2.0 + 2.0 + 0.5, abs=0.02)


def test_beat_fx_filter_runs(tmp_path, song_analysis):
    import subprocess
    from engine.video.compose import beat_fx_chain
    fx = promo.fx_spec(song_analysis, {"clip_in": 0.0, "clip_out": 3.0, "song_in": 10, "song_out": 13,
                                       "payoff_sync": {"clip_time": 1.0, "song_downbeat": 11.0}})
    assert fx and fx["beat"] == pytest.approx(60 / song_analysis["bpm"], abs=1e-4)
    chain = beat_fx_chain(fx, 270, 480)
    out = tmp_path / "fx.mp4"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=270x480:rate=30:duration=2",
                        "-vf", chain + ",format=yuv420p", "-c:v", "libx264", "-preset", "ultrafast", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    def brightness(t):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", "-f", "rawvideo",
                              "-pix_fmt", "gray", "-"], capture_output=True).stdout
        return np.frombuffer(raw, np.uint8).mean()
    assert brightness(1.0) > brightness(0.8) + 20                         # the flash on the chorus hit
