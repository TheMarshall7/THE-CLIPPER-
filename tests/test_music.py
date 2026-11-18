"""Music module acceptance tests (M13) on authorized synthetic audio."""
import json
import time

import numpy as np
import pytest

from engine.db import Database
from engine.music import detect, mixing
from engine.music.library import analyze
from engine.music.match import clip_music_block, song_ineligible
from engine.music.placement import place, preferred_modes


@pytest.fixture(scope="module")
def song_analysis(our_song):
    return analyze(our_song)


# ---------------------------------------------------------------- song analysis
def test_song_analysis_matches_ground_truth(our_song, song_analysis):
    truth = json.loads(our_song.with_suffix(".truth.json").read_text())
    a = song_analysis
    assert abs(a["bpm"] - truth["bpm"]) < 0.5
    tb = np.array(truth["bars"])
    err = [np.min(np.abs(tb - b)) for b in a["bars"] if b < tb[-1]]
    assert np.median(err) < 0.02 and max(err) < 0.05          # bar grid within 20/50 ms
    first_chorus = next(s for s in truth["sections"] if s["label"] == "chorus")["start"]
    assert abs(a["hook"]["downbeat"] - first_chorus) < 0.05   # hook = first chorus downbeat
    assert all(p["contains_hook"] for p in a["promo_segments"] if p["length"] >= 15)


# ---------------------------------------------------------------- source-music detection (M13 #1, #2)
def test_car_radio_on_is_detected(samples):
    tl = detect.music_timeline(samples["sample_radio"]["wav"])
    regs = detect.regions(tl, 0, 150, 0.55)
    assert regs, "radio music not detected"
    assert regs[0]["start"] <= 2.0 and 30.0 <= regs[0]["end"] <= 38.0
    assert all(r["start"] < 38 for r in regs), "speech after the radio stopped was flagged as music"


def test_car_radio_off_not_flagged(samples):
    tl = detect.music_timeline(samples["sample_stream"]["wav"])
    assert detect.regions(tl, 0, 150, 0.55) == []
    assert not detect.stream_baseline(tl, 0.55)["constant_bgm"]


def test_constant_bgm_flagged_consistently():  # M13 #3
    tl = {"t": np.arange(0, 600, 0.5), "prob": np.full(1200, 0.8)}
    base = detect.stream_baseline(tl, 0.55)
    assert base["constant_bgm"]
    regs = detect.regions(tl, 100, 140, 0.55, base["constant_bgm"])
    assert regs and regs[0]["type"] == "streamer_bgm"
    assert "background music" in clip_music_block(None, 1.0, False, True)


def test_singing_along_blocks_song():  # M13 #4
    regs = [{"start": 0, "end": 20}]
    words = [{"word": w, "start": i, "end": i + 0.4} for i, w in enumerate("oh I love this song turn it up".split())]
    assert detect.music_is_content(words, regs)
    assert "part of the moment" in clip_music_block(None, 0.5, True, False)


# ---------------------------------------------------------------- placement
def test_payoff_drop_lands_on_downbeat_and_bars(song_analysis):  # M13 #6
    payoff = 12.3
    p = place(song_analysis, clip_dur=30.0, payoff=payoff, conflicts=[], speech=[(0, 11)], modes=["payoff_drop"])
    assert p and p.mode == "payoff_drop"
    # the song time playing at the payoff is the hook downbeat
    song_t_at_payoff = p.song_in + (payoff - p.clip_in)
    assert abs(song_t_at_payoff - song_analysis["hook"]["downbeat"]) < 0.05
    bars = np.array(song_analysis["bars"])
    assert np.min(np.abs(bars - p.song_in)) < 0.03 and np.min(np.abs(bars - p.song_out)) < 0.03


def test_placement_avoids_source_music(song_analysis):  # M13 #1
    # radio plays during the first 18 s; payoff at 12 s is inside it -> drop impossible, tail allowed
    p = place(song_analysis, 30.0, 12.0, conflicts=[(0.0, 18.4)], speech=[(0, 20)], modes=["payoff_drop", "reaction_tail"])
    assert p is not None and p.clip_in >= 18.4
    # music everywhere -> nothing
    assert place(song_analysis, 30.0, 12.0, [(0.0, 30.0)], [(0, 20)], ["payoff_drop", "reaction_tail", "underscore"]) is None


def test_dialogue_prefers_reaction_tail():  # M13 #5
    assert preferred_modes("funny", 3.0, 0.4)[0] == "reaction_tail"
    assert preferred_modes("crazy_gameplay", 1.5, 0.9)[0] == "payoff_drop"


# ---------------------------------------------------------------- mixing
def test_ducking_depth_and_loudness(tmp_path, our_song, song_analysis):  # M7, M13 #10
    sr = mixing.SR
    dur = 20.0
    t = np.arange(int(dur * sr)) / sr
    voice = np.zeros((t.size, 2), np.float32)
    speech = [(2.0, 6.0), (12.0, 14.0)]
    for a, b in speech:  # speech-like bursts
        m = (t >= a) & (t < b)
        voice[m] = (0.3 * np.sin(2 * np.pi * 220 * t[m]) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t[m])))[:, None]
    src = mixing.write_wav(tmp_path / "voice.wav", voice)
    p = place(song_analysis, dur, 8.0, [], speech, ["payoff_drop"])
    plan = mixing.SongPlan(our_song, p.song_in, p.song_out, p.clip_in, p.fade_in, p.fade_out, -10.0, 15.0, 120, 400, carve=False)
    mixed, _ = mixing.mix(voice, speech, plan)
    song_only = mixed - voice
    rms = lambda a, b: 20 * np.log10(np.sqrt(np.mean(song_only[int(a * sr):int(b * sr)] ** 2)) + 1e-12)  # noqa: E731
    assert rms(15.0, 17.0) - rms(12.5, 13.5) >= 12.0       # ducked >= 12 dB under speech
    res = mixing.render_tracks(src, 0.0, dur, [(0.0, dur)], speech, tmp_path, -14.0, -1.0, plan)
    from engine.media.ffmpeg import measure_loudness
    loud = measure_loudness(tmp_path / "mixed.wav")
    assert abs(loud["integrated_lufs"] + 14.0) <= 1.0 and loud["true_peak_db"] <= -0.5
    assert (tmp_path / "clean.wav").exists() and res.get("mixed")


def test_rendered_drop_onset_within_50ms(tmp_path, our_song, song_analysis):  # M13 #6, measured on audio
    sr = mixing.SR
    dur, payoff = 24.0, 10.0
    voice = np.zeros((int(dur * sr), 2), np.float32)
    p = place(song_analysis, dur, payoff, [], [], ["payoff_drop"])
    plan = mixing.SongPlan(our_song, p.song_in, p.song_out, p.clip_in, 0.0, p.fade_out, -10.0, 15.0, 120, 400, carve=False)
    mixed, _ = mixing.mix(voice, [], plan)
    x = mixed[:, 0]
    win = int(0.003 * sr)
    seg = x[int((payoff - 0.2) * sr): int((payoff + 0.2) * sr)]
    env = np.sqrt(np.convolve(seg ** 2, np.ones(win) / win, mode="same"))
    rise = np.diff(np.log(env + 1e-6))
    onset_t = payoff - 0.2 + (int(np.argmax(rise)) / sr)
    assert abs(onset_t - payoff) < 0.05


# ---------------------------------------------------------------- eligibility rules
def _db_with_song(tmp_path, our_song, analysis):
    db = Database(tmp_path / "m.db")
    aid = db.insert("artists", {"name": "A", "avoid_moments_json": ["tragedy"], "created_at": time.time()})
    sid = db.insert("songs", {"slug": "s1", "artist_id": aid, "title": "S1", "file_path": str(our_song),
                              "analysis_json": analysis, "created_at": time.time()})
    return db, aid, sid


def test_rights_brand_rotation(tmp_path, our_song, song_analysis):  # M10, M13 #7, #9
    db, aid, sid = _db_with_song(tmp_path, our_song, song_analysis)
    song = db.one("SELECT * FROM songs WHERE id=?", (sid,))
    artist = db.one("SELECT * FROM artists WHERE id=?", (aid,))
    assert song_ineligible(song, artist, "x", "funny", False, db, 10, 3, None) == "rights not confirmed"
    db.insert("song_rights_confirmations", {"song_id": sid, "confirmed_by": "t", "statement": "own", "confirmed_at": 0})
    assert song_ineligible(song, artist, "a real tragedy happened", "funny", False, db, 10, 3, None).startswith("brand avoid")
    assert song_ineligible(song, artist, "chase", "crazy_gameplay", False, db, 10, 3, None) is None
    for _ in range(3):
        db.insert("song_usage", {"song_id": sid, "clip_id": None, "used_at": 0})
    assert "rotation" in song_ineligible(song, artist, "chase", "crazy_gameplay", False, db, 10, 3, None)
    assert song_ineligible(song, artist, "chase", "crazy_gameplay", False, db, 10, 3, "s1") is None  # campaign mode


def test_program_restriction_blocks_music():  # M13 #8
    streamer = {"allows_added_music": 0, "allows_third_party_promotion": None, "permission_status": "official_clipping_program"}
    assert "does not allow added music" in clip_music_block(streamer, 0.0, False, False)
    streamer = {"allows_added_music": None, "allows_third_party_promotion": None, "permission_status": "official_clipping_program"}
    assert "program rules unknown" in clip_music_block(streamer, 0.0, False, False)


# ---------------------------------------------------------------- trailer placement
def test_reveal_prefers_cut_with_audio_build():
    from engine.video.cuts import pick_reveal
    cuts = [(130.77, 0.88), (133.39, 0.81), (133.89, 0.48), (135.89, 0.53)]
    rises = {130.77: 0.6, 133.39: 0.9, 133.89: 2.1, 135.89: 1.8}
    t = pick_reveal(cuts, 130.0, lambda x: rises[x])
    assert t == 133.89                                   # biggest picture change + build, not just the first cut
    assert pick_reveal([], 130.0, lambda x: 0) == 130.0   # no cuts -> keep the audio peak


def test_auto_entry_starts_under_dialogue_when_quiet_entry_is_too_late(song_analysis):
    from engine.music.placement import auto_entry, place_custom
    bar = song_analysis["bar_s"]
    speech = [(2, 6), (9, 12), (16, 19), (25, 27), (34, 36)]        # talk almost everywhere before the payoff
    payoff = 44.0
    e = auto_entry(payoff, speech, [], bar)
    assert e is not None and payoff - e >= 12.0                      # long build, not a 4 s stub
    p = place_custom(song_analysis, 52.0, e, payoff, [], prefer_text="x")
    assert p and p.clip_out - p.clip_in >= 12.0
    # never starts before source music that sits between the entry and the payoff
    e2 = auto_entry(payoff, speech, [(20.0, 30.0)], bar)
    assert e2 is None or e2 >= 30.0


def test_trailer_mix_is_loud_and_dialogue_stays_clear(tmp_path, our_song, song_analysis):
    sr = mixing.SR
    dur = 20.0
    t = np.arange(int(dur * sr)) / sr
    voice = np.zeros((t.size, 2), np.float32)
    speech = [(2.0, 5.0)]
    m = (t >= 2.0) & (t < 5.0)
    voice[m] = (0.3 * np.sin(2 * np.pi * 220 * t[m]))[:, None]
    voice += (0.05 * np.sin(2 * np.pi * 90 * t))[:, None].astype(np.float32)    # game ambience everywhere
    p = place(song_analysis, dur, 12.0, [], speech, ["payoff_drop"])
    pre = mixing.PRESETS["trailer"]
    plan = mixing.SongPlan(our_song, p.song_in, p.song_out, p.clip_in, p.fade_in, p.fade_out, pre["level_db"], pre["duck_db"],
                           120, 400, carve=False, bed_db=pre["bed_db"])
    mixed, stats = mixing.mix(voice, speech, plan)
    song_part = mixed[int(14 * sr):int(16 * sr)]
    speech_ref = voice[int(2.5 * sr):int(4.5 * sr)]
    rms = lambda a: 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-12)   # noqa: E731
    assert rms(song_part) >= rms(speech_ref) - 3                       # song at roughly dialogue level (trailer)
    assert stats["song_gain_db"] > mixing.PRESETS["balanced"]["level_db"] - 30
