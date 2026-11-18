"""Review / posting / watch folder / progress / placement-marker features."""
import json
import os
import time

import numpy as np

from engine import posting, progress, review
from engine.config import Settings
from engine.db import Database
from engine.music import library
from engine.music.placement import place, place_custom
from engine.video.compose import _lerp_expr
from engine.watch import scan_once


def _db(tmp_path):
    d = Database(tmp_path / "t.db")
    d.execute("INSERT INTO streams(id,platform,source_url,title,created_at,updated_at) VALUES (1,'file','x','S',0,0)")
    for i in range(3):
        d.execute("INSERT INTO clips(id,stream_id,start,end,score,status,video_path,created_at,updated_at) "
                  "VALUES (?,?,?,?,?,?,?,?,?)", (i + 1, 1, 0, 10 + i * 10, 50 + i * 20, "rendered", "v.mp4", 0, 0))
    return d


def test_auto_review_rules_and_human_decisions_win(tmp_path):
    d = _db(tmp_path)
    s = Settings()
    s.review.min_score = 60
    s.review.min_seconds = 15
    res = review.auto_review(d, s)
    assert res["rejected"] == 1 and "score under 60" in res["by_rule"]
    review.decide(d, 1, "keep")                      # a human keeps what the rules rejected
    assert review.auto_review(d, s)["rejected"] == 0
    assert d.one("SELECT approval FROM clips WHERE id=1")["approval"] == "approved"


def test_review_decisions_categories_and_undo(tmp_path):
    d = _db(tmp_path)
    review.decide(d, 2, "post")
    review.decide(d, 3, "reject", "No payoff")
    c = review.counts(d)
    assert c["post"] == 1 and c["reject"] == 1 and c["reasons"] == {"No payoff": 1} and c["waiting"] == 1
    review.undo(d, 3)
    assert review.counts(d)["waiting"] == 2 and d.one("SELECT approval FROM clips WHERE id=3")["approval"] == "pending"


def test_posting_flow_and_learned_times(tmp_path):
    d = _db(tmp_path)
    review.decide(d, 1, "post")
    assert [c["id"] for c in posting.to_post(d)] == [1]
    review.mark_posted(d, 1)
    assert posting.to_post(d) == [] and len(posting.to_post(d, posted=True)) == 1
    assert "general guidance" in posting.best_times(d)["tiktok"]
    for cid in (2, 3):                                # learned hours need 4 logged results
        review.decide(d, cid, "post")
        review.mark_posted(d, cid)
    d.execute("INSERT INTO clips(id,stream_id,start,end,status,created_at,updated_at) VALUES (4,1,0,9,'rendered',0,0)")
    review.decide(d, 4, "post")
    review.mark_posted(d, 4)
    for cid in (1, 2, 3, 4):
        posting.record_result(d, cid, "tiktok", 500 * cid)
    assert "your best hours" in posting.best_times(d)["tiktok"]


def test_watch_folder_waits_for_stable_files_and_queues_once(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x" * 50)
    os.utime(f, (time.time() - 60, time.time() - 60))
    (tmp_path / "links.txt").write_text("https://youtu.be/abc\n# comment\n")
    got, seen = [], {}
    assert scan_once(tmp_path, seen, got.append) == 1          # the link goes immediately, the file waits one scan
    assert scan_once(tmp_path, seen, got.append) == 1 and len(got) == 2
    assert scan_once(tmp_path, seen, got.append) == 0


def test_progress_estimates_from_clip_rate(tmp_path):
    d = _db(tmp_path)
    d.execute("DELETE FROM clips")
    d.execute("INSERT INTO jobs(kind,stream_id,status,created_at) VALUES ('process_stream',1,'running',0)")
    for s in progress.STEPS[:-1]:
        d.execute("INSERT INTO stream_steps(stream_id,step,status,duration_s) VALUES (1,?,'done',10)", (s,))
    now = time.time()
    for i in range(20):
        done = i < 5
        d.execute("INSERT INTO clips(stream_id,start,end,status,created_at,updated_at) VALUES (1,0,1,?,?,?)",
                  ("rendered" if done else "pending", now - 100, now - 100 + (i + 1) * 10 if done else now))
    snap = progress.snapshot(d)
    assert snap["clips_done"] == 5 and snap["clips_total"] == 20 and snap["eta_s"] and snap["eta_s"] > 0
    assert "Rendering clips 5/20" in snap["headline"]


def test_user_marked_hook_is_never_played_past(tmp_path):
    song = {"bars": list(np.arange(0, 200, 2.0)), "bar_s": 2.0, "duration": 200, "lyrics": [],
            "hook": {"start": 40.0, "end": 52.0, "downbeat": 40.0, "user": True},
            "sections": [{"label": "chorus", "start": 40, "end": 52, "is_hook": True}]}
    p = place(song, 30, 10, [], [(0, 5)], ["payoff_drop", "reaction_tail"])
    assert p and p.song_out <= 52.0
    c = place_custom(song, 30, 3, 10, [], "")
    assert c and c.song_out <= 52.0


def test_set_hook_overrides_analysis(tmp_path):
    d = Database(tmp_path / "t.db")
    a = {"bars": list(np.arange(0, 100, 2.0)), "duration": 100, "hook": {"start": 10, "end": 20, "downbeat": 10},
         "sections": [{"label": "verse", "start": 0, "end": 30, "energy": .5, "vocals": 1, "is_hook": True}]}
    d.execute("INSERT INTO songs(id,slug,title,file_path,analysis_json,created_at) VALUES (1,'s','S','x.wav',?,0)", (json.dumps(a),))
    library.set_hook(d, 1, 41.3, 55.0)
    eff = library.effective_analysis(d.one("SELECT * FROM songs WHERE id=1"))
    assert eff["hook"]["user"] and eff["hook"]["end"] == 55.0
    library.clear_hook(d, 1)
    assert not library.effective_analysis(d.one("SELECT * FROM songs WHERE id=1"))["hook"].get("user")


def test_face_track_expression_is_shallow_and_continuous():
    pts = [(i * 0.5, 100.0 + i * 10) for i in range(40)]
    e = _lerp_expr(pts)
    assert e.count("if(") < 40 and "clip(t," in e


def test_big_moment_spot_ignores_yelling_and_rejects_bad_ai():
    from engine.music import moments
    words = [{"word": w, "start": s, "end": s + 0.3} for w, s in
             [("so", 0.5), ("we're", 0.9), ("driving", 1.3), ("here", 1.7), ("NO", 9.0), ("WAY", 9.3), ("AHHH", 9.7)]]
    events = [{"t": 9.0, "score": 0.9, "voiced": True}, {"t": 1.5, "score": 0.8, "voiced": False}]
    rules = moments.rules_pick(events, words, [], 20.0)
    assert rules and abs(rules["drop"] - 8.85) < 0.01            # the scream, not the event buried in real speech

    class Good:
        def complete_json(self, s, p):
            return {"drop_at": 9.2, "reason": "crash", "confidence": 0.8}

    class Bad:
        def complete_json(self, s, p):
            return {"drop_at": 1.0}                              # too early, inside dialogue

    assert moments.pick_spot(Good(), "fail", words, events, [], 20.0)["by"] == "ai"
    assert moments.pick_spot(Bad(), "fail", words, events, [], 20.0)["by"] == "rules"
    assert moments.pick_spot(None, "fail", words, events, [(8.0, 10.0)], 20.0) is None    # never over the game's own music


def test_title_options_always_three():
    from engine.metadata.generate import generate
    for text in ("", "hi", "Oh my god. No way! What was that?"):
        m = generate({"streamer": None, "platform": "file", "url": "x", "category": "c", "summary": "", "transcript": text,
                      "vertical": "GTA 6"}, None)
        assert len(m["hooks"]) == 3


def test_friendly_errors_and_crash_log(tmp_path, monkeypatch):
    import errno
    from engine import errors

    assert "disk is full" in errors.friendly(OSError(errno.ENOSPC, "no space")).lower()
    assert "FFmpeg was not found" in errors.friendly(FileNotFoundError("ffmpeg"))
    assert "Network problem" in errors.friendly(ConnectionError("down"))
    assert "Unexpected problem" in errors.friendly(IndexError("list index out of range"))
    assert errors.friendly(KeyError("x"))                                  # never empty, never raises
    monkeypatch.setattr(errors, "LOG_PATH", tmp_path / "logs" / "errors.log")
    try:
        1 / 0
    except ZeroDivisionError as e:
        errors.log_exception("unit test", e)
    assert "ZeroDivisionError" in (tmp_path / "logs" / "errors.log").read_text(encoding="utf-8")
    e = ValueError("same problem")
    assert errors.should_notify(e) and not errors.should_notify(e)         # no dialog spam for a repeating error


def test_guidance_keywords_boost_and_penalise():
    from engine.intelligence import moments
    g = "Pick the funniest reactions and big crashes. Skip long setups, no clips about chat"
    assert moments.guidance_bias(g, "haha that was hilarious") > 0
    assert moments.guidance_bias(g, "the car crashed into the wall") > 0
    assert moments.guidance_bias(g, "reading chat for a while about chat") < 0
    assert moments.guidance_bias(g, "walking down the street") == 0
    assert moments.guidance_bias("", "anything") == 0


def test_ai_suggestions_fall_back_to_presets():
    from engine.ai.suggest import PRESETS, ideas

    class Bad:
        def complete_json(self, s, p):
            raise RuntimeError("no network")

    class Good:
        def complete_json(self, s, p):
            return {"ideas": ["Find rage-quit moments", "Prefer clean setups", "x"]}
    assert ideas(None)[:2] == PRESETS[:2] and ideas(Bad())[:2] == PRESETS[:2]
    assert ideas(Good()) == ["Find rage-quit moments", "Prefer clean setups"]       # too-short junk is dropped


def test_user_minimum_length_beats_video_type_default():
    from engine import profiles
    s = Settings()
    s.clips.min_length = 40
    eff = profiles.effective_settings(s, profiles.get("stream"))
    assert eff.clips.min_duration >= 40 and eff.clips.target_max >= 50


def test_cam_zoom_is_capped_and_tiny_cams_detected():
    from engine.config import VideoSettings
    from engine.video.compose import cam_window, cam_zoom_needed
    v = VideoSettings()
    tiny = {"layout": "split", "cam_box": [139, 280, 119, 216], "face_box": [164, 377, 43, 56]}
    assert cam_zoom_needed(tiny, v, 1920, 1080) > 3.0
    assert cam_window(tiny, 1080, 410, 1920, 1080, v.max_cam_zoom)[4] <= v.max_cam_zoom + 0.01


def _seed_learning_db(tmp_path):
    """20 moments: the ones with a strong hook + payoff get kept/posted, weak ones rejected."""
    import json as _j
    d = Database(tmp_path / "learn.db")
    d.execute("INSERT INTO streams(id,platform,source_url,title,created_at,updated_at) VALUES (1,'file','x','S',0,0)")
    for i in range(20):
        good = i % 2 == 0
        sub = {"hook_strength": 85 if good else 25, "payoff": 80 if good else 30, "entertainment": 70 if good else 40,
               "emotional_intensity": 50, "unexpectedness": 50, "contextual_clarity": 60, "shareability": 50,
               "audio_excitement": 60, "chat_activity": None}
        d.execute("INSERT INTO candidates(id,stream_id,start,end,category,scores_json,created_at) VALUES (?,?,?,?,?,?,0)",
                  (i + 1, 1, 0, 30, "funny" if good else "fail", _j.dumps(sub)))
        d.execute("INSERT INTO clips(id,stream_id,candidate_id,start,end,score,status,video_path,created_at,updated_at) "
                  "VALUES (?,?,?,?,?,?,?,?,0,0)", (i + 1, 1, i + 1, 0, 30, 70, "rendered", "v.mp4"))
    return d, sub


def test_swipes_teach_the_algorithm(tmp_path):
    from engine import learn
    d, _ = _seed_learning_db(tmp_path)
    for i in range(1, 9):                                   # 8 swipes: not enough yet
        review.decide(d, i, "keep" if i % 2 == 1 else "reject", None if i % 2 == 1 else "No payoff")
    assert learn.model(d) is None and learn.status(d)["need"] == 4
    for i in range(9, 21):
        review.decide(d, i, "post" if i % 2 == 1 else "reject", None if i % 2 == 1 else "Weak start / boring")
    m = learn.model(d)
    assert m and m["n"] == 20
    strong = {"hook_strength": 90, "payoff": 85, "entertainment": 75, "contextual_clarity": 60}
    weak = {"hook_strength": 20, "payoff": 25, "entertainment": 35, "contextual_clarity": 60}
    assert learn.adjust(d, strong, "funny", 30) > 3 and learn.adjust(d, weak, "fail", 30) < -3
    assert learn.adjust(d, strong, "funny", 30, strength=0) == 0                # switch off = no effect
    st = learn.status(d)
    assert any("hook" in x or "payoff" in x for x in st["likes"]) and st["active"]
    learn.forget(d)
    assert learn.model(d) is None and learn.adjust(d, strong, "funny", 30) == 0
    review.undo(d, 1)                                                           # undo keeps the learner consistent
    assert learn.model(d) is None                                               # forgotten decisions stay forgotten


def test_auto_rejects_do_not_teach(tmp_path):
    from engine import learn
    d, _ = _seed_learning_db(tmp_path)
    for i in range(1, 21):
        review.decide(d, i, "reject", "Auto: score under 99", auto=True)
    assert learn.status(d)["decisions"] == 0 and learn.model(d) is None


def test_videos_with_no_talking_are_treated_as_gameplay():
    from engine import profiles
    kw = ["gta 6"]
    assert profiles.detect("The FINAL Best GTA 5 Moments Of All Time", 0.1, kw, words_per_min=1)[0] == "gameplay"
    assert profiles.detect("Some video", 0.1, kw, words_per_min=120)[0] == "stream"        # a talker stays a streamer video
    assert profiles.detect("Some video", 0.1, kw, words_per_min=None)[0] == "stream"       # unknown: old behaviour


def test_all_rejected_for_no_speech_retries_as_gameplay_only_when_type_was_guessed(tmp_path):
    from engine.config import Settings
    from engine.pipeline import Engine
    d = Database(tmp_path / "t.db")
    s = Settings()
    s.storage_dir = str(tmp_path / "storage")
    eng = Engine(s, log=lambda m: None)
    eng.db = d
    for md, expect in (({"content_type": "stream", "content_type_reason": "guess"}, True),
                       ({"content_type": "stream"}, False)):                               # the user chose it: respect that
        d.execute("DELETE FROM streams")
        d.execute("DELETE FROM candidates")
        d.execute("INSERT INTO streams(id,platform,source_url,title,metadata_json,created_at,updated_at) VALUES (1,'youtube','x','t',?,0,0)",
                  (json.dumps(md),))
        for i in range(4):
            d.execute("INSERT INTO candidates(stream_id,start,end,status,reject_reason,created_at) VALUES (1,0,9,'rejected',?,0)",
                      ("AI rejected: No speech detected in the candidate window",))
        assert eng._retry_as_gameplay(1) is expect
    assert "Gameplay footage" in eng._no_clips_note(1) and "rejected" in eng._no_clips_note(1)


def test_new_clips_come_first_in_review_and_can_be_filtered(tmp_path):
    d = Database(tmp_path / "t.db")
    for sid, ts in ((1, 1000.0), (2, 5000.0)):                       # video 2 was made later
        d.execute("INSERT INTO streams(id,platform,source_url,title,created_at,updated_at) VALUES (?,?,?,?,0,0)",
                  (sid, "file", f"x{sid}", f"Video {sid}"))
        for k, score in enumerate((90, 50)):                          # an old video's clips can have HIGHER scores
            d.execute("INSERT INTO clips(stream_id,start,end,score,status,video_path,created_at,updated_at) VALUES (?,?,?,?,?,?,?,0)",
                      (sid, 0, 30, score + (10 if sid == 1 else 0), "rendered", "v.mp4", ts + k))
    order = [c["stream_id"] for c in review.to_review(d)]
    assert order[:2] == [2, 2] and order[2:] == [1, 1]                # the newest video's batch is shown first
    assert {c["stream_id"] for c in review.to_review(d, stream_id=1)} == {1}
    assert review.waiting_count(d) == 4
    assert [v["stream_id"] for v in review.waiting_by_video(d)] == [2, 1]
    review.decide(d, 4, "keep")
    assert review.waiting_count(d) == 3


def test_one_song_can_be_used_for_100_clips(tmp_path):
    from engine.music.match import song_ineligible
    d = Database(tmp_path / "t.db")
    d.execute("INSERT INTO artists(id,name,created_at) VALUES (1,'A',0)")
    a = {"bars": [0, 2], "duration": 60, "hook": {"start": 5, "end": 15, "downbeat": 5}}
    d.execute("INSERT INTO songs(id,slug,title,file_path,analysis_json,artist_id,created_at) VALUES (1,'s','S','x.wav',?,1,0)", (json.dumps(a),))
    d.execute("INSERT INTO song_rights_confirmations(song_id,confirmed_by,statement,confirmed_at) VALUES (1,'me','mine',0)")
    for i in range(50):
        d.execute("INSERT INTO song_usage(song_id,clip_id,used_at) VALUES (1,NULL,?)", (float(i),))
    song, artist = d.one("SELECT * FROM songs WHERE id=1"), d.one("SELECT * FROM artists WHERE id=1")
    assert song_ineligible(song, artist, "", "funny", True, d, 10, 0, None) is None                 # default: no limit
    assert "rotation limit" in song_ineligible(song, artist, "", "funny", True, d, 10, 3, None)     # still available if you ask for it
    assert Settings().music.rotation_max == 0


def test_credit_switches_default_on_and_drop_the_overlay_line():
    from engine.captions.ass import build_ass
    from engine.config import CaptionStyle
    st = CaptionStyle()
    assert st.show_attribution and st.credit_in_caption and Settings().folder_names == "video"
    with_credit = build_ass([], 3.0, st, 1080, 1920, "split", 0.38, attribution="Kai Cenat · YouTube")
    without = build_ass([], 3.0, st, 1080, 1920, "split", 0.38, attribution=None)
    assert "Kai Cenat" in with_credit and "Kai Cenat" not in without and ",Attr," not in without.split("[Events]")[1]
