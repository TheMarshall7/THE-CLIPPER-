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
