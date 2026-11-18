"""Trend engine (M15), style recipes, clip editing and the end card (M8). Synthetic data only."""
import subprocess
import time
from pathlib import Path

import pytest

from engine.config import Settings
from engine.db import Database, loads
from engine.trends import recipes, report, store


def _ref(i, layout, cuts, music, perf_views, followers=5000, age_days=3, silence=False, hits=1.0):
    return {"id": i, "url": f"u{i}", "platform": "youtube", "account": f"a{i}", "streamer": None,
            "publish_date": time.strftime("%Y-%m-%d", time.gmtime(time.time() - age_days * 86400)),
            "created_at": time.time(), "duration": 25, "moment_type": None,
            "metrics": {"views": perf_views * age_days, "views_per_day": perf_views, "followers": followers},
            "visual": {"duration_s": 25, "cuts_per_10s": cuts, "layout": {"kind": layout, "cam_ratio": 0.38},
                       "hook_frame": {"face_visible": True, "motion": 5, "has_text": True},
                       "captions": {"present": True, "position": "lower"}, "loops_cleanly": False},
            "audio": {"music": {"kind": music}, "silence_before_peak": silence, "transient_hits_per_10s": hits, "bpm": 140}}


def _refs():
    refs = []
    for i in range(10):      # fast-cut split clips do well
        refs.append(_ref(i, "split", 5, "none", 9000 + i * 100))
    for i in range(10, 20):  # calm full-screen clips do poorly
        refs.append(_ref(i, "gameplay_only", 1, "bed", 1000 + i * 10))
    perf = store.performance_index(refs)
    for r in refs:
        r["perf"] = perf[r["id"]]
    return refs


def test_report_ranks_by_normalized_performance_with_sample_sizes():
    rep = report.build_report(_refs(), "30d")
    by = {i["pattern"]: i for i in rep["all_patterns"]}
    fast, calm = by["pacing: fast (4+ cuts / 10 s)"], by["pacing: calm (under 2 cuts / 10 s)"]
    assert fast["n"] == 10 and fast["of"] == 20 and calm["n"] == 10
    assert fast["median_perf"] > calm["median_perf"]
    assert fast["vs_baseline"] > 1 > calm["vs_baseline"]
    assert not fast["early_signal"]                 # n=10 >= 8
    assert "pacing: fast (4+ cuts / 10 s)" in [w["pattern"] for w in rep["working_now"]]
    small = report.build_report(_refs()[:4], "30d")
    assert all(i["early_signal"] for i in small["all_patterns"])      # small samples are never presented as trends


def test_small_account_outperforming_counts_more():
    a = _ref(1, "split", 4, "none", 1000, followers=1000)
    b = _ref(2, "split", 4, "none", 1000, followers=1_000_000)
    idx = store.performance_index([a, b])
    assert idx[1] > idx[2]


def test_unavailable_metrics_are_not_estimated():
    r = _ref(1, "split", 4, "none", 100)
    r["metrics"] = {"views": None, "views_per_day": None, "followers": None}
    assert store.performance_index([r]) == {1: None}


def test_rising_and_saturated_detection():
    now = time.time()
    cur = [_ref(i, "split", 5, "none", 5000 + i, age_days=2) for i in range(8)] + [_ref(100 + i, "gameplay_only", 1, "none", 1000, age_days=2) for i in range(2)]
    prev = [_ref(200 + i, "split", 5, "none", 5000, age_days=40) for i in range(2)] + [_ref(300 + i, "gameplay_only", 1, "none", 5000, age_days=40) for i in range(8)]
    refs = cur + prev
    perf = store.performance_index(refs)
    for r in refs:
        r["perf"] = perf[r["id"]]
    rep = report.build_report(refs, "30d", now)
    assert "layout: split (face cam + gameplay)" in [i["pattern"] for i in rep["rising"]]


def test_recipe_from_report_is_style_only_and_validates():
    refs = _refs()
    rep = report.build_report(refs, "30d")
    rec = recipes.from_report(rep, refs, "chase", 1)
    assert rec["layout"] == "split" and rec["pacing"]["remove_pauses_over_s"] == 0.4
    recipes.validate(rec)
    with pytest.raises(ValueError):
        recipes.validate({**rec, "watermark": "@someoneelse"})
    with pytest.raises(ValueError):
        recipes.validate({**rec, "audio": {"music_mode": "payoff_drop", "song_file": "trending.mp3"}})
    tw = recipes.twists(rec, rep)
    assert 2 <= len(tw) <= 3


def test_recipe_changes_pacing_captions_layout_and_music_mode():
    base = Settings()
    rec = {"recipe_id": "x_v1", "layout": "split", "cam_ratio": 0.3, "pacing": {"remove_pauses_over_s": 0.4},
           "captions": {"words_per_line": 2, "animation": "highlight", "position": "seam"},
           "audio": {"music_mode": "cold_open"}, "zoom": {"punch_in_on_reaction": False}}
    s = recipes.apply(base, rec)
    assert (s.video.max_pause, s.video.cam_ratio, s.captions.max_words_per_line, s.captions.animation, s.captions.position,
            s.video.punch_in) == (0.4, 0.3, 2, "highlight", "seam", False)
    assert base.video.max_pause != 0.4                      # the original is untouched
    assert recipes.music_mode(rec) == "cold_open"


def test_learning_loop_scores_recipes_against_baseline(tmp_path):
    db = Database(tmp_path / "t.db")
    now = time.time()
    sid = db.insert("streams", {"platform": "file", "source_url": "x", "status": "processed", "created_at": now, "updated_at": now})
    for i in range(10):
        lay = {"recipe": {"recipe_id": "r_v1"}} if i < 5 else {}
        cid = db.insert("clips", {"stream_id": sid, "start": 0, "end": 30, "layout_json": lay, "created_at": now, "updated_at": now})
        recipes.record_performance(db, cid, "tiktok", 5000 if i < 5 else 1000)
    sc = recipes.score(db, "r_v1")
    assert sc["clips"] == 5 and sc["vs_baseline"] == 5.0 and sc["verdict"].startswith("beats the baseline")
    recipes.save(db, {"recipe_id": "r_v1", "layout": "split"}, [])
    recipes.set_status(db, "r_v1", "default")
    assert recipes.default_recipe(db)["recipe_id"] == "r_v1"


def test_edit_clip_validates_and_flags_render(tmp_path):
    from engine import editing

    db = Database(tmp_path / "t.db")
    now = time.time()
    sid = db.insert("streams", {"platform": "file", "source_url": "x", "status": "processed", "created_at": now, "updated_at": now})
    cid = db.insert("clips", {"stream_id": sid, "start": 10, "end": 40, "metadata_json": {"hooks": ["a", "b", "c"], "title": "t"},
                              "layout_json": {}, "created_at": now, "updated_at": now})
    with pytest.raises(ValueError):
        editing.edit_clip(db, cid, start=10, end=12)
    with pytest.raises(ValueError):
        editing.edit_clip(db, cid, layout="sideways")
    with pytest.raises(ValueError):
        editing.edit_clip(db, cid, caption={"evil": 1})
    r = editing.edit_clip(db, cid, start=12, end=44, hook_text="new hook", layout="fullcam", caption={"position": "upper"},
                          title="T2", description="desc")
    assert r["needs_render"]
    row = db.one("SELECT * FROM clips WHERE id=?", (cid,))
    assert (row["start"], row["end"]) == (12, 44)
    meta = loads(row["metadata_json"])
    assert meta["hooks"][0] == "new hook" and meta["title"] == "T2" and meta["description"] == "desc"
    assert loads(row["layout_json"])["layout"] == "fullcam" and loads(row["caption_style_json"])["position"] == "upper"
    assert editing.edit_clip(db, cid, title="only text")["needs_render"] is False


def test_end_card_appends_card_to_clip(tmp_path):
    from engine.config import VideoSettings
    from engine.media import ffmpeg
    from engine.video.endcard import append_end_card, make_card_image

    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=1080x1920:r=30:d=3", "-f", "lavfi", "-i",
                    "sine=f=440:d=3", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(clip)], check=True)
    png = make_card_image(tmp_path / "card.png", 1080, 1920, "A Very Long Song Title Indeed", "Some Artist", "Full song in bio")
    out = append_end_card(clip, tmp_path / "out.mp4", png, 2.0, VideoSettings(encoder="libx264", preset="ultrafast"))
    a, b = ffmpeg.probe(clip), ffmpeg.probe(out)
    assert abs((b.duration - a.duration) - 2.0) < 0.15
    assert (b.width, b.height) == (1080, 1920) and b.has_audio
    assert ffmpeg.decode_check(out) is None


def test_demucs_is_optional():
    from engine.music import separate

    if not separate.available():
        with pytest.raises(RuntimeError, match="Demucs"):
            separate.voice_stem(Path("x.wav"), 0, 1, Path("y.wav"))
