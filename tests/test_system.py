"""Hardware detection, job queue resume, source classification, metadata (PRD §16, §20, §13)."""
import time

import pytest

from engine.db import Database
from engine.hardware import Environment, recommend
from engine.ingestion.sources import SourceError, classify
from engine.metadata.generate import generate
from engine.pipeline import _claim, enqueue


def test_recommendations_scale_with_hardware():
    big = Environment("w", "3.13", "cpu", 8, 16, 32.0, cuda_for_whisper=True)
    big.gpus = [type("G", (), {"vram_mb": 12000})()]
    small = Environment("w", "3.13", "cpu", 4, 8, 8.0)
    assert recommend(big).whisper_device == "cuda" and recommend(big).max_concurrent_jobs == 1
    r = recommend(small)
    assert r.whisper_device == "cpu" and r.whisper_model in ("small", "base")


def test_classify_sources(tmp_path):
    f = tmp_path / "x.mp4"
    f.write_bytes(b"0")
    assert classify(str(f)) == "file"
    assert classify("https://www.youtube.com/watch?v=abc") == "youtube"
    assert classify("https://www.twitch.tv/videos/123") == "twitch"
    assert classify("https://kick.com/someone") == "kick"
    with pytest.raises(SourceError):
        classify("https://example.com/video")
    with pytest.raises(SourceError):
        classify(str(tmp_path / "missing.mp4"))


def test_queue_claim_and_resume(tmp_path):
    db = Database(tmp_path / "q.db")
    now = time.time()
    sid = db.insert("streams", {"platform": "file", "source_url": "a", "status": "queued", "created_at": now, "updated_at": now})
    jid = enqueue(db, sid, {})
    assert enqueue(db, sid, {}) == jid                      # no duplicate jobs for one stream
    job = _claim(db)
    assert job["id"] == jid and _claim(db) is None
    # simulate a crash mid-job: run_queue's reset makes it claimable again
    db.execute("UPDATE jobs SET status='queued' WHERE status='running'")
    assert _claim(db)["id"] == jid


def test_metadata_attribution_and_hooks():
    ctx = {"streamer": "Streamer", "platform": "twitch", "url": "https://twitch.tv/videos/1", "category": "funny",
           "summary": "", "transcript": "Wait what is that? No way! That is insane!", "vertical": "GTA 6",
           "attribution_format": None}
    m = generate(ctx, provider=None)
    assert len(m["hooks"]) == 3
    assert "Streamer" in m["attribution"] and "twitch.tv" in m["description"]
    assert all(tags[0] == "#GTA6" for tags in m["hashtags"].values())
