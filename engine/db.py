"""SQLite storage. One file, WAL mode, schema created idempotently.

The music / trend tables are created now (per the PRD addendum §25) so later
phases don't need a migration.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 1

PERMISSION_STATUSES = (
    "explicitly_authorized",
    "official_clipping_program",
    "permission_requested",
    "unknown",
    "not_authorized",
)
PUBLISHABLE_PERMISSIONS = ("explicitly_authorized", "official_clipping_program")

SCHEMA = f"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS streamers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    platform TEXT NOT NULL,
    channel_url TEXT,
    channel_id TEXT,
    permission_status TEXT NOT NULL DEFAULT 'unknown'
        CHECK (permission_status IN ({",".join(repr(s) for s in PERMISSION_STATUSES)})),
    allows_clipping INTEGER,
    allows_monetization INTEGER,
    allows_added_music INTEGER,
    allows_third_party_promotion INTEGER,
    requires_attribution_format TEXT,
    clipping_program_rules TEXT,
    favorite INTEGER NOT NULL DEFAULT 0,
    excluded INTEGER NOT NULL DEFAULT 0,
    watchlist INTEGER NOT NULL DEFAULT 0,
    baseline_json TEXT,
    notes TEXT,
    created_at REAL NOT NULL,
    UNIQUE (platform, name)
);

CREATE TABLE IF NOT EXISTS streams (
    id INTEGER PRIMARY KEY,
    streamer_id INTEGER REFERENCES streamers(id),
    platform TEXT NOT NULL,
    source_url TEXT NOT NULL UNIQUE,
    title TEXT,
    channel_url TEXT,
    duration REAL,
    started_at TEXT,
    viewer_count INTEGER,
    total_views INTEGER,
    is_live INTEGER,
    discovery_score REAL,
    status TEXT NOT NULL DEFAULT 'discovered',
    error TEXT,
    audio_path TEXT,
    chat_path TEXT,
    metadata_json TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS stream_steps (
    stream_id INTEGER NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    step TEXT NOT NULL,
    status TEXT NOT NULL,
    output_json TEXT,
    duration_s REAL,
    peak_rss_mb REAL,
    finished_at REAL,
    PRIMARY KEY (stream_id, step)
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id INTEGER PRIMARY KEY,
    stream_id INTEGER NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL,
    speaker TEXT,
    confidence REAL,
    words_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_seg_stream ON transcript_segments(stream_id, start);

CREATE TABLE IF NOT EXISTS analysis (
    stream_id INTEGER NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    path TEXT,
    summary_json TEXT,
    created_at REAL NOT NULL,
    PRIMARY KEY (stream_id, kind)
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY,
    stream_id INTEGER NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    start REAL NOT NULL,
    end REAL NOT NULL,
    peak REAL,
    payoff REAL,
    category TEXT,
    summary TEXT,
    signals_json TEXT,
    scores_json TEXT,
    total_score REAL,
    ai_json TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',
    reject_reason TEXT,
    music_eligible INTEGER,
    music_reason TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY,
    stream_id INTEGER NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    candidate_id INTEGER REFERENCES candidates(id),
    start REAL NOT NULL,
    end REAL NOT NULL,
    score REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    approval TEXT NOT NULL DEFAULT 'pending',
    publish_ready INTEGER NOT NULL DEFAULT 0,
    publish_block_reason TEXT,
    hook_choice INTEGER NOT NULL DEFAULT 0,
    hooks_json TEXT,
    metadata_json TEXT,
    layout_json TEXT,
    caption_style_json TEXT,
    captions_json TEXT,
    qc_json TEXT,
    video_path TEXT,
    thumb_path TEXT,
    render_seconds REAL,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    stream_id INTEGER REFERENCES streams(id) ON DELETE CASCADE,
    payload_json TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    level TEXT NOT NULL,
    stream_id INTEGER,
    job_id INTEGER,
    message TEXT NOT NULL
);

-- ---------------- Music promotion module (M2-M11) ----------------
CREATE TABLE IF NOT EXISTS artists (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    pronunciation TEXT,
    brand_description TEXT,
    vibe_keywords_json TEXT,
    good_fit_moments_json TEXT,
    avoid_moments_json TEXT,
    sound_tag_format TEXT,
    cta_options_json TEXT,
    social_handles_json TEXT,
    visual_style_json TEXT,
    default_volume TEXT DEFAULT 'balanced',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    artist_id INTEGER REFERENCES artists(id),
    title TEXT NOT NULL,
    file_path TEXT NOT NULL,
    release_status TEXT,
    release_date TEXT,
    links_json TEXT,
    official_sounds_json TEXT,
    campaign_priority INTEGER DEFAULT 3,
    campaign_start TEXT,
    campaign_end TEXT,
    explicit INTEGER DEFAULT 0,
    positioning_notes TEXT,
    analysis_json TEXT,
    analysis_overrides_json TEXT,
    file_hash TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS song_sections (
    id INTEGER PRIMARY KEY, song_id INTEGER NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    label TEXT, start REAL, end REAL, energy REAL, vocals INTEGER, is_hook INTEGER, user_override INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS song_lyrics (
    id INTEGER PRIMARY KEY, song_id INTEGER NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    start REAL, end REAL, text TEXT
);
CREATE TABLE IF NOT EXISTS song_promo_segments (
    id INTEGER PRIMARY KEY, song_id INTEGER NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    length INTEGER, start REAL, end REAL, contains_hook INTEGER
);
CREATE TABLE IF NOT EXISTS song_stems (
    id INTEGER PRIMARY KEY, song_id INTEGER NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    kind TEXT, file_path TEXT, generated INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS song_rights_confirmations (
    id INTEGER PRIMARY KEY, song_id INTEGER NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    confirmed_by TEXT, statement TEXT, confirmed_at REAL
);
CREATE TABLE IF NOT EXISTS music_conflict_regions (
    id INTEGER PRIMARY KEY, clip_id INTEGER REFERENCES clips(id) ON DELETE CASCADE,
    stream_id INTEGER REFERENCES streams(id) ON DELETE CASCADE,
    start REAL, end REAL, type TEXT, confidence REAL, level_db REAL
);
CREATE TABLE IF NOT EXISTS music_placements (
    id INTEGER PRIMARY KEY, clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    song_id INTEGER REFERENCES songs(id), mode TEXT, song_in REAL, song_out REAL,
    clip_in REAL, clip_out REAL, payoff_sync_json TEXT, uses_stem TEXT,
    source_music_removed INTEGER DEFAULT 0, fit_score REAL, reasoning TEXT, chosen INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS song_usage (
    id INTEGER PRIMARY KEY, song_id INTEGER REFERENCES songs(id), clip_id INTEGER REFERENCES clips(id), used_at REAL
);
CREATE TABLE IF NOT EXISTS reference_clips (
    id INTEGER PRIMARY KEY, url TEXT UNIQUE, platform TEXT, account TEXT, publish_date TEXT,
    duration REAL, metrics_json TEXT, moment_type TEXT, streamer TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS reference_clip_features (
    reference_clip_id INTEGER PRIMARY KEY REFERENCES reference_clips(id) ON DELETE CASCADE,
    visual_json TEXT, audio_json TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS trend_reports (id INTEGER PRIMARY KEY, window TEXT, report_json TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS style_recipes (id INTEGER PRIMARY KEY, recipe_id TEXT UNIQUE, recipe_json TEXT, status TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS twists (id INTEGER PRIMARY KEY, recipe_id TEXT, twist_json TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS clip_review (
    clip_id INTEGER PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
    decision TEXT NOT NULL,
    reason TEXT,
    auto INTEGER NOT NULL DEFAULT 0,
    posted INTEGER NOT NULL DEFAULT 0,
    posted_at REAL,
    posted_platform TEXT,
    ai_rating REAL,
    ai_note TEXT,
    decided_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS clip_performance (
    id INTEGER PRIMARY KEY, clip_id INTEGER REFERENCES clips(id) ON DELETE CASCADE, platform TEXT,
    views INTEGER, likes INTEGER, shares INTEGER, saves INTEGER, comments INTEGER,
    song_results_json TEXT, recorded_at REAL
);
"""

_local = threading.local()


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)
            con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        return con

    @contextmanager
    def tx(self):
        con = self.connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            yield con
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
        finally:
            con.close()

    # ---- tiny helpers -------------------------------------------------
    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        con = self.connect()
        try:
            return con.execute(sql, tuple(params)).fetchall()
        finally:
            con.close()

    def one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        con = self.connect()
        try:
            cur = con.execute(sql, tuple(params))
            return cur.lastrowid
        finally:
            con.close()

    def insert(self, table: str, row: dict) -> int:
        cols = ",".join(row)
        qs = ",".join("?" for _ in row)
        return self.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", [_enc(v) for v in row.values()])

    def update(self, table: str, row_id: int, fields: dict) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE {table} SET {sets} WHERE id=?", [*(_enc(v) for v in fields.values()), row_id])

    def log(self, level: str, message: str, stream_id: int | None = None, job_id: int | None = None) -> None:
        self.execute(
            "INSERT INTO logs(ts, level, stream_id, job_id, message) VALUES (?,?,?,?,?)",
            (time.time(), level, stream_id, job_id, message),
        )


def _enc(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, Path):
        return str(v)
    return v


def loads(value: Optional[str], default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default
