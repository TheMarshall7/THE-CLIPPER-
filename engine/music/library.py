"""Song library: import, rights confirmation, one-time analysis (M2).

Analysis (cached by file hash, never repeated for an unchanged file):
- tempo + beat grid (librosa), downbeats/bars (phase with the strongest low-end onsets)
- key/mode (Krumhansl-Schmuckler on mean chroma)
- energy curve, structure (recurrence-based segmentation snapped to bars),
  section labels by energy + repetition (the hook = loudest repeated section)
- mood (valence/arousal + tags; heuristic, editable)
- timed lyrics + vocal map via Faster-Whisper (optional)
- bar-aligned promo segments of 8/15/30/60 s that contain the hook
User corrections are stored in analysis_overrides_json and always win.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from engine.db import Database, loads

MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "song"


def estimate_key(chroma_mean: np.ndarray) -> tuple[str, float]:
    best, score = "C major", -1.0
    for i in range(12):
        for prof, mode in ((MAJOR, "major"), (MINOR, "minor")):
            r = float(np.corrcoef(np.roll(prof, i), chroma_mean)[0, 1])
            if r > score:
                best, score = f"{NOTES[i]} {mode}", r
    return best, round(score, 3)


def analyze(path: Path, lyrics_cfg=None, log: Callable[[str], None] = print) -> dict:
    import librosa

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    dur = len(y) / sr
    hop = 512
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    if beats.size < 8:
        raise ValueError("beat tracking failed (track may be beatless); enter BPM/sections manually")
    # 1) snap each beat to the strongest onset within +-60 ms
    fps = sr / hop
    w = int(0.06 * fps)
    ref = []
    for f in beat_frames:
        lo, hi = max(0, f - w), min(onset.size, f + w + 1)
        ref.append(lo + int(np.argmax(onset[lo:hi])))
    beats = librosa.frames_to_time(np.array(ref), sr=sr, hop_length=hop)
    # 2) most produced tracks sit on a fixed grid: robust-fit t = t0 + k*period and use it if it fits
    k_idx = np.arange(beats.size)
    period, t0 = np.polyfit(k_idx, beats, 1)
    for _ in range(3):  # re-fit without outliers
        res = beats - (t0 + period * k_idx)
        keep = np.abs(res) < max(0.03, 2.5 * np.std(res))
        period, t0 = np.polyfit(k_idx[keep], beats[keep], 1)
    res = beats - (t0 + period * k_idx)
    grid_fit = float(np.median(np.abs(res)))
    if grid_fit < 0.025:
        # spectral-flux onsets peak ~1-2 frames after the true attack: fine-align the grid
        # against a 2.9 ms resolution energy-rise envelope from the raw waveform
        fh = 64
        e = np.sqrt(np.convolve(y ** 2, np.ones(256) / 256, mode="same")[::fh] + 1e-12)
        rise = np.maximum(0, np.diff(np.log(e), prepend=np.log(e[0])))
        k_all = np.arange(int((dur - t0) / period))
        best_s, best_v = 0.0, -1.0
        for s in np.arange(-0.08, 0.0201, 0.002):
            idx = ((t0 + s + period * k_all) * sr / fh).astype(int)
            idx = idx[(idx >= 0) & (idx < rise.size)]
            v = float(rise[idx].sum())
            if v > best_v:
                best_s, best_v = s, v
        beats = t0 + best_s + period * k_idx
        tempo = 60.0 / period
    beat_frames = librosa.time_to_frames(beats, sr=sr, hop_length=hop)
    # downbeat phase: chords change on the "1" (harmonic novelty), and the bass/kick tends to be
    # strongest there. Four-on-the-floor kicks alone can't tell the phase, so both cues are used.
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    low = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[:24]), sr=sr, hop_length=hop)
    bf = np.clip(beat_frames, 0, low.size - 1)
    chroma_b = librosa.util.sync(librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop), beat_frames, aggregate=np.mean)
    # sync() column 0 is the span before the first beat, so column j+1 starts at beat j:
    # diff[j] = col[j+1] - col[j] = harmonic change INTO beat j
    nov = np.linalg.norm(np.diff(chroma_b, axis=1), axis=0)[: beats.size]
    nov = np.r_[nov, np.zeros(max(0, beats.size - nov.size))]

    def zs(a):
        return (a - a.mean()) / (a.std() + 1e-9)

    harm = zs(nov[: beats.size])
    lowb = zs(low[bf])
    phase = int(np.argmax([0.65 * harm[p::4].mean() + 0.35 * lowb[p::4].mean() for p in range(4)]))
    bars = beats[phase::4]
    beat_period = 60.0 / tempo
    # extrapolate the grid back to the song start so early sections can snap too
    while bars.size and bars[0] - 4 * beat_period > -1e-3:
        bars = np.r_[bars[0] - 4 * beat_period, bars]
    bars = bars[bars >= -1e-3]

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    key, key_conf = estimate_key(chroma.mean(axis=1))
    rms = librosa.feature.rms(S=S)[0]
    t_rms = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=hop)
    energy = rms / (rms.max() + 1e-9)
    centroid = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())

    # structure: bar-synchronous timbre + energy + harmony, boundaries where they change most
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
    bar_frames = librosa.time_to_frames(bars, sr=sr, hop_length=hop)
    bar_frames = bar_frames[(bar_frames > 0) & (bar_frames < rms.size)]
    feats = np.vstack([librosa.util.normalize(mfcc, axis=1), 3 * energy[None, :], chroma])
    bsync = librosa.util.sync(feats, bar_frames, aggregate=np.mean)    # one column per bar
    nb = bsync.shape[1]
    k = int(np.clip(nb // 6, 3, 12))
    if nb > k:
        bounds_b = librosa.segment.agglomerative(bsync, k)
        cuts = [0.0] + [float(librosa.frames_to_time(([0] + list(bar_frames))[min(b, nb - 1)], sr=sr, hop_length=hop))
                        for b in bounds_b[1:]]
    else:
        cuts = [0.0]
    # refine each boundary to the bar (+-2) with the largest 2-bar before/after contrast
    bar_starts = np.r_[0.0, librosa.frames_to_time(bar_frames, sr=sr, hop_length=hop)]  # column j starts here
    refined = []
    for t in cuts[1:]:
        j = int(np.argmin(np.abs(bar_starts - t)))
        best, best_v = j, -1.0
        for c in range(max(2, j - 2), min(nb - 1, j + 2) + 1):
            v = float(np.linalg.norm(bsync[:, c:c + 2].mean(axis=1) - bsync[:, c - 2:c].mean(axis=1)))
            if v > best_v:
                best, best_v = c, v
        refined.append(float(bar_starts[min(best, bar_starts.size - 1)]))
    snapped = sorted(set([0.0] + [float(bars[np.argmin(np.abs(bars - t))]) for t in refined] + [dur]))
    sections = []
    for a, b in zip(snapped, snapped[1:]):
        if b - a < 2 * 4 * beat_period * 0.9 and (sections or b < dur - 1):   # merge sections < 2 bars
            if sections:
                sections[-1]["end"] = b
            else:
                pending_start = a
            continue
        if not sections and "pending_start" in locals():
            a = pending_start
        sel = (t_rms >= a) & (t_rms < b)
        fr = (librosa.time_to_frames(a, sr=sr, hop_length=hop), librosa.time_to_frames(b, sr=sr, hop_length=hop))
        sections.append({"start": round(a, 3), "end": round(b, 3),
                         "energy": round(float(energy[sel].mean()) if sel.any() else 0.0, 3),
                         "_chroma": chroma[:, fr[0]:fr[1]].mean(axis=1)})
    # labels: repeated + loud = chorus (hook); first quiet = intro; last quiet = outro
    es = np.array([s["energy"] for s in sections])
    hi = es >= np.percentile(es, 60)
    for i, s in enumerate(sections):
        rep = sum(1 for j, o in enumerate(sections) if j != i and
                  float(np.dot(s["_chroma"], o["_chroma"]) / (np.linalg.norm(s["_chroma"]) * np.linalg.norm(o["_chroma"]) + 1e-9)) > 0.97)
        if i == 0 and s["energy"] < es.mean():
            s["label"] = "intro"
        elif i == len(sections) - 1 and s["energy"] < es.mean():
            s["label"] = "outro"
        elif hi[i] and (rep or s["energy"] == es.max()):
            s["label"] = "chorus"
        else:
            s["label"] = "verse"
        s["repeats"] = rep
    choruses = [s for s in sections if s["label"] == "chorus"] or [max(sections, key=lambda s: s["energy"])]
    hook = choruses[0]
    for s in sections:
        s["is_hook"] = s is hook
        s.pop("_chroma", None)

    # mood (heuristic, editable)
    arousal = float(np.clip(0.55 * (tempo - 70) / 100 + 0.45 * energy.mean() / 0.6, 0, 1))
    major = key.endswith("major")
    valence = float(np.clip(0.55 * (1.0 if major else 0.25) + 0.45 * np.clip((centroid - 1200) / 2500, 0, 1), 0, 1))
    tags = []
    if arousal > 0.65:
        tags += ["hype", "energetic"] + (["triumphant"] if valence > 0.5 else ["dark", "chaotic", "menacing"])
    elif arousal > 0.4:
        tags += ["bouncy", "playful"] if valence > 0.5 else ["moody", "cinematic"]
    else:
        tags += ["smooth", "chill"] if valence > 0.5 else ["melancholic", "late-night"]

    lyrics, vocals = [], []
    if lyrics_cfg is not None:
        try:
            from engine.transcription.whisper import load_model
            model, _ = load_model(lyrics_cfg, log)
            y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
            segs, _ = model.transcribe(y16, word_timestamps=True, vad_filter=True, language=lyrics_cfg.language)
            for s in segs:
                if s.no_speech_prob < 0.6 and s.text.strip():
                    lyrics.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
            vocals = [[l["start"], l["end"]] for l in lyrics]
        except Exception as e:  # lyrics are optional
            log(f"lyric transcription skipped: {e}")
    for s in sections:
        s["vocals"] = any(v0 < s["end"] and v1 > s["start"] for v0, v1 in vocals)

    bar_len = 4 * beat_period
    promo = []
    hook_bar = float(bars[np.argmin(np.abs(bars - hook["start"]))])
    for L in (8, 15, 30, 60):
        start = max(0.0, hook_bar - (bar_len if L >= 15 else 0))
        ends = bars[bars >= start + L - bar_len * 0.5]
        end = float(ends[0]) if ends.size else dur
        if end - start >= L * 0.75 and end <= dur:
            promo.append({"length": L, "start": round(start, 3), "end": round(end, 3),
                          "contains_hook": start <= hook["start"] < end})
    t_e = np.arange(0, dur, 0.5)
    return {
        "duration": round(dur, 3), "bpm": round(tempo, 2), "beats": [round(float(b), 4) for b in beats],
        "bars": [round(float(b), 4) for b in bars], "bar_s": round(bar_len, 4),
        "key": key, "key_confidence": key_conf,
        "energy_curve": [round(float(v), 3) for v in np.interp(t_e, t_rms, energy)],
        "sections": sections, "hook": {"start": hook["start"], "end": hook["end"], "downbeat": hook_bar},
        "mood": {"valence": round(valence, 2), "arousal": round(arousal, 2), "tags": tags, "source": "auto"},
        "lyrics": lyrics, "lyric_themes": [], "promo_segments": promo, "analyzed_at": time.time(),
    }


def effective_analysis(song_row) -> dict:
    """Automatic analysis with the user's corrections applied on top."""
    a = loads(song_row["analysis_json"], {}) or {}
    for k, v in (loads(song_row["analysis_overrides_json"], {}) or {}).items():
        a[k] = v
    return a


def set_hook(db: Database, song_id: int, start: float, end: float) -> dict:
    """The user marked the part of the song to use: [start, end] seconds. The engine will always enter the song at this
    part (on the nearest bar) and never play past its end."""
    import json

    import numpy as np

    row = db.one("SELECT * FROM songs WHERE id=?", (song_id,))
    a = effective_analysis(row)
    if end - start < 1.5:
        raise ValueError("The marked part must be at least 1.5 seconds long.")
    bars = np.array(a.get("bars") or [start])
    near = float(bars[np.argmin(np.abs(bars - start))])
    downbeat = near if abs(near - start) <= 0.6 else float(start)
    ov = loads(row["analysis_overrides_json"], {}) or {}
    ov["hook"] = {**(a.get("hook") or {}), "start": float(start), "end": float(end), "downbeat": downbeat, "user": True}
    secs = [dict(s) for s in a.get("sections", [])]
    best, best_ov = None, 0.0
    for s in secs:
        o = max(0.0, min(s["end"], end) - max(s["start"], start))
        s["is_hook"] = False
        if o > best_ov:
            best, best_ov = s, o
    if best is not None:
        best["is_hook"], best["label"] = True, "chorus"
    else:
        secs.append({"label": "chorus", "start": float(start), "end": float(end), "energy": 1.0, "vocals": 1, "is_hook": True})
    ov["sections"] = secs
    db.execute("UPDATE songs SET analysis_overrides_json=? WHERE id=?", (json.dumps(ov), song_id))
    return ov["hook"]


def clear_hook(db: Database, song_id: int) -> None:
    import json

    row = db.one("SELECT analysis_overrides_json FROM songs WHERE id=?", (song_id,))
    ov = loads(row["analysis_overrides_json"], {}) or {}
    ov.pop("hook", None)
    ov.pop("sections", None)
    db.execute("UPDATE songs SET analysis_overrides_json=? WHERE id=?", (json.dumps(ov), song_id))


def import_song(db: Database, library: Path, src: Path, title: str, artist: str, rights_statement: str,
                confirmed_by: str, explicit: bool = False, priority: int = 3, official_sounds: Optional[dict] = None,
                lyrics_cfg=None, log: Callable[[str], None] = print) -> int:
    if not rights_statement:
        raise ValueError("Rights must be confirmed: only songs you own or control can be placed.")
    library.mkdir(parents=True, exist_ok=True)
    h = file_hash(src)
    existing = db.one("SELECT * FROM songs WHERE file_hash=?", (h,))
    if existing:
        log(f"song already imported as '{existing['slug']}' (analysis cached)")
        return existing["id"]
    slug = slugify(f"{title}")
    if db.one("SELECT id FROM songs WHERE slug=?", (slug,)):
        slug = f"{slug}_{h[:6]}"
    dest = library / f"{slug}{src.suffix.lower()}"
    shutil.copy2(src, dest)
    art = db.one("SELECT id FROM artists WHERE name=?", (artist,))
    artist_id = art["id"] if art else db.insert("artists", {"name": artist, "created_at": time.time()})
    log(f"analyzing '{title}' (one time)...")
    try:
        analysis = analyze(dest, lyrics_cfg, log)
        err = None
    except Exception as e:
        analysis, err = {"error": str(e)}, str(e)
        log(f"analysis failed: {e}. The song is kept; enter BPM/sections manually.")
    sid = db.insert("songs", {
        "slug": slug, "artist_id": artist_id, "title": title, "file_path": str(dest), "explicit": explicit,
        "campaign_priority": priority, "official_sounds_json": official_sounds or {}, "analysis_json": analysis,
        "file_hash": h, "release_status": "released", "created_at": time.time()})
    db.insert("song_rights_confirmations", {"song_id": sid, "confirmed_by": confirmed_by, "statement": rights_statement,
                                            "confirmed_at": time.time()})
    if not err:
        for s in analysis["sections"]:
            db.insert("song_sections", {"song_id": sid, "label": s["label"], "start": s["start"], "end": s["end"],
                                        "energy": s["energy"], "vocals": s["vocals"], "is_hook": s["is_hook"]})
        for p in analysis["promo_segments"]:
            db.insert("song_promo_segments", {"song_id": sid, **p})
        for l in analysis["lyrics"]:
            db.insert("song_lyrics", {"song_id": sid, **l})
    return sid
