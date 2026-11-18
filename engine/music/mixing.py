"""Clip audio rendering (M7): jump cuts, song placement, speech-aware ducking, EQ carve,
beat-aligned fades, loudness normalisation. Done in numpy for sample-accurate control;
FFmpeg only decodes and runs the final two-pass loudnorm.

Used for EVERY clip: without a song it produces the clean (voice/game) track.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

SR = 48000
# song level vs dialogue (not ducked), duck depth under dialogue, game-audio bed while the song plays
PRESETS = {
    "subtle":   {"level_db": -15.0, "duck_db": 15.0, "bed_db": 0.0},
    "balanced": {"level_db": -10.0, "duck_db": 15.0, "bed_db": 0.0},
    "featured": {"level_db": -5.0,  "duck_db": 12.0, "bed_db": -3.0},
    "trailer":  {"level_db": 1.0,   "duck_db": 7.0,  "bed_db": -8.0},   # movie-trailer: music up front
}
PRESET_DB = {k: v["level_db"] for k, v in PRESETS.items()}


def decode(path: Path, start: float, dur: float) -> np.ndarray:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{max(0.0, start):.4f}",
                           "-t", f"{dur:.4f}", "-i", str(path), "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"audio decode failed: {proc.stderr.decode(errors='replace')[-300:]}")
    a = np.frombuffer(proc.stdout, dtype="<f4").reshape(-1, 2).copy()
    need = int(round(dur * SR))
    if a.shape[0] < need:
        a = np.vstack([a, np.zeros((need - a.shape[0], 2), np.float32)])
    return a[:need]


def apply_cuts(a: np.ndarray, keep: list[tuple[float, float]], xfade_ms: float = 8.0) -> np.ndarray:
    """Concatenate kept intervals (seconds relative to the clip start) with short crossfades."""
    if len(keep) == 1 and keep[0][0] <= 1e-6 and abs(keep[0][1] * SR - a.shape[0]) < 2:
        return a
    x = int(xfade_ms / 1000 * SR)
    out = None
    for k0, k1 in keep:
        seg = a[int(round(k0 * SR)): int(round(k1 * SR))]
        if out is None or seg.shape[0] < 2 * x or out.shape[0] < x:
            out = seg.copy() if out is None else np.vstack([out, seg])
            continue
        ramp = np.linspace(0, 1, x, dtype=np.float32)[:, None]
        out[-x:] = out[-x:] * (1 - ramp) + seg[:x] * ramp
        out = np.vstack([out, seg[x:]])
    return out if out is not None else a[:0]


def _rms_db(a: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9)


def speech_level_db(voice: np.ndarray, speech: list[tuple[float, float]]) -> float:
    parts = [voice[int(a * SR): int(b * SR)] for a, b in speech if b > a]
    parts = [p for p in parts if p.size]
    return _rms_db(np.vstack(parts)) if parts else _rms_db(voice)


def duck_envelope(n: int, speech: list[tuple[float, float]], duck_db: float, attack_ms: int, release_ms: int,
                  rate: int = 1000) -> np.ndarray:
    """Gain (linear) per sample: 1.0 in gaps, -duck_db under speech, smoothed attack/release.
    Starts ducking slightly BEFORE each word so the first syllable is never masked."""
    m = int(np.ceil(n / SR * rate)) + 1
    target = np.ones(m)
    low = 10 ** (-duck_db / 20)
    for a, b in speech:
        target[max(0, int((a - attack_ms / 1000) * rate)): min(m, int((b + 0.05) * rate))] = low
    env = np.empty(m)
    g = target[0]
    ca = np.exp(-1 / (attack_ms / 1000 * rate / 3))
    cr = np.exp(-1 / (release_ms / 1000 * rate / 3))
    for i, t in enumerate(target):
        c = ca if t < g else cr
        g = c * g + (1 - c) * t
        env[i] = g
    return np.interp(np.arange(n) / SR * rate, np.arange(m), env).astype(np.float32)


def merge_speech(speech: list[tuple[float, float]], hold_s: float) -> list[tuple[float, float]]:
    """Join talking that stops and starts: gaps shorter than hold_s count as still talking, so the song sits at one
    steady lower level through the whole stretch instead of jumping up in every little pause (pumping)."""
    out: list[list[float]] = []
    for a, b in sorted(speech):
        if out and a - out[-1][1] < hold_s:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _fade(n: int, fin: float, fout: float) -> np.ndarray:
    f = np.ones(n, np.float32)
    a, b = int(fin * SR), int(fout * SR)
    if a > 0:
        f[:a] = (np.linspace(0, 1, a) ** 2)          # quadratic: a soft swell, not a click
    if b > 0:
        f[-b:] *= np.linspace(1, 0, b) ** 1.5
    return f


@dataclass
class SongPlan:
    path: Path
    song_in: float
    song_out: float
    clip_in: float
    fade_in: float
    fade_out: float
    level_db: float            # relative to speech level
    duck_db: float
    attack_ms: int
    release_ms: int
    carve: bool = True
    bed_db: float = 0.0        # trailer style: lower the GAME audio by this much while the song plays (not under dialogue)
    dropout: Optional[tuple[float, float]] = None   # (start, end) clip time: everything drops out right before the chorus hits
    dropout_db: float = -30.0
    hold_s: float = 0.0        # smooth ducking: talking with gaps shorter than this keeps the song steadily ducked (0 = per phrase)


def mix(voice: np.ndarray, speech: list[tuple[float, float]], plan: Optional[SongPlan]) -> tuple[np.ndarray, dict]:
    if plan is None:
        return voice, {}
    n_song = int((plan.song_out - plan.song_in) * SR)
    song = decode(plan.path, plan.song_in, plan.song_out - plan.song_in)[:n_song]
    start = int(round(plan.clip_in * SR))
    n = min(song.shape[0], voice.shape[0] - start)
    if n <= 0:
        return voice, {"error": "song window outside clip"}
    fade = _fade(n, plan.fade_in, min(plan.fade_out, n / SR / 2))
    song = song[:n] * fade[:, None]
    v_db = speech_level_db(voice, speech)
    s_db = _rms_db(song[: int(min(n, 8 * SR))])
    gain = 10 ** ((v_db + plan.level_db - s_db) / 20)
    local_speech = [(a - plan.clip_in, b - plan.clip_in) for a, b in speech if b > plan.clip_in]
    release = plan.release_ms
    if plan.hold_s > 0:                        # smooth: one steady dip per stretch of talking, and a gentle return
        local_speech = merge_speech(local_speech, plan.hold_s)
        release = max(release, 900)
    env = duck_envelope(n, local_speech, plan.duck_db, plan.attack_ms, release)
    if plan.carve and local_speech:            # gentle 1-4 kHz dip in the song while the streamer talks
        sos = butter(2, [1000, 4000], btype="bandpass", fs=SR, output="sos")
        band = sosfiltfilt(sos, song, axis=0).astype(np.float32)
        duck_amt = (1 - env) / max(1e-6, 1 - 10 ** (-plan.duck_db / 20))     # 0 in gaps .. 1 under speech
        song = song - band * (1 - 10 ** (-4 / 20)) * duck_amt[:, None]
    out = voice.copy()
    if plan.bed_db < 0:
        # trailer style: the song carries the scene. Game audio (SFX, ambience) drops by bed_db while the
        # song plays, but comes back up under dialogue so lines stay clear. Follows the song's fades.
        low = 10 ** (-plan.duck_db / 20)
        speech_mask = (1 - env) / max(1e-6, 1 - low)                    # 1 under dialogue, 0 in gaps
        bed = 10 ** (plan.bed_db / 20)
        src_gain = bed + (1 - bed) * np.clip(speech_mask, 0, 1)
        src_gain = 1 - (1 - src_gain) * fade
        out[start:start + n] *= src_gain[:, None]
    out[start:start + n] += song * (gain * env)[:, None]
    if plan.dropout:
        out *= dropout_gain(out.shape[0], plan.dropout, plan.dropout_db)[:, None]
    return out, {"song_gain_db": round(20 * np.log10(gain + 1e-12), 2), "speech_level_db": round(v_db, 2),
                 "song_level_db": round(s_db, 2)}


def dropout_gain(n: int, window: tuple[float, float], depth_db: float = -30.0, down_ms: float = 15.0,
                 up_ms: float = 3.0) -> np.ndarray:
    """Gain per sample: 1 everywhere except `window` (seconds), where the mix falls to depth_db with a short fade
    down and comes back almost instantly, so the chorus hit lands on silence."""
    g = np.ones(n, dtype=np.float32)
    a, b = int(round(window[0] * SR)), int(round(window[1] * SR))
    a, b = max(0, min(n, a)), max(0, min(n, b))
    if b - a < 4:
        return g
    low = 10 ** (depth_db / 20)
    g[a:b] = low
    d = min(int(down_ms / 1000 * SR), (b - a) // 2)
    if d > 0:
        g[a:a + d] = np.linspace(1.0, low, d)
    u = min(int(up_ms / 1000 * SR), max(0, n - b))
    if u > 0:
        g[b:b + u] = np.linspace(low, 1.0, u)
    return g


def write_wav(path: Path, a: np.ndarray) -> Path:
    import soundfile as sf

    sf.write(str(path), np.clip(a, -1, 1), SR, subtype="FLOAT")
    return path


def _measure(path: Path) -> tuple[Optional[float], Optional[float]]:
    from engine.media.ffmpeg import measure_loudness

    m = measure_loudness(path)
    return m.get("integrated_lufs"), m.get("true_peak_db")


def loudnorm(src: Path, dst: Path, lufs: float, tp: float, tol: float = 0.3) -> dict:
    """Hit the integrated-loudness target precisely: measured static gain + 4x-oversampled peak
    limiter (with headroom for AAC overshoot), re-measured and corrected up to 3 times.
    Static gain keeps the mix dynamics intact (no pumping from dynamic loudnorm)."""
    import soundfile as sf

    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    i0, _ = _measure(src)
    if i0 is None or i0 < -70:              # silent: nothing to normalise
        sf.write(str(dst), x, sr, subtype="PCM_16")
        return {"input_i": i0}
    ceiling = 10 ** ((tp - 0.7) / 20)       # margin: AAC encoding adds ~0.3-0.6 dB of overshoot
    gain_db = lufs - i0
    prev: Optional[tuple[float, float]] = None
    i1 = None
    for _ in range(6):
        y = _true_peak_limit(x * 10 ** (gain_db / 20), ceiling, sr)
        sf.write(str(dst), y, sr, subtype="PCM_16")
        i1, _ = _measure(dst)
        if i1 is None or abs(i1 - lufs) <= tol:
            break
        # the limiter eats loudness on peaky speech (screams): secant step on the measured
        # gain->loudness slope instead of a 1:1 correction
        slope = 1.0
        if prev and abs(gain_db - prev[0]) > 1e-3:
            slope = min(1.0, max(0.15, (i1 - prev[1]) / (gain_db - prev[0])))
        prev = (gain_db, i1)
        gain_db += min(12.0, (lufs - i1) / slope)
    return {"input_i": i0, "output_i": i1, "gain_db": round(gain_db, 2)}


def _true_peak_limit(y: np.ndarray, ceiling: float, sr: int, release_ms: float = 80.0) -> np.ndarray:
    """Look-ahead peak limiter on a 4x-oversampled envelope (approximates true peak)."""
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import resample_poly

    over = np.abs(resample_poly(y, 4, 1, axis=0)).max(axis=1)
    peak = over.reshape(-1, 4).max(axis=1) if over.size % 4 == 0 else np.abs(y).max(axis=1)
    peak = peak[: y.shape[0]]
    if peak.max() <= ceiling:
        return y
    look = int(0.005 * sr)
    need = np.minimum(1.0, ceiling / np.maximum(maximum_filter1d(peak, size=2 * look + 1), 1e-9))
    # smooth gain: instant attack (already look-ahead via max filter), exponential release
    rel = np.exp(-1.0 / (release_ms / 1000 * sr))
    g = np.empty_like(need)
    cur = 1.0
    step = 64                                # control-rate loop for speed
    for i in range(0, need.size, step):
        t = need[i:i + step].min()
        cur = t if t < cur else rel ** step * cur + (1 - rel ** step) * t
        g[i:i + step] = cur
    return (y * g[:, None]).astype(np.float32)


def render_tracks(src: Path, clip_start: float, clip_dur: float, keep: list[tuple[float, float]],
                  speech_out: list[tuple[float, float]], out_dir: Path, lufs: float, tp: float,
                  plan: Optional[SongPlan] = None, voice_only: Optional[Path] = None) -> dict:
    """Write clean.wav (always) and mixed.wav (when a song plan is given). Returns paths + stats.

    `voice_only`: a Demucs voice stem covering exactly the clip window; the song is mixed over it instead
    of the original audio (source music removed), while clean.wav keeps the original sound."""
    voice = apply_cuts(decode(src, clip_start, clip_dur), keep)
    base = apply_cuts(decode(voice_only, 0.0, clip_dur), keep) if voice_only is not None else voice
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"duration": voice.shape[0] / SR}
    tmp = write_wav(out_dir / "_clean_raw.wav", voice)
    loudnorm(tmp, out_dir / "clean.wav", lufs, tp)
    tmp.unlink(missing_ok=True)
    res["clean"] = str(out_dir / "clean.wav")
    if plan is not None:
        mixed, stats = mix(base, speech_out, plan)
        tmp = write_wav(out_dir / "_mixed_raw.wav", mixed)
        loudnorm(tmp, out_dir / "mixed.wav", lufs, tp)
        tmp.unlink(missing_ok=True)
        res["mixed"] = str(out_dir / "mixed.wav")
        res["stats"] = stats
    return res
