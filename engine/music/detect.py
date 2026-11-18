"""Source-music detection - "the car radio problem" (M4).

Runs once over the whole stream during the audio-first pass (chunked, no extra
download) and stores a per-0.5 s music-probability timeline.

Three independent cues, because no single one is reliable:
  1. beat periodicity  - autocorrelation of the onset envelope at 70-180 BPM over 6 s.
                         Music has a steady pulse; speech and gameplay SFX don't.
  2. chroma stability  - pitch-class profile correlation 0.25-0.5 s apart over 3 s.
                         Notes/chords sustain; speech pitch moves syllable to syllable.
  3. tonal continuity  - harmonic (non-noisy) energy that persists in the GAPS between
                         words. Speech stops between words; a radio doesn't.
A per-stream baseline catches constant streamer BGM (flagged consistently instead of
treated as one-off music).

This is DSP, not a trained classifier: it is tuned to be conservative (when in doubt,
call it music - a false "music" only costs a song placement; a false "no music" puts
our song over someone else's). Upgrade path: a PANNs/YAMNet tagger as a 4th cue.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from engine.media.ffmpeg import SAMPLE_RATE, iter_wav_blocks

HOP_S = 0.5
N_FFT = 2048
HOP = 512                         # 32 ms at 16 kHz
FPS = SAMPLE_RATE / HOP           # frames per second of the STFT
_CHROMA_FB: Optional[np.ndarray] = None


def _chroma_fb() -> np.ndarray:
    global _CHROMA_FB
    if _CHROMA_FB is None:
        import librosa
        _CHROMA_FB = librosa.filters.chroma(sr=SAMPLE_RATE, n_fft=N_FFT)
    return _CHROMA_FB


def _stft_mag(x: np.ndarray) -> np.ndarray:
    n = 1 + (len(x) - N_FFT) // HOP
    if n <= 0:
        return np.zeros((N_FFT // 2 + 1, 0), np.float32)
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(N_FFT).astype(np.float32)
    return np.abs(np.fft.rfft(frames, axis=1)).T.astype(np.float32)


def frame_cues(x: np.ndarray) -> dict[str, np.ndarray]:
    """Per-STFT-frame raw cues for a mono 16 kHz block."""
    S = _stft_mag(x)
    if S.shape[1] == 0:
        return {k: np.zeros(0) for k in ("onset", "chroma", "flatness", "energy")}
    logS = np.log1p(100 * S)
    onset = np.r_[0.0, np.maximum(0, np.diff(logS, axis=1)).mean(axis=0)]
    # spectral whitening: keep only peaks above the local spectral envelope, so broadband
    # noise (engine rumble, wind, brown noise) contributes ~nothing to the pitch profile
    from scipy.ndimage import maximum_filter, median_filter as _mf
    rel = logS - _mf(logS, size=(31, 1), mode="nearest")
    is_max = logS == maximum_filter(logS, size=(3, 1))
    peaks = np.maximum(0.0, rel - 1.0) * is_max
    chroma = _chroma_fb() @ peaks
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    # fraction of frames carrying clear harmonic peaks (stricter threshold)
    tonal_mass = (np.maximum(0.0, rel - 1.5) * is_max)[8:600].sum(axis=0)
    band = S[8:280]                               # ~60 Hz - 2.2 kHz: where tonal music lives
    flat = np.exp(np.log(band + 1e-9).mean(axis=0)) / (band.mean(axis=0) + 1e-9)
    energy = 10 * np.log10((S ** 2).mean(axis=0) + 1e-12)
    return {"onset": onset, "chroma": chroma, "flatness": flat, "energy": energy, "tonal_mass": tonal_mass}


def _beat_strength(onset: np.ndarray, win_s: float = 6.0) -> np.ndarray:
    """Per-frame max normalised autocorrelation of the onset envelope at 70-180 BPM."""
    w = int(win_s * FPS)
    lo, hi = int(FPS * 60 / 180), int(FPS * 60 / 70)
    out = np.zeros(onset.size)
    step = int(FPS * HOP_S)
    for c in range(0, onset.size, step):
        seg = onset[max(0, c - w // 2): c + w // 2]
        if seg.size < hi * 2:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[seg.size - 1:]
        if ac[0] <= 1e-9:
            continue
        ac = ac / ac[0]
        # require the pulse to repeat: average the lag and its double (a steady beat has both)
        lags = np.arange(lo, min(hi, ac.size // 2))
        if lags.size == 0:
            continue
        val = 0.6 * ac[lags] + 0.4 * ac[np.minimum(2 * lags, ac.size - 1)]
        out[c:c + step] = val.max()
    return np.clip(out, 0, 1)


def _chroma_stability(chroma: np.ndarray, win_s: float = 3.0) -> np.ndarray:
    """Pitch-class profile similarity 0.35 s apart, averaged over 3 s (on whitened peaks).
    Calibrated on test audio: music ~0.70-0.78, speech/noise ~0.25-0.30."""
    lag = int(0.35 * FPS)
    if chroma.shape[1] <= lag:
        return np.zeros(chroma.shape[1])
    sim = (chroma[:, lag:] * chroma[:, :-lag]).sum(axis=0)
    sim = np.r_[sim, np.full(lag, sim[-1])]
    return uniform_filter1d(sim, size=int(win_s * FPS))


def analyze_block(x: np.ndarray, speech_mask: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
    """Return per-frame cue arrays and a raw music probability for one block."""
    cues = frame_cues(x)
    if cues["onset"].size == 0:
        return {"prob": np.zeros(0)}
    beat = _beat_strength(cues["onset"])
    stab = _chroma_stability(cues["chroma"])
    tonal = (cues["tonal_mass"] > 0.5).astype(float)
    # harmonic content that persists - especially through speech gaps, where speech stops but a radio doesn't
    if speech_mask is not None and speech_mask.size >= tonal.size:
        gaps = (~speech_mask[: tonal.size]).astype(float)
        cont = uniform_filter1d(tonal * gaps, size=int(3 * FPS)) / np.maximum(uniform_filter1d(gaps, size=int(3 * FPS)), 0.15)
    else:
        cont = uniform_filter1d(tonal, size=int(3 * FPS))
    stab_n = np.clip((stab - 0.40) / 0.30, 0, 1)
    beat_n = np.clip((beat - 0.15) / 0.35, 0, 1)
    cont_n = np.clip((cont - 0.55) / 0.35, 0, 1)
    # Harmonic stability only counts as music when it is SUSTAINED (>= 4 s) or continuous through
    # gaps. Short tonal bursts are game SFX - sirens, engines, explosion ringing - not a soundtrack.
    from scipy.ndimage import minimum_filter1d
    stab_long = minimum_filter1d(stab_n, size=max(1, int(4 * FPS)), mode="nearest")
    harm = stab_n * np.maximum(cont_n, stab_long)
    # noisy-OR: a steady beat, or sustained harmony, is enough to call it music
    raw = 1 - (1 - 0.85 * beat_n) * (1 - 0.9 * harm) * (1 - 0.3 * cont_n)
    cont = cont_n
    raw = raw * np.clip((cues["energy"] - cues["energy"].min()) / 20, 0, 1)  # silence is not music
    return {"prob": raw, "beat": beat_n, "stability": stab_n, "continuity": np.clip(cont, 0, 1)}


def _pool(a: np.ndarray) -> np.ndarray:
    k = int(FPS * HOP_S)
    n = a.size // k
    return a[: n * k].reshape(n, k).mean(axis=1) if n else np.zeros(0)


def music_timeline(wav: Path, speech_words: Optional[list[dict]] = None, block_s: float = 60.0,
                   context_s: float = 6.0, workers: Optional[int] = None) -> dict[str, np.ndarray]:
    """Whole-stream music probability per 0.5 s hop. Blocks are analyzed several at a time (same result as one by one);
    memory stays bounded to a few groups of blocks."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    sr = SAMPLE_RATE
    workers = workers or min(8, max(1, (os.cpu_count() or 4) // 2))
    probs, beats, stabs, conts = [], [], [], []

    def tasks():
        prev_tail = np.zeros(0, np.float32)
        for off, block in iter_wav_blocks(wav, block_s):
            x = np.concatenate([prev_tail, block])
            mask = None
            if speech_words:
                t0 = off - prev_tail.size / sr
                nfr = 1 + max(0, (x.size - N_FFT) // HOP)
                ft = t0 + np.arange(nfr) * HOP / sr
                mask = np.zeros(nfr, bool)
                for w in speech_words:
                    if w["end"] < t0 or w["start"] > t0 + x.size / sr:
                        continue
                    mask |= (ft >= w["start"]) & (ft <= w["end"])
            yield x, mask, int(prev_tail.size / HOP)
            prev_tail = block[-int(context_s * sr):] if block.size > context_s * sr else block

    def collect(res: dict, skip: int) -> None:
        for acc, key in ((probs, "prob"), (beats, "beat"), (stabs, "stability"), (conts, "continuity")):
            acc.append(_pool(res.get(key, np.zeros(0))[skip:]))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        group: list = []
        for x, mask, skip in tasks():
            group.append((pool.submit(analyze_block, x, mask), skip))
            if len(group) >= workers * 2:              # bound memory: finish a group before reading more audio
                for fut, sk in group:
                    collect(fut.result(), sk)
                group = []
        for fut, sk in group:
            collect(fut.result(), sk)
    prob = np.concatenate(probs) if probs else np.zeros(0)
    prob = median_filter(prob, size=5, mode="nearest") if prob.size else prob
    return {"t": np.arange(prob.size) * HOP_S, "prob": prob,
            "beat": np.concatenate(beats) if beats else prob, "stability": np.concatenate(stabs) if stabs else prob,
            "continuity": np.concatenate(conts) if conts else prob}


def stream_baseline(tl: dict[str, np.ndarray], threshold: float) -> dict:
    p = tl["prob"]
    frac = float((p > threshold).mean()) if p.size else 0.0
    return {"music_fraction": round(frac, 3), "median_prob": round(float(np.median(p)) if p.size else 0.0, 3),
            "constant_bgm": frac > 0.6}


def regions(tl: dict[str, np.ndarray], start: float, end: float, threshold: float,
            constant_bgm: bool = False, min_len: float = 1.5, pad: float = 0.75) -> list[dict]:
    """Music regions inside [start, end] (absolute times), padded for safety."""
    t, p = tl["t"], tl["prob"]
    sel = (t >= start - 2) & (t <= end + 2)
    tt, pp = t[sel], p[sel]
    # hysteresis: enter at `threshold`, stay "on" while above 60% of it (speech over a radio
    # makes the score dip briefly; the radio didn't stop)
    on = np.zeros(pp.size, bool)
    state = False
    for k, v in enumerate(pp):
        state = v > threshold if not state else v > threshold * 0.6
        on[k] = state
    out: list[dict] = []
    i = 0
    while i < on.size:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < on.size and on[j]:
            j += 1
        r0, r1 = float(tt[i]), float(tt[j - 1] + HOP_S)
        if r1 - r0 >= min_len:
            conf = float(pp[i:j].mean())
            out.append({"start": max(start, r0 - pad), "end": min(end, r1 + pad),
                        "type": "streamer_bgm" if constant_bgm else "music",
                        "confidence": round(min(1.0, conf / max(threshold, 1e-6) * 0.6), 2)})
        i = j
    merged: list[dict] = []
    for r in out:  # merge regions separated by < 5 s (a radio doesn't stop for a few seconds)
        if merged and r["start"] - merged[-1]["end"] < 5.0:
            merged[-1]["end"] = r["end"]
        else:
            merged.append(r)
    return [r for r in merged if r["end"] > r["start"]]


SING_CUES = ("sing", "singing", "this song", "the song", "radio", "banger", "my song", "turn it up", "jam", "lyrics", "♪")


def music_is_content(words: list[dict], regs: list[dict]) -> bool:
    """Is the existing music part of the moment (singing along, reacting to the radio)?
    Then we must neither remove it nor add our song (M4 / acceptance test 4)."""
    if not regs:
        return False
    inside = [w["word"].lower() for w in words if any(r["start"] <= w["start"] <= r["end"] for r in regs)]
    text = " ".join(inside)
    if any(c in text for c in SING_CUES):
        return True
    # sung lyrics transcribed by Whisper tend to repeat lines heavily
    toks = text.split()
    return len(toks) >= 12 and len(set(toks)) / len(toks) < 0.35


def save(path: Path, tl: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **tl)


def load(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path))
