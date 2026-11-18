"""Signal A: audio excitement (PRD §7).

Chunked DSP over the mono 16 kHz WAV. Per 50 ms frame we compute RMS, peak,
spectral centroid and an autocorrelation pitch estimate; these are pooled to
0.5 s hops. Excitement is measured as deviation from *this streamer's*
baseline (robust median/MAD over voiced frames plus a rolling 5-minute
median), so a naturally loud streamer is not inflated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from engine.media.ffmpeg import SAMPLE_RATE, iter_wav_blocks

FRAME = 800          # 50 ms
HOP_FRAMES = 10      # 0.5 s pooled hop
HOP_S = FRAME * HOP_FRAMES / SAMPLE_RATE
F0_MIN, F0_MAX = 75.0, 600.0


def _frame_features(x: np.ndarray) -> dict[str, np.ndarray]:
    n = x.size // FRAME
    if n == 0:
        return {k: np.zeros(0) for k in ("rms", "peak", "centroid", "f0", "flux")}
    fr = x[: n * FRAME].reshape(n, FRAME)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    peak = np.abs(fr).max(axis=1)
    win = np.hanning(FRAME).astype(np.float32)
    spec = np.abs(np.fft.rfft(fr * win, n=1024, axis=1))
    freqs = np.fft.rfftfreq(1024, 1 / SAMPLE_RATE)
    centroid = (spec * freqs).sum(axis=1) / (spec.sum(axis=1) + 1e-9)
    flux = np.r_[0.0, np.sqrt((np.diff(spec, axis=0).clip(min=0) ** 2).sum(axis=1))]
    # autocorrelation pitch (cheap YIN-lite) on frames loud enough to be voice
    ac = np.fft.irfft(np.abs(np.fft.rfft(fr - fr.mean(axis=1, keepdims=True), n=2048, axis=1)) ** 2, axis=1)
    lo, hi = int(SAMPLE_RATE / F0_MAX), int(SAMPLE_RATE / F0_MIN)
    seg = ac[:, lo:hi]
    lag = seg.argmax(axis=1) + lo
    strength = seg.max(axis=1) / (ac[:, 0] + 1e-9)
    f0 = np.where((strength > 0.45) & (rms > 0.01), SAMPLE_RATE / lag, np.nan)
    return {"rms": rms, "peak": peak, "centroid": centroid, "f0": f0, "flux": flux}


def _pool(a: np.ndarray, fn) -> np.ndarray:
    n = a.size // HOP_FRAMES
    return fn(a[: n * HOP_FRAMES].reshape(n, HOP_FRAMES), axis=1) if n else np.zeros(0)


def compute_features(wav: Path, block_seconds: float = 60.0) -> dict[str, np.ndarray]:
    """Stream the WAV block by block; memory stays ~ block size."""
    acc: dict[str, list[np.ndarray]] = {k: [] for k in ("rms_db", "peak_db", "centroid", "f0", "f0_std", "flux", "env_mod")}
    block_samples = int(block_seconds * SAMPLE_RATE)
    block_samples -= block_samples % (FRAME * HOP_FRAMES)
    for _, x in iter_wav_blocks(wav, block_samples / SAMPLE_RATE):
        f = _frame_features(x)
        if f["rms"].size < HOP_FRAMES:
            continue
        acc["rms_db"].append(20 * np.log10(_pool(f["rms"], np.mean) + 1e-9))
        acc["peak_db"].append(20 * np.log10(_pool(f["peak"], np.max) + 1e-9))
        acc["centroid"].append(_pool(f["centroid"], np.mean))
        acc["flux"].append(_pool(f["flux"], np.mean))
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                acc["f0"].append(_pool(f["f0"], np.nanmedian))
                acc["f0_std"].append(_pool(f["f0"], np.nanstd))
        # envelope modulation 3-8 Hz (laughter is a ~4-6 Hz burst pattern)
        env = f["rms"]
        n = env.size // HOP_FRAMES
        e = env[: n * HOP_FRAMES].reshape(n, HOP_FRAMES)
        e = e - e.mean(axis=1, keepdims=True)
        mod = np.abs(np.fft.rfft(e, axis=1))  # 20 Hz frame rate -> bins of 2 Hz
        acc["env_mod"].append(mod[:, 2:5].sum(axis=1) / (np.abs(env[: n * HOP_FRAMES].reshape(n, HOP_FRAMES)).mean(axis=1) + 1e-6))
    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in acc.items()}
    out["t"] = np.arange(out["rms_db"].size) * HOP_S
    return out


@dataclass
class Baseline:
    loud_median: float
    loud_mad: float
    f0_median: float
    f0_mad: float
    centroid_median: float
    silence_db: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def baseline(feat: dict[str, np.ndarray]) -> Baseline:
    rms = feat["rms_db"]
    if rms.size == 0:
        return Baseline(-30, 6, 180, 40, 1500, -50)
    silence = float(np.percentile(rms, 10))
    active = rms[rms > silence + 10] if (rms > silence + 10).sum() > 20 else rms
    f0 = feat["f0"][~np.isnan(feat["f0"])]

    def mad(a: np.ndarray, floor: float) -> float:
        return max(floor, float(np.median(np.abs(a - np.median(a)))) * 1.4826) if a.size else floor

    return Baseline(
        loud_median=float(np.median(active)), loud_mad=mad(active, 2.0),
        f0_median=float(np.median(f0)) if f0.size else 180.0, f0_mad=mad(f0, 15.0),
        centroid_median=float(np.median(feat["centroid"])) if feat["centroid"].size else 1500.0,
        silence_db=silence,
    )


def excitement(feat: dict[str, np.ndarray], base: Optional[Baseline] = None) -> dict[str, np.ndarray]:
    """Per-hop excitement in [0,1] plus the component z-scores used to explain it."""
    base = base or baseline(feat)
    rms = feat["rms_db"]
    if rms.size == 0:
        return {"t": feat["t"], "score": np.zeros(0)}
    # local baseline handles volume drift across a 4-hour stream
    win = max(3, int(300 / HOP_S)) | 1
    local = median_filter(rms, size=min(win, (rms.size // 2) * 2 + 1), mode="nearest")
    local = np.maximum(local, base.loud_median - 2 * base.loud_mad)
    z_loud = (rms - local) / base.loud_mad
    smooth = uniform_filter1d(rms, size=4)
    z_rise = np.r_[np.zeros(4), (smooth[4:] - smooth[:-4])] / base.loud_mad  # 2 s rise
    f0 = np.nan_to_num(feat["f0"], nan=base.f0_median)
    z_pitch = (f0 - base.f0_median) / base.f0_mad
    z_pvar = np.nan_to_num(feat["f0_std"], nan=0.0) / base.f0_mad
    z_bright = (feat["centroid"] - base.centroid_median) / max(200.0, base.centroid_median * 0.3)
    laugh = np.clip(uniform_filter1d(feat["env_mod"], size=4) - 1.0, 0, None)
    clip = (feat["peak_db"] > -0.5).astype(float)

    raw = (0.40 * z_loud.clip(0) + 0.15 * z_rise.clip(0) + 0.15 * z_pitch.clip(0)
           + 0.10 * z_pvar.clip(0, 4) + 0.10 * z_bright.clip(0) + 0.10 * laugh + 0.05 * clip)
    raw = uniform_filter1d(raw, size=3)
    score = 1 - np.exp(-raw / 1.2)            # saturating map to [0,1]
    voiced = rms > base.silence_db + 8
    return {"t": feat["t"], "score": score, "z_loud": z_loud, "z_pitch": z_pitch, "z_rise": z_rise,
            "laugh": laugh, "voiced": voiced.astype(float)}


def scream_or_laugh(ex: dict[str, np.ndarray], i0: int, i1: int) -> dict:
    sl = slice(max(0, i0), max(i0 + 1, i1))
    return {
        "laughter_like": float(np.mean(ex["laugh"][sl] > 0.8)) if ex["laugh"].size else 0.0,
        "scream_like": float(np.mean((ex["z_loud"][sl] > 3) & (ex["z_pitch"][sl] > 2))) if ex["z_loud"].size else 0.0,
    }


def save(path: Path, feat: dict[str, np.ndarray], ex: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **{f"f_{k}": v for k, v in feat.items()}, **{f"x_{k}": v for k, v in ex.items()})


def load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data = np.load(path)
    feat = {k[2:]: data[k] for k in data.files if k.startswith("f_")}
    ex = {k[2:]: data[k] for k in data.files if k.startswith("x_")}
    return feat, ex
