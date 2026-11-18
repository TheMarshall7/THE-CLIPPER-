"""Source-music removal (PRD M4, option 2) with Demucs.

Demucs splits audio into vocals / drums / bass / other. We keep the VOICE stem only, which removes the
game radio or the streamer's BGM so our song can be placed. Game sound effects live mostly in the other
stems, so they are lost under the song; the clean (no-song) export always keeps the untouched original audio.

Optional dependency: `pip install demucs` (pulls PyTorch, ~2 GB). Nothing else in the engine needs it.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

MODEL = "htdemucs"


def available() -> bool:
    return importlib.util.find_spec("demucs") is not None and importlib.util.find_spec("torch") is not None


def voice_stem(src: Path, start: float, dur: float, out_wav: Path, device: Optional[str] = None) -> Path:
    """Extract [start, start+dur] of `src` and write only its voice stem to `out_wav` (48 kHz stereo)."""
    if not available():
        raise RuntimeError("source-music removal needs Demucs: .venv/Scripts/pip install demucs")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="demucs_") as tmp:
        tmpd = Path(tmp)
        piece = tmpd / "piece.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-ss", f"{max(0.0, start):.3f}",
                        "-t", f"{dur:.3f}", "-i", str(src), "-vn", "-ac", "2", "-ar", "44100", str(piece)], check=True)
        cmd = [sys.executable, "-m", "demucs", "-n", MODEL, "--two-stems=vocals", "-o", str(tmpd / "out"), str(piece)]
        if device:
            cmd += ["-d", device]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stem = tmpd / "out" / MODEL / "piece" / "vocals.wav"
        if proc.returncode != 0 or not stem.exists():
            raise RuntimeError(f"demucs failed: {(proc.stderr or proc.stdout)[-300:]}")
        subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-i", str(stem), "-ac", "2",
                        "-ar", "48000", str(out_wav)], check=True)
    return out_wav


def residual_music_db(voice_wav: Path) -> Optional[float]:
    """Crude artifact check: how much of the voice stem's energy sits outside the speech band (warble/leakage)."""
    try:
        import numpy as np
        import soundfile as sf

        x, sr = sf.read(str(voice_wav), dtype="float32", always_2d=True)
        m = x.mean(axis=1)
        spec = np.abs(np.fft.rfft(m)) ** 2
        f = np.fft.rfftfreq(m.size, 1 / sr)
        speech = spec[(f >= 80) & (f <= 6000)].sum()
        rest = spec[(f > 6000)].sum() + spec[(f < 80)].sum()
        return float(10 * np.log10(rest / (speech + 1e-12) + 1e-12))
    except Exception:
        return None
