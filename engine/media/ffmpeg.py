"""FFmpeg / ffprobe wrappers and streaming WAV readers.

Nothing here loads a whole recording into memory: audio is read from a mono
16 kHz WAV in fixed-size blocks via the stdlib `wave` module.
"""
from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

SAMPLE_RATE = 16000


class MediaError(RuntimeError):
    pass


def run(cmd: list[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        raise MediaError(f"{Path(cmd[0]).name} failed ({proc.returncode}): {tail}")
    return proc


@dataclass
class ProbeInfo:
    duration: float
    has_video: bool
    has_audio: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    vcodec: Optional[str] = None
    acodec: Optional[str] = None
    video_duration: Optional[float] = None
    audio_duration: Optional[float] = None


def probe(path: Path | str) -> ProbeInfo:
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic") != 1), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    dur = float(data.get("format", {}).get("duration") or (v or a or {}).get("duration") or 0)
    fps = 0.0
    if v:
        num, _, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1").partition("/")
        try:
            fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            fps = 0.0
    if dur <= 0:
        raise MediaError(f"Could not read a duration from {path}; the file may be corrupted.")
    return ProbeInfo(
        duration=dur, has_video=v is not None, has_audio=a is not None,
        width=int(v.get("width", 0)) if v else 0, height=int(v.get("height", 0)) if v else 0,
        fps=fps, vcodec=v.get("codec_name") if v else None, acodec=a.get("codec_name") if a else None,
        video_duration=float(v["duration"]) if v and v.get("duration") else None,
        audio_duration=float(a["duration"]) if a and a.get("duration") else None,
    )


def extract_audio(src: Path | str, dst_wav: Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Decode any media to mono 16-bit PCM WAV (lossless intermediate for DSP + ASR)."""
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_wav.with_suffix(".part.wav")
    run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1",
         "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(tmp)])
    tmp.replace(dst_wav)
    return dst_wav


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def iter_wav_blocks(path: Path, block_seconds: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (offset_seconds, float32 samples) blocks from a mono 16-bit WAV."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise MediaError("Expected mono 16-bit WAV")
        sr = w.getframerate()
        n = int(block_seconds * sr)
        pos = 0
        while True:
            raw = w.readframes(n)
            if not raw:
                break
            yield pos / sr, np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            pos += len(raw) // 2


def read_wav_range(path: Path, start: float, end: float) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        s = max(0, int(start * sr))
        e = min(w.getnframes(), int(end * sr))
        w.setpos(s)
        raw = w.readframes(max(0, e - s))
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def iter_speech_chunks(path: Path, chunk_seconds: float, search_seconds: float = 15.0) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ~chunk_seconds blocks, each cut at the quietest point near its end so
    words are not split across chunks. Timestamps (offsets) are preserved exactly."""
    sr = SAMPLE_RATE
    carry = np.zeros(0, dtype=np.float32)
    carry_offset = 0.0
    for offset, block in iter_wav_blocks(path, chunk_seconds):
        buf = np.concatenate([carry, block]) if carry.size else block
        buf_offset = carry_offset if carry.size else offset
        if buf.size < (chunk_seconds + search_seconds) * sr:
            carry, carry_offset = buf, buf_offset
            continue
        cut = _quiet_cut(buf, int(chunk_seconds * sr), int(search_seconds * sr))
        yield buf_offset, buf[:cut]
        carry, carry_offset = buf[cut:], buf_offset + cut / sr
    if carry.size:
        yield carry_offset, carry


def _quiet_cut(buf: np.ndarray, target: int, search: int) -> int:
    lo, hi = max(1, target - search), min(len(buf) - 1, target + search // 3)
    win = int(0.25 * SAMPLE_RATE)
    region = buf[lo:hi]
    if region.size < win * 2:
        return target
    nwin = region.size // win
    energy = (region[: nwin * win].reshape(nwin, win) ** 2).mean(axis=1)
    return lo + int(np.argmin(energy)) * win + win // 2


def cut_av(src: Path | str, start: float, end: float, dst: Path) -> Path:
    """Frame-accurate cut (re-encode at high quality) of a local file."""
    run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src),
         "-t", f"{end - start:.3f}", "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
         "-c:a", "aac", "-b:a", "256k", str(dst)])
    return dst


def measure_loudness(path: Path) -> dict:
    """Integrated loudness (LUFS) and true peak (dBTP) via the ebur128 filter."""
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-vn", "-af",
                           "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    text = proc.stderr
    summary = text[text.rfind("Summary:"):]
    out: dict = {}
    for line in summary.splitlines():
        line = line.strip()
        if line.startswith("I:"):
            out["integrated_lufs"] = float(line.split()[1])
        elif line.startswith("Peak:"):
            out["true_peak_db"] = float(line.split()[1])
    return out


def decode_check(path: Path) -> Optional[str]:
    """Decode the whole file; return an error string if anything is wrong."""
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(path), "-f", "null", "-"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    err = proc.stderr.strip()
    return err or (None if proc.returncode == 0 else f"exit {proc.returncode}")


def grab_frame(src: Path, t: float, dst: Path) -> Path:
    # convert to full-range yuvj420p first: FFmpeg 9's JPEG encoder rejects limited-range (TV) YUV,
    # which is what most YouTube/Twitch/NVENC video is
    run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src),
         "-frames:v", "1", "-vf", "scale=out_range=full,format=yuvj420p", "-strict", "unofficial", "-q:v", "3", str(dst)])
    return dst
