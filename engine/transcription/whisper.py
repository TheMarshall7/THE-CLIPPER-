"""Faster-Whisper transcription in quiet-point-cut chunks (PRD §6).

Each chunk is cached to disk as JSON so an interrupted 4-hour stream resumes
where it stopped and is never re-transcribed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

from engine.config import TranscriptionSettings
from engine.media.ffmpeg import iter_speech_chunks

_MODEL_CACHE: dict[tuple, object] = {}


def _add_cuda_dll_dirs() -> None:
    """On Windows, pip-installed nvidia-cublas/cudnn wheels ship DLLs that must be on the search path."""
    if sys.platform != "win32":
        return
    bases: list[Path] = []
    try:
        import nvidia  # type: ignore
        bases += [Path(b) for b in getattr(nvidia, "__path__", [])]
    except ImportError:
        pass
    if getattr(sys, "frozen", False):          # packaged exe: the GPU libraries ship inside the bundle
        bases.append(Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "nvidia")
    for base in bases:
        for sub in ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"):
            d = base / sub / "bin"
            if d.is_dir():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def load_model(cfg: TranscriptionSettings, log: Callable[[str], None] = print):
    from faster_whisper import WhisperModel

    _add_cuda_dll_dirs()
    device = cfg.device
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = cfg.compute_type if cfg.compute_type != "auto" else ("float16" if device == "cuda" else "int8")
    key = (cfg.model, device, compute)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], device
    try:
        model = WhisperModel(cfg.model, device=device, compute_type=compute)
        if device == "cuda":  # surface missing cuBLAS/cuDNN now, not mid-stream
            import numpy as np
            list(model.transcribe(np.zeros(16000, dtype="float32"), language="en")[0])
    except Exception as e:
        if device != "cuda":
            raise
        log(f"CUDA transcription unavailable ({str(e)[:160]}); falling back to CPU int8.")
        device, compute = "cpu", "int8"
        model = WhisperModel(cfg.model, device=device, compute_type=compute)
        key = (cfg.model, device, compute)
    _MODEL_CACHE[key] = model
    return model, device


def transcribe_wav(wav: Path, cache_dir: Path, cfg: TranscriptionSettings,
                   log: Callable[[str], None] = print,
                   progress: Optional[Callable[[float], None]] = None) -> list[dict]:
    """Return segments: {start,end,text,speaker,confidence,words:[{start,end,word,prob}]}.
    All times are absolute seconds on the original media timeline."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = None
    segments: list[dict] = []
    for idx, (offset, audio) in enumerate(iter_speech_chunks(wav, cfg.chunk_seconds)):
        cpath = cache_dir / f"chunk_{idx:04d}.json"
        if cpath.exists():
            data = json.loads(cpath.read_text(encoding="utf-8"))
            if abs(data.get("offset", -1) - offset) < 1e-3:
                segments.extend(data["segments"])
                continue
        if model is None:
            model, device = load_model(cfg, log)
            log(f"Transcribing with Whisper '{cfg.model}' on {device}")
        seg_iter, _info = model.transcribe(
            audio, language=cfg.language, beam_size=cfg.beam_size, word_timestamps=True,
            vad_filter=cfg.vad_filter, vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )
        chunk = []
        for s in seg_iter:
            words = [{"start": round(offset + w.start, 3), "end": round(offset + w.end, 3),
                      "word": w.word.strip(), "prob": round(w.probability, 3)} for w in (s.words or []) if w.word.strip()]
            if not words and not s.text.strip():
                continue
            conf = sum(w["prob"] for w in words) / len(words) if words else None
            chunk.append({
                "start": round(offset + s.start, 3), "end": round(offset + s.end, 3), "text": s.text.strip(),
                "speaker": "streamer",  # single-speaker assumption; diarization is a later add-on
                "confidence": round(conf, 3) if conf is not None else None,
                "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3),
                "words": words,
            })
        cpath.write_text(json.dumps({"offset": offset, "segments": chunk}), encoding="utf-8")
        segments.extend(chunk)
        if progress:
            progress(offset + len(audio) / 16000)
    return segments


def to_srt(segments: list[dict]) -> str:
    def ts(t: float) -> str:
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)) % 1000:03d}"

    out = []
    for i, s in enumerate(segments, 1):
        out.append(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n")
    return "\n".join(out)
