"""Environment + hardware detection and model recommendations (PRD §15-16, §27)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional

import psutil


def _run(cmd: list[str], timeout: int = 15) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr).strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


@dataclass
class GPUInfo:
    name: str
    vram_mb: int
    driver: Optional[str] = None


@dataclass
class Environment:
    os: str
    python: str
    cpu: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    gpus: list[GPUInfo] = field(default_factory=list)
    cuda_for_whisper: bool = False
    ffmpeg: Optional[str] = None
    ffprobe: Optional[str] = None
    nvenc: bool = False
    node: Optional[str] = None
    yt_dlp: Optional[str] = None
    disk_free_gb: float = 0.0

    @property
    def vram_mb(self) -> int:
        return max((g.vram_mb for g in self.gpus), default=0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vram_mb"] = self.vram_mb
        return d


def detect_gpus() -> list[GPUInfo]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    gpus = []
    if out:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    gpus.append(GPUInfo(parts[0], int(float(parts[1])), parts[2] if len(parts) > 2 else None))
                except ValueError:
                    pass
    return gpus


def ffmpeg_has_nvenc() -> bool:
    out = _run(["ffmpeg", "-hide_banner", "-encoders"])
    if not out or "h264_nvenc" not in out:
        return False
    # Encoders can be listed but unusable (no driver). Do a 1-frame test encode.
    test = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
         "-c:v", "h264_nvenc", "-f", "null", "-"],
        capture_output=True, text=True, timeout=30,
    )
    return test.returncode == 0


def detect_environment(disk_path: str = ".") -> Environment:
    vm = psutil.virtual_memory()
    first = lambda s: s.splitlines()[0] if s else None  # noqa: E731
    try:
        import ctranslate2
        cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:  # pragma: no cover - import failure
        cuda = False
    env = Environment(
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        python=sys.version.split()[0],
        cpu=platform.processor() or platform.machine(),
        cpu_cores=psutil.cpu_count(logical=False) or 1,
        cpu_threads=psutil.cpu_count(logical=True) or 1,
        ram_gb=round(vm.total / 1e9, 1),
        gpus=detect_gpus(),
        cuda_for_whisper=cuda,
        ffmpeg=first(_run(["ffmpeg", "-version"])),
        ffprobe=first(_run(["ffprobe", "-version"])),
        node=_run(["node", "--version"]),
        disk_free_gb=round(shutil.disk_usage(os.path.abspath(disk_path)).free / 1e9, 1),
    )
    try:
        import yt_dlp
        env.yt_dlp = yt_dlp.version.__version__
    except Exception:
        env.yt_dlp = None
    env.nvenc = bool(env.ffmpeg) and ffmpeg_has_nvenc()
    return env


@dataclass
class Recommendation:
    whisper_model: str
    whisper_device: str
    whisper_compute: str
    max_concurrent_jobs: int
    encoder: str
    notes: list[str]


def recommend(env: Environment) -> Recommendation:
    notes: list[str] = []
    vram = env.vram_mb
    if env.cuda_for_whisper and vram >= 6000:
        wm, dev, ct = "large-v3-turbo", "cuda", "float16"
        notes.append(f"Whisper large-v3-turbo on GPU ({vram} MB VRAM): best accuracy/speed balance.")
    elif env.cuda_for_whisper and vram >= 3500:
        wm, dev, ct = "small", "cuda", "float16"
        notes.append("Whisper 'small' on GPU: limited VRAM.")
    elif env.ram_gb >= 16:
        wm, dev, ct = "small", "cpu", "int8"
        notes.append("No usable CUDA GPU: Whisper 'small' int8 on CPU. Expect roughly 0.3-0.6x realtime.")
    else:
        wm, dev, ct = "base", "cpu", "int8"
        notes.append("Low RAM: Whisper 'base' on CPU. Accuracy will be lower.")

    # One GPU => transcription jobs serialize anyway. CPU-only => allow 2 on big machines.
    if env.cuda_for_whisper:
        conc = 1
    else:
        conc = 2 if env.cpu_threads >= 16 and env.ram_gb >= 24 else 1
    notes.append(f"Max concurrent stream jobs: {conc} (seven streams are queued, not run at once).")
    return Recommendation(wm, dev, ct, conc, "h264_nvenc" if env.nvenc else "libx264", notes)
