import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "tests" / "fixtures" / "generated"
sys.path.insert(0, str(ROOT))


def _ensure_song(name: str, bpm: float, root: float, seed: int) -> Path:
    p = GEN / f"{name}.wav"
    if not p.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_song.py"), str(p), "--bpm", str(bpm),
                        "--root", str(root), "--seed", str(seed)], check=True)
    return p


def _to16k(src: Path) -> Path:
    dst = src.with_name(src.stem + "_16k.wav")
    if not dst.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)], check=True)
    return dst


@pytest.fixture(scope="session")
def our_song() -> Path:
    return _ensure_song("our_song", 140, 92.5, 3)


@pytest.fixture(scope="session")
def radio_song() -> Path:
    return _ensure_song("radio_song", 104, 130, 7)


@pytest.fixture(scope="session")
def samples(radio_song) -> dict:
    """TTS 'streams' with and without car radio (Windows SAPI; skipped elsewhere)."""
    if sys.platform != "win32":
        pytest.skip("sample generation uses Windows text-to-speech")
    out = {}
    for name, radio in (("sample_stream", None), ("sample_radio", radio_song)):
        mp4 = GEN / f"{name}.mp4"
        if not mp4.exists():
            cmd = [sys.executable, str(ROOT / "scripts" / "make_sample.py"), str(GEN), "--name", mp4.name]
            if radio:
                cmd += ["--radio", str(radio)]
            subprocess.run(cmd, check=True)
        out[name] = {"mp4": mp4, "wav": _to16k(mp4)}
    return out


@pytest.fixture()
def tmp_settings(tmp_path):
    from engine.config import Settings

    s = Settings(storage_dir=str(tmp_path / "storage"), library_dir=str(tmp_path / "library"))
    s.ai.mode = "heuristic"
    type(s).db_path = property(lambda self: tmp_path / "test.db")  # isolate DB per test
    return s
