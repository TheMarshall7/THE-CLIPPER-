"""Build the self-contained release folder:  release/GTA6Clipper/GTA6Clipper.exe

    .venv/Scripts/python scripts/build_standalone.py

The result needs nothing installed on the target PC (no Python, no FFmpeg). It bundles the engine, the dashboard,
the face model, FFmpeg/ffprobe (copied from this PC) and the NVIDIA CUDA libraries for GPU transcription (~3 GB total).
The Whisper speech model (1.6 GB) is downloaded on the first transcription and cached.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "release"
WORK = ROOT / "storage" / "build"


def main() -> int:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not (ffmpeg and ffprobe):
        print("ffmpeg/ffprobe not found on PATH: needed to bundle them.")
        return 1
    if (OUT / "GTA6Clipper").exists():
        shutil.rmtree(OUT / "GTA6Clipper")
    sep = ";" if sys.platform == "win32" else ":"
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed", "--name", "GTA6Clipper",
           "--distpath", str(OUT), "--workpath", str(WORK), "--specpath", str(WORK), "--paths", str(ROOT),
           "--add-data", f"{ROOT / 'engine' / 'dashboard' / 'static'}{sep}engine/dashboard/static",
           "--add-data", f"{ROOT / 'models' / 'vision'}{sep}models/vision",
           "--collect-all", "faster_whisper", "--collect-all", "ctranslate2", "--collect-all", "librosa",
           "--collect-all", "av", "--collect-all", "onnxruntime",
           "--collect-submodules", "uvicorn", "--collect-submodules", "yt_dlp", "--collect-submodules", "engine",
           "--collect-submodules", "python_multipart", "--collect-submodules", "multipart",
           "--collect-binaries", "nvidia.cublas", "--collect-binaries", "nvidia.cudnn", "--collect-binaries", "nvidia.cuda_nvrtc",
           "--hidden-import", "soundfile",
           "--hidden-import", "tkinter", "--collect-all", "customtkinter",
           str(ROOT / "scripts" / "gui_main.py")]
    print(" ".join(cmd[:8]), "...")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        return r.returncode
    app = OUT / "GTA6Clipper"
    (app / "bin").mkdir(exist_ok=True)
    for exe in (ffmpeg, ffprobe):
        shutil.copy2(exe, app / "bin" / Path(exe).name)
    (app / "README.txt").write_text(
        "GTA 6 Clip Engine - standalone\n\nDouble-click GTA6Clipper.exe. Your browser opens the dashboard.\n"
        "In the window: paste YouTube links, put songs in the Songs folder, press Analyze & make clips.\n\n"
        "Your data (settings, database, clips, songs) is stored in this folder. To keep existing data, set the\n"
        "environment variable GTA6_DATA_DIR to that folder, or copy its config, database, library and storage folders here.\n"
        "The speech model (1.6 GB) downloads the first time a video is transcribed.\n", encoding="utf-8")
    size = sum(f.stat().st_size for f in app.rglob("*") if f.is_file()) / 1e9
    print(f"\nBuilt {app / 'GTA6Clipper.exe'}  ({size:.1f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
