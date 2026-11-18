"""Generate a synthetic, fully authorized test "stream" for repeatable tests.

Speech: Windows SAPI text-to-speech (System.Speech). Video: FFmpeg test pattern
"gameplay" with a static bordered "webcam" overlay. Calm commentary with two
excited moments (louder, faster, higher pitch) so the detector has something real
to find, and a long quiet stretch it should ignore.

Usage: python scripts/make_sample.py [out_dir]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LINES = [  # (start_s, text, excited)
    (2, "Alright chat, we are finally in. Let me just drive around for a bit and see what this city looks like.", False),
    (11, "The map is huge. I'm going to head toward the beach and check out the shops.", False),
    (20, "Okay so the traffic in this game feels a lot more realistic than before.", False),
    (29, "Let me try parking here. Hmm, the parking lot is pretty empty today.", False),
    (38, "Wait. What is that? Is that a shark jumping out of the water? No way! No way! That is insane!", True),
    (47, "Oh my god chat, did you see that? A shark just ate the jet ski! I can't believe they put that in the game!", True),
    (57, "Okay okay. That was crazy. I need a second.", False),
    (64, "Anyway, let's go check the clothing store and pick a new outfit.", False),
    (73, "This jacket is fine I guess. The prices seem normal.", False),
    (82, "I'll grab the sunglasses too.", False),
    (100, "Now let's try the police chase. I'm going to steal this car and see how fast they show up.", False),
    (110, "Here they come. Here they come! Go go go! Let's go! We lost them! We actually lost them!", True),
    (119, "That was the cleanest getaway I have ever done. Chat, clip that!", True),
    (128, "Alright, I'm going to take a break and grab some water. Be right back.", False),
]
DURATION = 150


def tts(text: str, path: Path, excited: bool) -> None:
    rate = 3 if excited else 0
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.Rate = {rate}; $s.Volume = 100;"
        f"$s.SetOutputToWaveFile('{path}');"
        f"$s.Speak(@'\n{text}\n'@);$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)


def main(out_dir: Path, radio: Path | None = None, radio_until: float = 33.0, name: str = "sample_stream.mp4") -> Path:
    """radio: optional music file mixed under [0, radio_until] at radio level (car radio scenario)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = out_dir / "parts"
    parts.mkdir(exist_ok=True)
    inputs, filters = [], []
    for i, (start, text, excited) in enumerate(LINES):
        wav = parts / f"line_{i:02d}.wav"
        if not wav.exists():
            tts(text, wav, excited)
        inputs += ["-i", str(wav)]
        # excited lines: +10 dB and pitched up ~15% (asetrate) to mimic a raised voice
        fx = "asetrate=22050*1.15,aresample=22050,volume=10dB" if excited else "volume=-2dB"
        filters.append(f"[{i + 1}:a]{fx},adelay={start * 1000}|{start * 1000},aresample=48000[a{i}]")
    n = len(LINES)
    mix = "".join(f"[a{i}]" for i in range(n))
    extra = 0
    if radio:
        inputs += ["-i", str(radio)]
        filters.append(f"[{n + 1}:a]atrim=0:{radio_until},afade=t=out:st={radio_until - 1}:d=1,volume=-14dB,"
                       f"aformat=channel_layouts=mono,aresample=48000[radio]")
        mix += "[radio]"
        extra = 1
    # game audio bed: quiet brown noise
    filters.insert(0, f"anoisesrc=d={DURATION}:c=brown:a=0.02,aresample=48000[bed]")
    filters.append(f"[bed]{mix}amix=inputs={n + 1 + extra}:normalize=0:duration=first,alimiter=limit=0.95[aout]")
    # video: test pattern gameplay + webcam box (bordered) in the top-left corner
    v = (f"testsrc2=s=1920x1080:r=30:d={DURATION}[game];"
         f"mandelbrot=s=480x270:r=30,trim=duration={DURATION}[cam];"
         "[cam]pad=496:286:8:8:color=white[camb];"
         "[game][camb]overlay=40:40,format=yuv420p[vout]")
    out = out_dir / name
    cmd = ["ffmpeg", "-hide_banner", "-y", "-v", "error", "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=mono:d={DURATION}",
           *inputs, "-filter_complex", ";".join([v] + filters), "-map", "[vout]", "-map", "[aout]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "160k", "-t", str(DURATION), str(out)]
    subprocess.run(cmd, check=True)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", type=Path,
                    default=Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "generated")
    ap.add_argument("--radio", type=Path, help="music file to play as 'car radio' under the first 33 s")
    ap.add_argument("--name", default="sample_stream.mp4")
    a = ap.parse_args()
    print(main(a.out_dir, a.radio, name=a.name))
