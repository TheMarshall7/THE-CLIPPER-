"""Optional end card (PRD M8): 1.5-3 s song card appended after the clip.

Cover art (if the song has one), song title, artist and a CTA, in the artist's brand colours.
Only added when a song was actually mixed in; the "no song" clean version never carries it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from engine.config import VideoSettings
from engine.media import ffmpeg
from engine.media.ffmpeg import run

_FONTS = ["ariblk.ttf", "arialbd.ttf", "impact.ttf", "DejaVuSans-Bold.ttf"]


def _font(size: int):
    from PIL import ImageFont

    for name in _FONTS:
        for root in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
            p = Path(root) / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def make_card_image(out: Path, width: int, height: int, title: str, artist: str, cta: str,
                    accent: str = "#FFD400", background: str = "#0B0B10", cover: Optional[Path] = None) -> Path:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), background)
    d = ImageDraw.Draw(img)
    y = int(height * 0.24)
    if cover and Path(cover).exists():
        side = int(width * 0.56)
        art = Image.open(cover).convert("RGB").resize((side, side))
        img.paste(art, ((width - side) // 2, y))
        y += side + int(height * 0.04)
    else:
        y = int(height * 0.34)
    maxw = int(width * 0.84)
    for text, size, color in ((title, int(width * 0.085), "#FFFFFF"), (artist, int(width * 0.05), accent)):
        f = _font(size)
        for line in _wrap(d, text, f, maxw):
            w = d.textlength(line, font=f)
            d.text(((width - w) / 2, y), line, font=f, fill=color, stroke_width=2, stroke_fill="#000000")
            y += int(size * 1.2)
        y += int(height * 0.015)
    if cta:
        f = _font(int(width * 0.042))
        y += int(height * 0.03)
        pad = int(width * 0.03)
        lines = _wrap(d, cta, f, maxw - 2 * pad)
        bw = max(d.textlength(l, font=f) for l in lines) + 2 * pad
        bh = int(len(lines) * width * 0.042 * 1.25) + pad
        d.rounded_rectangle([(width - bw) / 2, y, (width + bw) / 2, y + bh], radius=pad, outline=accent, width=4)
        ty = y + pad // 2
        for line in lines:
            w = d.textlength(line, font=f)
            d.text(((width - w) / 2, ty), line, font=f, fill="#FFFFFF")
            ty += int(width * 0.042 * 1.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def append_end_card(video: Path, out: Path, card_png: Path, seconds: float, cfg: VideoSettings) -> Path:
    """Re-encode `video` + a still card (with silent audio) into `out`."""
    from engine.video.compose import encoder_args

    info = ffmpeg.probe(video)
    fps = max(24, min(cfg.max_fps, round(info.fps) or 30))
    fc = (f"[0:v]fps={fps},setsar=1,format=yuv420p[v0];"
          f"[1:v]fps={fps},setsar=1,format=yuv420p,fade=t=in:st=0:d=0.3[v1];"
          f"[0:a]aresample=48000,aformat=channel_layouts=stereo[a0];"
          f"[2:a]aformat=channel_layouts=stereo[a1];"
          f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-i", str(video),
           "-loop", "1", "-framerate", str(fps), "-t", f"{seconds:.2f}", "-i", str(card_png),
           "-f", "lavfi", "-t", f"{seconds:.2f}", "-i", "anullsrc=r=48000:cl=stereo",
           "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
           "-c:a", "aac", "-b:a", cfg.audio_bitrate, *encoder_args(cfg), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    run(cmd)
    return out
