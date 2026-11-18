"""Finishing touches applied to a rendered clip (after QC):

  lead-in      the best 1-3 seconds play first, then the whole clip plays from its start ("watch till the end" hook)
  loop ending  the last half second dissolves into the clip's first frame, so a looping player feels seamless
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from engine.config import VideoSettings
from engine.media.ffmpeg import probe, run
from engine.video.compose import encoder_args


def _encode(src: Path, dst: Path, graph: str, maps: list[str], cfg: VideoSettings, has_audio: bool, t: Optional[float] = None) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-i", str(src), "-filter_complex", graph]
    for m in maps:
        cmd += ["-map", m]
    if t:
        cmd += ["-t", f"{t:.3f}"]
    cmd += encoder_args(cfg)
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", cfg.audio_bitrate, "-ar", "48000"]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    try:
        run(cmd)
    except Exception:
        if "h264_nvenc" not in cmd:
            raise
        i, end = cmd.index("-c:v"), cmd.index("-pix_fmt")
        cmd[i:end] = ["-c:v", "libx264", "-crf", str(cfg.crf), "-preset", cfg.preset]
        run(cmd)


def add_lead_in(video: Path, dst: Path, peak_t: float, seconds: float, cfg: VideoSettings) -> Path:
    """Best part first (`seconds` long, starting just before `peak_t`), then the full clip."""
    info = probe(video)
    secs = max(0.8, min(seconds, info.duration / 3))
    t0 = min(max(0.0, peak_t - 0.4), max(0.0, info.duration - secs))
    if info.has_audio:
        g = (f"[0:v]split=2[va][vb];[va]trim=start={t0:.3f}:duration={secs:.3f},setpts=PTS-STARTPTS[lv];[vb]setpts=PTS-STARTPTS[fv];"
             f"[0:a]asplit=2[aa][ab];[aa]atrim=start={t0:.3f}:duration={secs:.3f},asetpts=PTS-STARTPTS,"
             f"afade=t=out:st={max(0.0, secs - 0.06):.3f}:d=0.06[la];[ab]asetpts=PTS-STARTPTS[fa];"
             f"[lv][la][fv][fa]concat=n=2:v=1:a=1[v][a]")
        _encode(video, dst, g, ["[v]", "[a]"], cfg, True)
    else:
        g = (f"[0:v]split=2[va][vb];[va]trim=start={t0:.3f}:duration={secs:.3f},setpts=PTS-STARTPTS[lv];[vb]setpts=PTS-STARTPTS[fv];"
             f"[lv][fv]concat=n=2:v=1:a=0[v]")
        _encode(video, dst, g, ["[v]"], cfg, False)
    return dst


def add_loop_ending(video: Path, dst: Path, cfg: VideoSettings, fade: float = 0.5) -> Path:
    """Dissolve the last `fade` seconds into the first frame; the audio fades out under it."""
    info = probe(video)
    f = min(fade, info.duration / 4)
    still = (f"[0:v]split=2[a][b];[b]trim=end_frame=1,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={f + 0.2:.3f}[still];"
             f"[a][still]xfade=transition=fade:duration={f:.3f}:offset={info.duration - f:.3f}[v]")
    if info.has_audio:
        g = still + f";[0:a]afade=t=out:st={info.duration - f:.3f}:d={f:.3f}[a]"
        _encode(video, dst, g, ["[v]", "[a]"], cfg, True, t=info.duration)
    else:
        _encode(video, dst, still, ["[v]"], cfg, False, t=info.duration)
    return dst


def finish(video: Path, cfg: VideoSettings, peak_t: float, lead: bool, loop: bool) -> Path:
    """Apply the chosen touches to `video` in place and return it."""
    cur = video
    if lead:
        tmp = video.with_name(video.stem + "_lead.mp4")
        add_lead_in(cur, tmp, peak_t, cfg.lead_seconds, cfg)
        tmp.replace(video)
    if loop:
        tmp = video.with_name(video.stem + "_loop.mp4")
        add_loop_ending(video, tmp, cfg)
        tmp.replace(video)
    return video
