"""Vertical 9:16 composition + render (PRD §10, M8).

Graph:  source -> jump cuts (trim/concat) -> layout (cam top / gameplay bottom, full cam,
        or gameplay with blurred fill) -> punch-in on the payoff -> divider -> captions/hook/
        sound tag (ASS) -> encode.  Audio comes from a pre-rendered WAV (mixing.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from engine.config import VideoSettings
from engine.media.ffmpeg import MediaError, run

_NVENC_OK: Optional[bool] = None


def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def _hex(c: str) -> str:
    return "0x" + c.lstrip("#")


def encoder_args(cfg: VideoSettings) -> list[str]:
    global _NVENC_OK
    enc = cfg.encoder
    if enc == "auto":
        if _NVENC_OK is None:
            from engine.hardware import ffmpeg_has_nvenc
            _NVENC_OK = ffmpeg_has_nvenc()
        enc = "h264_nvenc" if _NVENC_OK else "libx264"
    if enc == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", str(max(12, cfg.crf - 2)), "-b:v", "0",
                "-spatial-aq", "1", "-profile:v", "high"]
    return ["-c:v", "libx264", "-crf", str(cfg.crf), "-preset", cfg.preset, "-profile:v", "high"]


@dataclass
class RenderSpec:
    src: Path                    # media file containing the clip
    src_start: float             # where the clip starts inside src
    duration: float              # source duration of the clip window (before jump cuts)
    src_w: int
    src_h: int
    src_fps: float
    layout: dict                 # from webcam.Layout.to_dict() (possibly manually edited)
    ass_file: Optional[Path]
    out: Path
    audio: Optional[Path] = None             # pre-rendered audio for the cut timeline
    keep: list[tuple[float, float]] = field(default_factory=list)   # jump-cut plan, relative to src_start
    punch_at: Optional[float] = None         # output-timeline second to punch in
    gameplay_focus_x: float = 0.5
    gameplay_mode: Optional[str] = None


def _lerp_expr(points: list[tuple[float, float]]) -> str:
    """ffmpeg expression of t: straight lines through (t, value) points, held flat before the first / after the last.
    Built as a balanced tree of if(lt()) so it stays shallow however many points there are."""
    segs = list(zip(points[:-1], points[1:]))

    def build(lo: int, hi: int) -> str:
        if hi - lo == 1:
            (t0, a), (t1, b) = segs[lo]
            if t1 - t0 < 1e-3:
                return f"{b:.1f}"
            return f"({a:.1f}+({b - a:.1f})*(clip(t,{t0:.3f},{t1:.3f})-{t0:.3f})/{t1 - t0:.3f})"
        mid = (lo + hi) // 2
        return f"if(lt(t,{segs[mid][0][0]:.3f}),{build(lo, mid)},{build(mid, hi)})"
    return build(0, len(segs))


def cam_window(layout: dict, W: int, cam_h: int, sw: int, sh: int, max_zoom: float = 0.0) -> tuple[float, float, float, float, float]:
    """Source window (rw, rh, rx, ry) shown in the camera panel and the zoom it needs (panel pixels per source pixel).
    max_zoom > 0 widens the window so the picture is never blown up more than that (a tiny cam stays sharp-ish)."""
    x, y, w, h = layout["cam_box"]
    x, y = max(0, x), max(0, y)
    w, h = _even(min(w, sw - x)), _even(min(h, sh - y))
    fb = layout.get("face_box")
    ar = W / cam_h
    rw = max(w, h * ar)
    if fb:
        rw = max(rw, fb[3] * 1.7 * ar, fb[2] * 2.2)           # head + hair + a little shoulder around the face
    if max_zoom:
        rw = max(rw, W / max_zoom)
    rw = min(rw, sw, sh * ar)
    rh = rw / ar
    fcx0, fcy0 = (fb[0] + fb[2] / 2, fb[1] + fb[3] / 2) if fb else (x + w / 2, y + h / 2)
    rx = min(max(0.0, fcx0 - rw / 2), sw - rw)
    ry = min(max(0.0, fcy0 - rh * 0.42), sh - rh)             # face a little above the middle of the panel
    return rw, rh, rx, ry, W / rw


def cam_zoom_needed(layout: dict, cfg: VideoSettings, sw: int, sh: int) -> float:
    """How much the camera panel would be enlarged with no cap (high = blurry)."""
    cam_h = _even(cfg.height * cfg.cam_ratio)
    return cam_window(layout, cfg.width, cam_h, sw, sh, 0.0)[4]


def build_filter(spec: RenderSpec, cfg: VideoSettings) -> str:
    W, H = cfg.width, cfg.height
    sw, sh = spec.src_w, spec.src_h
    lay = spec.layout.get("layout", "gameplay_only")
    mode = spec.gameplay_mode or cfg.gameplay_mode
    fps = min(cfg.max_fps, round(spec.src_fps) or 30)
    chains: list[str] = []
    zoom = cfg.punch_in_zoom
    punch = spec.punch_at if cfg.punch_in else None
    pin = f"between(t,{punch:.3f},{punch + 1.6:.3f})" if punch is not None else None

    # ---- jump cuts on the video timeline (fps normalised first so concat is clean)
    keep = spec.keep or [(0.0, spec.duration)]
    if len(keep) > 1:
        chains.append(f"[0:v]fps={fps},split={len(keep)}" + "".join(f"[k{i}]" for i in range(len(keep))))
        for i, (a, b) in enumerate(keep):
            chains.append(f"[k{i}]trim=start={a:.4f}:end={b:.4f},setpts=PTS-STARTPTS[t{i}]")
        chains.append("".join(f"[t{i}]" for i in range(len(keep))) + f"concat=n={len(keep)}:v=1:a=0[src]")
    else:
        chains.append(f"[0:v]fps={fps}[src]")

    def gameplay(label_in: str, out: str, gw: int, gh: int, m: str) -> None:
        if m == "fill":
            aspect = gw / gh
            cw = min(sw, _even(sh * aspect))
            ch = min(sh, _even(cw / aspect))
            cx = int(min(max(0, spec.gameplay_focus_x * sw - cw / 2), sw - cw))
            cy = (sh - ch) // 2
            chains.append(f"[{label_in}]crop={cw}:{ch}:{cx}:{cy},scale={gw}:{gh}:flags=lanczos,setsar=1[{out}]")
        else:  # fit_blur: whole frame (keeps HUD) over a blurred, darkened fill
            chains.append(f"[{label_in}]split=2[{out}a][{out}b]")
            chains.append(f"[{out}a]scale={gw}:{gh}:force_original_aspect_ratio=increase,crop={gw}:{gh},"
                          f"boxblur=24:2,eq=brightness=-0.10:saturation=0.8[{out}bg]")
            fg_w = gw if gh >= gw * sh / sw else _even(gh * sw / sh)
            chains.append(f"[{out}b]scale={fg_w}:-2:flags=lanczos[{out}fg]")
            chains.append(f"[{out}bg][{out}fg]overlay=(W-w)/2:(H-h)/2,setsar=1[{out}]")

    def punch_in(label_in: str, out: str, w: int, h: int, cx: float, cy: float) -> None:
        """Hard-cut zoom during the payoff (the 'punch-in' editors use on reactions)."""
        if not pin:
            chains.append(f"[{label_in}]null[{out}]")
            return
        zw, zh = _even(w / zoom), _even(h / zoom)
        x = int(min(max(0, cx - zw / 2), w - zw))
        y = int(min(max(0, cy - zh / 2), h - zh))
        chains.append(f"[{label_in}]split=2[{out}n][{out}z]")
        chains.append(f"[{out}z]crop={zw}:{zh}:{x}:{y},scale={w}:{h}:flags=lanczos[{out}zz]")
        chains.append(f"[{out}n][{out}zz]overlay=0:0:enable='{pin}'[{out}]")

    if lay == "split" and spec.layout.get("cam_box"):
        cam_h = _even(H * cfg.cam_ratio)
        game_h = H - cam_h
        x, y, w, h = spec.layout["cam_box"]
        x, y = max(0, x), max(0, y)
        w, h = _even(min(w, sw - x)), _even(min(h, sh - y))
        chains.append("[src]split=2[srcA][srcB]")
        # A window with the PANEL's aspect ratio that contains the whole cam box and head, centred on the face, but never
        # enlarged more than max_cam_zoom (beyond that the picture just turns to mush).
        fb = spec.layout.get("face_box")
        rw, rh, rx, ry, zoom = cam_window(spec.layout, W, cam_h, sw, sh, cfg.max_cam_zoom)
        fcx0, fcy0 = (fb[0] + fb[2] / 2, fb[1] + fb[3] / 2) if fb else (x + w / 2, y + h / 2)
        rw_i, rh_i, rx_i, ry_i = _even(rw), _even(rh), int(rx), int(ry)
        sharpen = ",unsharp=5:5:0.7:5:5:0.0" if zoom > 1.4 else ""     # enlarged pictures lose edge contrast: win a little back
        track = spec.layout.get("face_track") if cfg.face_tracking and fb else None
        if track and len(track) >= 2:
            # the window follows the face: same size, moving centre (clamped inside the picture)
            xs = [(t, min(max(0.0, cx - rw_i / 2), sw - rw_i)) for t, cx, cy in track]
            ys = [(t, min(max(0.0, cy - rh_i * 0.42), sh - rh_i)) for t, cx, cy in track]
            chains.append(f"[srcA]crop={rw_i}:{rh_i}:x='{_lerp_expr(xs)}':y='{_lerp_expr(ys)}',"
                          f"scale={W}:{cam_h}:flags=lanczos{sharpen},setsar=1[cam0]")
            fcx, fcy = W / 2, cam_h * 0.42
        else:
            chains.append(f"[srcA]crop={rw_i}:{rh_i}:{rx_i}:{ry_i},scale={W}:{cam_h}:flags=lanczos{sharpen},setsar=1[cam0]")
        if track and len(track) >= 2:
            pass
        elif fb:  # face centre inside the scaled cam panel
            fcx = (fcx0 - rx_i) * W / rw_i
            fcy = (fcy0 - ry_i) * cam_h / rh_i
        else:
            fcx, fcy = W / 2, cam_h / 2
        punch_in("cam0", "cam", W, cam_h, fcx, fcy)
        gameplay("srcB", "game", W, game_h, mode)
        chains.append("[cam][game]vstack=inputs=2[stack0]")
        if cfg.divider:
            chains.append(f"[stack0]drawbox=x=0:y={cam_h - 3}:w={W}:h=6:color={_hex(cfg.divider_color)}@1:t=fill[stack]")
        else:
            chains.append("[stack0]null[stack]")
    elif lay == "fullcam":
        cw = min(sw, _even(sh * 9 / 16))
        fb = spec.layout.get("face_box")
        cx_face = fb[0] + fb[2] / 2 if fb else sw / 2
        cx = int(min(max(0, cx_face - cw / 2), sw - cw))
        chains.append(f"[src]crop={cw}:{sh}:{cx}:0,scale={W}:{H}:flags=lanczos,setsar=1[full]")
        fy = (fb[1] + fb[3] / 2) * H / sh if fb else H * 0.4
        punch_in("full", "stack", W, H, W / 2, fy)
    else:
        gameplay("src", "game", W, H, ("fit_blur" if cfg.full_view_mode == "blur" else "fill") if spec.gameplay_mode is None else mode)
        punch_in("game", "stack", W, H, W / 2, H / 2)

    tail = "[stack]format=yuv420p"
    if spec.ass_file:
        tail += f",subtitles={spec.ass_file.name}"
    if cfg.progress_bar:
        out_dur = sum(b - a for a, b in keep)
        chains.append(tail + "[pre]")
        chains.append(f"color=c={_hex(cfg.divider_color)}:s={W}x8:r={fps}[bar]")
        chains.append(f"[pre][bar]overlay=x='-w+w*t/{out_dur:.3f}':y={H - 8}:shortest=1[vout]")
    else:
        chains.append(tail + "[vout]")
    return ";".join(chains)


def render(spec: RenderSpec, cfg: VideoSettings) -> Path:
    spec.out.parent.mkdir(parents=True, exist_ok=True)
    fgraph = build_filter(spec, cfg)
    out_dur = sum(b - a for a, b in (spec.keep or [(0.0, spec.duration)]))
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error",
           "-ss", f"{spec.src_start:.3f}", "-t", f"{spec.duration:.3f}", "-i", str(spec.src)]
    if spec.audio:
        cmd += ["-i", str(spec.audio)]
    cmd += ["-filter_complex", fgraph, "-map", "[vout]"]
    if spec.audio:
        cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", cfg.audio_bitrate, "-ar", "48000"]
    cmd += ["-t", f"{out_dur:.3f}"] + encoder_args(cfg) + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", spec.out.name]
    try:
        # cwd = output dir so the subtitles filter can use a bare relative filename (Windows path escaping)
        run(cmd, cwd=spec.out.parent)
    except MediaError:
        if "h264_nvenc" not in cmd:
            raise
        i, end = cmd.index("-c:v"), cmd.index("-pix_fmt")
        cmd[i:end] = ["-c:v", "libx264", "-crf", str(cfg.crf), "-preset", cfg.preset]
        run(cmd, cwd=spec.out.parent)
    return spec.out


def mux_audio(video: Path, audio: Path, out: Path, bitrate: str = "192k") -> Path:
    """Same picture, different soundtrack (clean version for in-app official sounds)."""
    run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-i", str(video), "-i", str(audio),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", bitrate, "-shortest",
         "-movflags", "+faststart", str(out)])
    return out
