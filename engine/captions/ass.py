"""Word-by-word animated captions + hook card + sound tag as an ASS file (PRD §11, M8).

Word times come from Whisper on the ORIGINAL audio, remapped through the jump-cut
plan, so captions stay synced with what the viewer hears.

Layout respects short-form safe zones (1080x1920): nothing important in the top
~150 px (status bar / search), the bottom ~420 px (caption, buttons) or the right
~140 px (like/comment/share rail).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from engine.config import CaptionStyle

EMPHASIS = re.compile(r"^(no+|way|what+|wait|insane|crazy|omg|god|holy|bro+|yes+|go+|lets|let's|wow|wtf|clip|"
                      r"unbelievable|actually|never|lost|won|dead|run|shark)[!?.,]*$", re.I)
SAFE_TOP, SAFE_BOTTOM, SAFE_RIGHT = 150, 420, 140


def _ass_color(hex_rgb: str, alpha: int = 0) -> str:
    h = hex_rgb.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _ts(t: float) -> str:
    t = max(0.0, t)
    cs_total = int(round(t * 100))
    h, rem = divmod(cs_total, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def group_lines(words: list[dict], max_words: int, max_gap: float = 0.6) -> list[list[dict]]:
    lines: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and (len(cur) >= max_words or w["start"] - cur[-1]["end"] > max_gap
                    or re.search(r"[.!?]$", cur[-1]["word"])
                    or sum(len(x["word"]) for x in cur) + len(w["word"]) > 18):
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def caption_y(style: CaptionStyle, layout: str, height: int, cam_ratio: float) -> int:
    cam_h = int(height * cam_ratio)
    if style.position == "upper":
        return max(SAFE_TOP + 120, int(height * 0.22)) if layout != "split" else cam_h + 170
    if style.position == "seam":
        return cam_h + 90 if layout == "split" else int(height * 0.62)
    # "lower": lower-middle of the gameplay area, clear of the bottom UI and the minimap row
    return min(height - SAFE_BOTTOM - 60, int(height * 0.64))


def build_ass(words: list[dict], dur: float, style: CaptionStyle, width: int, height: int, layout: str,
              cam_ratio: float, hook: Optional[str] = None, attribution: Optional[str] = None,
              sound_tag: Optional[tuple[str, float, float]] = None, band_bottom: Optional[int] = None) -> str:
    """words: [{start,end,word}] on the OUTPUT timeline (seconds from clip start).
    band_bottom: for gameplay-only clips with a blurred fill, the y where the gameplay band ends -
    captions then sit in the clean space below it instead of over the action."""
    prim = _ass_color(style.primary_color)
    hi = _ass_color(style.highlight_color)
    emph = _ass_color(style.emphasis_color)
    outl = _ass_color(style.outline_color)
    y = caption_y(style, layout, height, cam_ratio)
    if band_bottom and style.position == "lower":
        y = min(height - SAFE_BOTTOM - 40, band_bottom + int(style.font_size * 1.3))
    cx = (width - SAFE_RIGHT // 2) // 2          # nudge left of the like/share rail
    hook_size = int(style.font_size * 0.78)
    # title card looks: (text colour, box/outline colour, border style 3=box 1=outline, thickness)
    hook_text, hook_box, hook_bs, hook_th = {
        "box": ("&H00101010", "&H00FFFFFF", 3, 22),
        "yellow": ("&H00101010", "&H0000D4FF", 3, 22),
        "dark": ("&H00FFFFFF", "&H8C000000", 3, 26),
        "red": ("&H00FFFFFF", "&H002B2BE5", 3, 22),
        "outline": ("&H0000D4FF", "&H00000000", 1, 9),
    }.get(getattr(style, "title_style", "box"), ("&H00101010", "&H00FFFFFF", 3, 22))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{style.font},{style.font_size},{prim},{hi},{outl},&H96000000,-1,0,0,0,100,100,1,0,1,{style.outline},{style.shadow},5,60,{SAFE_RIGHT},0,1
Style: Hook,{style.font},{hook_size},{hook_text},{hook_text},{hook_box},{hook_box},-1,0,0,0,100,100,0,0,{hook_bs},{hook_th},{3 if hook_bs == 1 else 0},5,90,{SAFE_RIGHT},0,1
Style: Tag,Arial,34,&H00FFFFFF,&H00FFFFFF,&H64000000,&H64000000,-1,0,0,0,100,100,0,0,3,12,0,7,0,0,0,1
Style: Attr,Arial,28,&H50FFFFFF,&H50FFFFFF,&H80000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,8,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    rel = [w for w in words if w["end"] > 0 and w["start"] < dur]
    for line in group_lines(rel, style.max_words_per_line):
        toks = [_escape(w["word"].upper() if style.uppercase else w["word"]) for w in line]
        base_col = [emph if EMPHASIS.match(w["word"].strip("\"'")) else prim for w in line]
        for k, w in enumerate(line):
            start = w["start"]
            end = line[k + 1]["start"] if k + 1 < len(line) else max(w["end"], w["start"] + 0.3)
            if end <= start:
                continue
            parts = []
            for m, tok in enumerate(toks):
                if m == k and style.animation != "none":
                    pop = r"\fscx118\fscy118\t(0,110,\fscx100\fscy100)" if style.animation == "pop" else ""
                    parts.append(f"{{\\c{hi}{pop}}}{tok}{{\\r}}")
                elif base_col[m] != prim:
                    parts.append(f"{{\\c{base_col[m]}}}{tok}{{\\r}}")
                else:
                    parts.append(tok)
            events.append(f"Dialogue: 1,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{{\\an5\\pos({cx},{y})}}" + " ".join(parts))
    if hook and style.show_hook:
        if layout == "split":
            hook_y = int(height * cam_ratio)
        elif band_bottom:   # centred in the space above the gameplay band
            band_top = height - band_bottom
            hook_y = max(SAFE_TOP + 110, (SAFE_TOP + band_top) // 2 + 20)
        else:
            hook_y = max(SAFE_TOP + 140, int(height * 0.24))
        wrapped = _wrap(_escape(hook), 22)
        events.insert(0, f"Dialogue: 3,{_ts(0)},{_ts(min(style.hook_seconds, dur))},Hook,,0,0,0,,"
                         f"{{\\an5\\pos({cx},{hook_y})\\fscx90\\fscy90\\t(0,140,\\fscx100\\fscy100)\\fad(60,220)}}{wrapped}")
    if sound_tag:
        text, t0, t1 = sound_tag
        tag_x, tag_y = width - 40, height - 70            # bottom right, just above the progress bar
        events.append(f"Dialogue: 2,{_ts(t0)},{_ts(min(dur, t1))},Tag,,0,0,0,,"
                      f"{{\\an3\\pos({tag_x},{tag_y})\\fad(250,300)}}{_escape(text)}")
    if attribution:
        events.append(f"Dialogue: 0,{_ts(0)},{_ts(dur)},Attr,,0,0,0,,{{\\an8\\pos({width // 2},{SAFE_TOP - 70})}}{_escape(attribution)}")
    return header + "\n".join(events) + "\n"


def _wrap(text: str, width: int) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return r"\N".join(lines[:3])


def write_ass(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8-sig")
    return path
