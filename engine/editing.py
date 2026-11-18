"""Clip editing shared by the dashboard (PRD §14 D) and the CLI: boundaries, crop, layout, captions,
hooks, social copy. Applies the change to the database; the caller re-renders."""
from __future__ import annotations

import time
from typing import Any, Optional

from engine.db import Database, loads

LAYOUTS = ("split", "fullcam", "gameplay_only")


def edit_clip(db: Database, clip_id: int, *, start: Optional[float] = None, end: Optional[float] = None,
              hook_choice: Optional[int] = None, hook_text: Optional[str] = None, layout: Optional[str] = None,
              cam_box: Optional[list[int]] = None, focus_x: Optional[float] = None, gameplay_mode: Optional[str] = None,
              caption: Optional[dict] = None, title: Optional[str] = None, description: Optional[str] = None,
              recipe: Optional[str] = None) -> dict:
    """Returns {"fields": changed columns, "needs_render": bool}. Raises ValueError on bad input."""
    c = db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
    if not c:
        raise ValueError(f"no clip {clip_id}")
    s_new = start if start is not None else c["start"]
    e_new = end if end is not None else c["end"]
    if e_new - s_new < 5:
        raise ValueError("a clip must be at least 5 seconds long")
    if e_new - s_new > 120:
        raise ValueError("a clip can be at most 2 minutes long")
    fields: dict[str, Any] = {}
    render = False
    if start is not None or end is not None:
        fields["start"], fields["end"] = s_new, e_new
        render = True
    lay = loads(c["layout_json"], {}) or {}
    if layout is not None:
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        lay["manual"], lay["layout"] = True, layout
        render = True
    if cam_box is not None:
        if len(cam_box) != 4 or min(cam_box) < 0 or cam_box[2] < 16 or cam_box[3] < 16:
            raise ValueError("cam_box must be x,y,w,h in source pixels")
        lay["manual"], lay["cam_box"] = True, [int(v) for v in cam_box]
        lay.setdefault("layout", "split")
        if lay["layout"] == "gameplay_only":
            lay["layout"] = "split"
        render = True
    if focus_x is not None:
        lay["gameplay_focus_x"] = min(1.0, max(0.0, focus_x))
        render = True
    if gameplay_mode is not None:
        if gameplay_mode not in ("fill", "fit_blur"):
            raise ValueError("gameplay_mode must be fill or fit_blur")
        lay["gameplay_mode"] = gameplay_mode
        render = True
    if recipe is not None:
        lay["recipe"] = {"off": True} if recipe in ("", "off") else {"recipe_id": recipe}
        render = True
    if layout is not None or cam_box is not None or focus_x is not None or gameplay_mode is not None or recipe is not None:
        fields["layout_json"] = lay
    if caption:
        style = loads(c["caption_style_json"], {}) or {}
        ok = {"font", "font_size", "highlight_color", "primary_color", "max_words_per_line", "position", "animation",
              "uppercase", "show_hook"}
        bad = set(caption) - ok
        if bad:
            raise ValueError(f"unknown caption fields: {sorted(bad)}")
        style.update({k: v for k, v in caption.items() if v is not None})
        fields["caption_style_json"] = style
        render = True
    meta = loads(c["metadata_json"], {}) or {}
    if hook_text is not None or hook_choice is not None:
        hooks = list(meta.get("hooks") or loads(c["hooks_json"], []) or [])
        idx = hook_choice if hook_choice is not None else c["hook_choice"]
        if idx < 0 or idx > 5:
            raise ValueError("hook index out of range")
        if hook_choice is not None:
            fields["hook_choice"] = hook_choice
        if hook_text is not None:
            while len(hooks) <= idx:
                hooks.append("")
            hooks[idx] = hook_text.strip()
            meta["hooks"] = hooks
            fields["hooks_json"] = hooks
        render = True
    if title is not None:
        meta["title"] = title.strip()
    if description is not None:
        meta["description"] = description
    if title is not None or description is not None or hook_text is not None:
        fields["metadata_json"] = meta
    if fields:
        fields["updated_at"] = time.time()
        db.update("clips", clip_id, fields)
    return {"fields": sorted(fields), "needs_render": render}
