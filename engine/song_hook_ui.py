"""Song hook editor: see the song as a waveform, play it, and mark the START and END of the part the engine should use.

Click the waveform to jump, press Play to listen from there, "Set start / end here" to mark, "Preview" to hear only the
marked part. Saved marks are used for every clip from now on (the engine enters the song at the start, on the nearest
bar, and never plays past the end).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import customtkinter as ctk
import numpy as np

from engine.db import loads

try:
    import winsound
except ImportError:
    winsound = None

W, H = 760, 110


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def parse(s: str) -> float:
    s = s.strip()
    if ":" in s:
        m, _, sec = s.partition(":")
        return int(m) * 60 + float(sec)
    return float(s)


def envelope(path: Path, bins: int = W) -> tuple[np.ndarray, float]:
    """Loudness per pixel column + the song's length in seconds."""
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "4000",
                           "-f", "f32le", "-"], capture_output=True, timeout=120)
    x = np.frombuffer(proc.stdout, dtype="<f4")
    if x.size == 0:
        raise RuntimeError("could not read the song")
    n = x.size // bins * bins
    env = np.sqrt((x[:n].reshape(bins, -1) ** 2).mean(axis=1)) if n else np.zeros(bins)
    return env / (env.max() + 1e-9), x.size / 4000.0


def open_hook_editor(app, song_id: int) -> None:
    from engine import gui as G
    from engine.music import library

    row = app.db.one("SELECT * FROM songs WHERE id=?", (song_id,))
    if row is None:
        return
    a = library.effective_analysis(row)
    song_path = Path(row["file_path"])
    hook = a.get("hook") or {}
    state = {"start": float(hook.get("start", 0.0)), "end": float(hook.get("end", hook.get("start", 0.0) + 15)),
             "pos": float(hook.get("start", 0.0)), "playing": False, "t0": 0.0, "from": 0.0, "stop_at": None, "gen": 0,
             "dur": float(a.get("duration") or 1.0), "env": None}

    win = ctk.CTkToplevel(app, fg_color=G.BG)
    win.title(f"Hook — {row['title']}")
    win.geometry("820x640")
    win.minsize(700, 520)
    win.transient(app)
    win.after(150, lambda: (app._dark_caption(win), win.focus_force()))
    win.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(win, text=f"Hook: {row['title']}", font=G.font(22, "bold"), text_color=G.TEXT).grid(row=0, column=0, sticky="w", padx=28, pady=(24, 2))
    ctk.CTkLabel(win, text="Click the waveform to jump · play it · mark where the part you want starts and ends.", font=G.font(13),
                 text_color=G.MUTED).grid(row=1, column=0, sticky="w", padx=28)
    canvas = ctk.CTkCanvas(win, width=W, height=H, bg=G.CARD, highlightthickness=0)
    canvas.grid(row=2, column=0, padx=28, pady=14)
    info = ctk.StringVar()
    ctk.CTkLabel(win, textvariable=info, font=G.font(14, "bold"), text_color=G.IVORY).grid(row=3, column=0, sticky="w", padx=28)

    def x_of(t: float) -> float:
        return t / state["dur"] * W

    def draw_wave() -> None:
        canvas.delete("wave")
        env = state["env"]
        if env is None:
            canvas.create_text(W / 2, H / 2, text="loading waveform…", fill=G.MUTED, tags="wave")
        else:
            a, b = x_of(state["start"]), x_of(state["end"])
            for i, v in enumerate(env):
                h = max(1.0, v * (H - 14))
                canvas.create_line(i, H / 2 - h / 2, i, H / 2 + h / 2, fill=G.VIOLET if a <= i <= b else G.LINE, tags="wave")
        canvas.delete("marks")
        for key, col in (("start", G.GREEN), ("end", G.RED)):
            x = x_of(state[key])
            canvas.create_line(x, 0, x, H, fill=col, width=3, tags="marks")
        cursor()

    def cursor() -> None:
        canvas.delete("cursor")
        canvas.create_line(x_of(state["pos"]), 0, x_of(state["pos"]), H, fill=G.IVORY, width=2, tags="cursor")
        info.set(f"Playhead {mmss(state['pos'])}     Start {mmss(state['start'])}     End {mmss(state['end'])}     "
                 f"({state['end'] - state['start']:.1f}s)")

    def redraw() -> None:             # marks / waveform changed
        draw_wave()

    def load_env() -> None:
        try:
            env, dur = envelope(song_path)
            state["env"], state["dur"] = env, max(dur, 1.0)
        except Exception:
            state["env"] = np.zeros(W)
        app.after(0, redraw)
    threading.Thread(target=load_env, daemon=True).start()

    # ---------------------------------------------------------------- playback (ffmpeg cuts a snippet, winsound plays it)
    def stop() -> None:
        state["playing"] = False
        state["gen"] += 1
        if winsound is not None:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    def play(frm: float, until: float | None = None) -> None:
        stop()
        gen = state["gen"]
        length = (until - frm) if until else min(60.0, state["dur"] - frm)
        out = Path(tempfile.gettempdir()) / f"gta6_hook_{os.getpid()}_{gen}.wav"
        info.set("loading…")

        def work() -> None:
            try:
                subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-ss", f"{frm:.3f}", "-t", f"{length:.3f}",
                                "-i", str(song_path), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(out)], check=True, timeout=60)
            except Exception:
                return
            if gen != state["gen"]:
                out.unlink(missing_ok=True)
                return

            def go() -> None:
                if gen != state["gen"]:
                    return
                if winsound is not None:
                    winsound.PlaySound(str(out), winsound.SND_FILENAME | winsound.SND_ASYNC)
                state.update(playing=True, t0=time.perf_counter(), **{"from": frm}, stop_at=until or frm + length)
                tick(gen)
            app.after(0, go)
        threading.Thread(target=work, daemon=True).start()

    def tick(gen: int) -> None:
        if gen != state["gen"] or not state["playing"]:
            return
        state["pos"] = state["from"] + (time.perf_counter() - state["t0"])
        if state["pos"] >= (state["stop_at"] or state["dur"]):
            state["pos"] = state["stop_at"] or state["dur"]
            state["playing"] = False
            cursor()
            return
        cursor()
        win.after(60, lambda: tick(gen))

    def seek(t: float) -> None:
        state["pos"] = min(max(0.0, t), state["dur"])
        if state["playing"]:
            play(state["pos"])
        else:
            cursor()

    canvas.bind("<Button-1>", lambda e: seek(e.x / W * state["dur"]))

    def toggle() -> None:
        if state["playing"]:
            stop()
        else:
            play(state["pos"])

    # ---------------------------------------------------------------- controls
    ctrl = ctk.CTkFrame(win, fg_color="transparent")
    ctrl.grid(row=4, column=0, sticky="w", padx=28, pady=(12, 6))
    G.button(ctrl, "▶ / ❚❚  Play", toggle, primary=True, width=120, height=40).pack(side="left")
    for label, d in (("−5s", -5), ("−1s", -1), ("+1s", 1), ("+5s", 5)):
        G.button(ctrl, label, lambda d=d: seek(state["pos"] + d), width=54, height=40, fg_color=G.CARD2).pack(side="left", padx=3)
    G.button(ctrl, "⟲ From start mark", lambda: (seek(state["start"]), play(state["start"])), width=150, height=40,
             fg_color=G.CARD2).pack(side="left", padx=(12, 0))

    marks = ctk.CTkFrame(win, fg_color="transparent")
    marks.grid(row=5, column=0, sticky="w", padx=28, pady=6)
    s_var, e_var = ctk.StringVar(), ctk.StringVar()

    def sync_entries() -> None:
        s_var.set(mmss(state["start"]))
        e_var.set(mmss(state["end"]))
        redraw()

    def mark(which: str) -> None:
        state[which] = round(state["pos"], 2)
        if state["end"] <= state["start"] + 1.0:
            if which == "start":
                state["end"] = min(state["dur"], state["start"] + 15)
            else:
                state["start"] = max(0.0, state["end"] - 15)
        sync_entries()

    G.button(marks, "◀ Set START here", lambda: mark("start"), width=150, height=38, fg_color="#1D3A22", hover_color="#2E7D45").pack(side="left")
    ctk.CTkEntry(marks, textvariable=s_var, width=80, height=38, fg_color=G.CARD2, border_color=G.LINE).pack(side="left", padx=6)
    G.button(marks, "Set END here ▶", lambda: mark("end"), width=140, height=38, fg_color="#3A1618", hover_color=G.RED).pack(side="left", padx=(14, 0))
    ctk.CTkEntry(marks, textvariable=e_var, width=80, height=38, fg_color=G.CARD2, border_color=G.LINE).pack(side="left", padx=6)

    def typed(_=None) -> None:
        try:
            state["start"], state["end"] = max(0.0, parse(s_var.get())), min(state["dur"], parse(e_var.get()))
        except ValueError:
            pass
        redraw()
    for ent in marks.winfo_children():
        if isinstance(ent, ctk.CTkEntry):
            ent.bind("<Return>", typed)
            ent.bind("<FocusOut>", typed)
    sync_entries()

    msg = ctk.StringVar()
    pv = ctk.CTkFrame(win, fg_color="transparent")
    pv.grid(row=6, column=0, sticky="w", padx=28, pady=6)
    G.button(pv, "▶ Preview the marked part", lambda: (typed(), play(state["start"], state["end"])), width=210, height=38,
             fg_color=G.CARD2).pack(side="left")
    G.button(pv, "Use the engine's pick", lambda: (state.update(start=float(hook.get("start", 0)), end=float(hook.get("end", 15))), sync_entries()),
             width=170, height=38, fg_color=G.CARD2).pack(side="left", padx=8)
    ctk.CTkLabel(win, textvariable=msg, font=G.font(13), text_color=G.AMBER, wraplength=700, justify="left", anchor="w").grid(
        row=7, column=0, sticky="w", padx=28, pady=(8, 0))

    def save() -> None:
        typed()
        try:
            h = library.set_hook(app.db, song_id, state["start"], state["end"])
        except ValueError as e:
            msg.set(str(e))
            return
        stop()
        app._log(f"Hook saved for “{row['title']}”: {mmss(h['start'])} – {mmss(h['end'])}")
        win.destroy()

    def reset() -> None:
        library.clear_hook(app.db, song_id)
        msg.set("Back to the engine's own pick. Re-open to see it.")

    bar = ctk.CTkFrame(win, fg_color="transparent")
    bar.grid(row=8, column=0, sticky="ew", padx=28, pady=(14, 20))
    G.button(bar, "Save hook", save, primary=True, width=130, height=42).pack(side="left")
    G.button(bar, "Reset to automatic", reset, width=170, height=42, fg_color=G.CARD2).pack(side="left", padx=8)
    G.button(bar, "Close", lambda: (stop(), win.destroy()), width=90, height=42, fg_color=G.CARD2).pack(side="right")
    win.protocol("WM_DELETE_WINDOW", lambda: (stop(), win.destroy()))
