"""Desktop app for the clip engine (customtkinter): sidebar navigation, dashboard, dark neon theme.

Pages: Dashboard (stats, live queue, top clips), New job (paste YouTube/Twitch/Kick links), Songs (analyzed
automatically from the Songs folder), Clips (play / open / export), Activity (progress + log).
The engine runs inside this process. The web dashboard still runs in the background (no sidebar button).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

SONG_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
POLL_MS = 1500
DONE = ("done", "failed", "error", "completed", "discovered", "processed", "rendered", "ready")

# ISYNTH palette: matte black / burnt orange / copper / ivory. (Names VIOLET/PINK/CYAN kept for the
# primary accent / hover accent / tertiary accent roles so the rest of the file is unchanged.)
BG, SIDE, CARD, CARD2, LINE = "#0A0806", "#100C09", "#141110", "#1B1714", "#3A2A1E"
TEXT, MUTED = "#ECE6DC", "#9A9289"
VIOLET, PINK, GREEN, AMBER, RED, CYAN = "#E8651A", "#FF8A2A", "#6FBF73", "#F2C14E", "#E5484D", "#C98A55"
INK, IVORY = "#0A0806", "#F0E6D4"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


def open_path(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "startfile"):
        os.startfile(str(p))
    else:
        subprocess.Popen(["xdg-open", str(p)])


def font(size: int = 13, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family="Segoe UI Variable Display" if os.name == "nt" else None, size=size, weight=weight)


def card(parent, **kw) -> ctk.CTkFrame:
    return ctk.CTkFrame(parent, fg_color=CARD, corner_radius=18, border_width=1, border_color=LINE, **kw)


def pill(parent, text: str, color: str) -> ctk.CTkLabel:
    return ctk.CTkLabel(parent, text=f"  {text}  ", fg_color=color, text_color=INK, corner_radius=20,
                        font=font(11, "bold"), height=22)


def badge(parent, text: str, color: str, size: int = 52, fg: str = INK, fsize: int = 17) -> ctk.CTkFrame:
    """A true circle (a plain CTkLabel stretches into an oval)."""
    f = ctk.CTkFrame(parent, width=size, height=size, corner_radius=size // 2, fg_color=color)
    f.grid_propagate(False)
    f.pack_propagate(False)
    ctk.CTkLabel(f, text=text, font=font(fsize, "bold"), text_color=fg, fg_color="transparent").place(relx=0.5, rely=0.5, anchor="center")
    return f


def fit_text(app, row, label, text: str, fnt, reserved: int) -> None:
    """Ellipsize `text` to the room the row really has (re-fits when the window is resized)."""
    last = {"w": -1}

    def on_cfg(e) -> None:
        if abs(e.width - last["w"]) < 2:
            return
        last["w"] = e.width
        sc = app._get_window_scaling() or 1.0
        avail = e.width - reserved * sc
        t = text
        try:
            while len(t) > 1 and fnt.measure(t) * sc > avail:
                t = t[:-1]
            label.configure(text=text if t == text else t.rstrip() + "…")
        except Exception:
            pass
    row.bind("<Configure>", on_cfg, add="+")


def button(parent, text: str, command, primary: bool = False, **kw) -> ctk.CTkButton:
    opts = dict(height=42, corner_radius=14, font=font(13, "bold" if primary else "normal"),
                fg_color=VIOLET if primary else CARD2, hover_color=PINK if primary else LINE,
                text_color="white" if primary else TEXT)
    opts.update(kw)
    return ctk.CTkButton(parent, text=text, command=command, **opts)


class App(ctk.CTk):
    NAV = [("Dashboard", "▦"), ("New job", "＋"), ("Songs", "♪"), ("Clips", "▶"), ("Review", "⇄"), ("Post", "✈"), ("Activity", "≋")]

    def __init__(self) -> None:
        super().__init__(fg_color=BG)
        from engine import profiles, runtime
        from engine.config import ROOT, load_settings

        runtime.prepare()
        try:
            runtime.first_run(log=lambda m: None)
        except Exception:
            pass
        from engine.dashboard import server       # starts the background job worker

        self.server, self.profiles, self.ROOT = server, profiles, ROOT
        from engine.watch import Watcher
        self.watcher = Watcher(ROOT, load_settings, server._queue_source, server.worker.wake, lambda m: self.after(0, lambda: self._log(m)))
        self.watcher.start()
        self.db = server._db
        self.settings = load_settings()
        self.songs_dir = ROOT / "Songs"           # drop songs here; they are analyzed automatically
        self.songs_dir.mkdir(parents=True, exist_ok=True)
        from engine import naming
        self.library = naming.LIBRARY            # Clips/<Stream> - <Song> - Clip NN/
        self.seen_songs: dict[str, str] = {}
        self.songs_cache: list[dict] = []
        self.more_busy: set[int] = set()
        self.engine = None
        self.sigs: dict[str, object] = {}
        self.log_after = 0
        self.web_port: int | None = None
        self.pages: dict[str, ctk.CTkFrame] = {}
        self.nav_btns: dict[str, ctk.CTkButton] = {}
        self.current = "Dashboard"

        self.title("GTA 6 Clip Engine")
        w = int(min(1360, self.winfo_screenwidth() * 0.9))
        h = int(min(860, self.winfo_screenheight() * 0.9))
        self.geometry(f"{w}x{h}+{max(int((self.winfo_screenwidth() - w) / 2), 0)}+10")
        self.minsize(min(1100, w), min(700, h))
        self._build()
        self.show("Dashboard")
        self.after(150, self._style_titlebar)
        self.after(300, self._poll)
        self.after(1500, self.check_ai)
        self.protocol("WM_DELETE_WINDOW", self._quit)

    # ------------------------------------------------------------------ shell
    def _build(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        side = ctk.CTkFrame(self, fg_color=SIDE, corner_radius=0, width=256)
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_propagate(False)
        side.grid_rowconfigure(10, weight=1)
        brand = ctk.CTkFrame(side, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="w", padx=28, pady=(30, 28))
        ctk.CTkLabel(brand, text="GTA 6", font=font(24, "bold"), text_color=PINK).pack(anchor="w")
        ctk.CTkLabel(brand, text="CLIP ENGINE", font=font(12, "bold"), text_color=MUTED).pack(anchor="w")
        for i, (name, icon) in enumerate(self.NAV, start=1):
            b = ctk.CTkButton(side, text=f"   {icon}    {name}", anchor="w", height=48, corner_radius=14, font=font(14, "bold"),
                              fg_color="transparent", hover_color=CARD2, text_color=MUTED, command=lambda n=name: self.show(n))
            b.grid(row=i, column=0, sticky="ew", padx=16, pady=3)
            self.nav_btns[name] = b
        foot = ctk.CTkFrame(side, fg_color="transparent")
        foot.grid(row=11, column=0, sticky="ew", padx=16, pady=22)
        self.status = ctk.CTkLabel(foot, text="●  Idle", font=font(13, "bold"), text_color=GREEN, fg_color=CARD,
                                   corner_radius=14, height=44)
        self.status.pack(fill="x", pady=(0, 12))
        self.ai_light = ctk.CTkLabel(foot, text="●  AI: checking…", font=font(12, "bold"), text_color=MUTED, anchor="w")
        self.ai_light.pack(fill="x", padx=6, pady=(0, 8))
        button(foot, "👁  Watch folder", self._open_watch, height=40, corner_radius=14, fg_color=CARD2,
               hover_color=LINE).pack(fill="x", pady=(0, 8))
        button(foot, "⚙  Settings", self._open_settings, height=44, corner_radius=14, fg_color=CARD2,
               hover_color=LINE).pack(fill="x")

        host = ctk.CTkFrame(self, fg_color="transparent")
        host.grid(row=0, column=1, sticky="nsew", padx=36, pady=(30, 32))
        host.grid_columnconfigure(0, weight=1)
        host.grid_rowconfigure(0, weight=1)
        for name, make in (("Dashboard", self._page_dashboard), ("New job", self._page_job), ("Songs", self._page_songs),
                           ("Clips", self._page_clips), ("Review", self._page_review), ("Post", self._page_post), ("Activity", self._page_activity)):
            f = ctk.CTkFrame(host, fg_color="transparent")
            f.grid(row=0, column=0, sticky="nsew")
            self.pages[name] = f
            make(f)
        if not self._ffmpeg():
            self._log("FFmpeg was not found (expected in the 'bin' folder next to this program, or installed on this PC). "
                      "Install it with:  winget install Gyan.FFmpeg  then restart the app. Clips cannot be rendered without it.")

    def _style_titlebar(self) -> None:
        """Dark caption in the app's own colours (Windows 11: caption/border/text colour; Windows 10: dark mode)."""
        if os.name != "nt":
            return
        try:
            import ctypes

            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())

            def bgr(hex_: str) -> ctypes.c_int:
                r, g, b = int(hex_[1:3], 16), int(hex_[3:5], 16), int(hex_[5:7], 16)
                return ctypes.c_int(r | g << 8 | b << 16)

            def attr(n: int, v) -> None:
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, n, ctypes.byref(v), ctypes.sizeof(v))

            attr(20, ctypes.c_int(1))        # immersive dark mode
            attr(35, bgr(SIDE))              # caption colour
            attr(36, bgr(TEXT))              # caption text
            attr(34, bgr(SIDE))              # border
        except Exception:
            pass

    def _page_post(self, p) -> None:
        from engine.post_ui import PostPage
        self.post_page = PostPage(self, p)

    def _page_review(self, p) -> None:
        from engine.review_ui import ReviewPage
        self.review_page = ReviewPage(self, p)

    def show(self, name: str) -> None:
        rp = getattr(self, "review_page", None)
        if rp is not None and self.current == "Review" and name != "Review":
            rp.leave()
        self.current = name
        if name == "Review" and rp is not None:
            self.after(50, rp.refresh)
        if name == "Post" and getattr(self, "post_page", None) is not None:
            self.post_page.refresh()
        self.pages[name].tkraise()
        for n, b in self.nav_btns.items():
            on = n == name
            b.configure(fg_color=VIOLET if on else "transparent", text_color=IVORY if on else MUTED)

    def _heading(self, page, title: str, sub: str, row: int = 0) -> ctk.CTkFrame:
        h = ctk.CTkFrame(page, fg_color="transparent")
        h.grid(row=row, column=0, sticky="ew", pady=(0, 26))
        h.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(h, text=title, font=font(30, "bold"), text_color=TEXT, anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(h, text=sub, font=font(14), text_color=MUTED, anchor="w").grid(row=1, column=0, sticky="w", pady=(4, 0))
        return h

    def _card_title(self, parent, text: str, sub: str = "") -> ctk.CTkFrame:
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=0, column=0, sticky="ew", padx=28, pady=(24, 12))
        ctk.CTkLabel(box, text=text, font=font(17, "bold"), text_color=TEXT).pack(side="left")
        if sub:
            ctk.CTkLabel(box, text="  " + sub, font=font(12), text_color=MUTED).pack(side="left")
        return box

    # ------------------------------------------------------------------ pages
    def _page_dashboard(self, p) -> None:
        p.grid_columnconfigure(0, weight=1)
        p.grid_rowconfigure(3, weight=1)
        h = self._heading(p, "Dashboard", "Everything the engine is doing, at a glance.")
        button(h, "＋  New job", lambda: self.show("New job"), primary=True, width=160, height=48, corner_radius=24).grid(row=0, column=1, rowspan=2)

        tiles = ctk.CTkFrame(p, fg_color="transparent")
        tiles.grid(row=1, column=0, sticky="ew", pady=(0, 22))
        self.stat = {}
        for i, (key, label, color) in enumerate((("clips", "Clips ready", VIOLET), ("top", "Best score", GREEN),
                                                 ("videos", "Videos", CYAN), ("songs", "Songs analyzed", PINK))):
            tiles.grid_columnconfigure(i, weight=1, uniform="t")
            t = card(tiles)
            t.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 9, 0 if i == 3 else 9))
            ctk.CTkFrame(t, width=36, height=5, corner_radius=3, fg_color=color).pack(anchor="w", padx=26, pady=(24, 0))
            v = ctk.CTkLabel(t, text="0", font=font(38, "bold"), text_color=TEXT, anchor="w")
            v.pack(anchor="w", padx=26, pady=(8, 0))
            ctk.CTkLabel(t, text=label, font=font(13), text_color=MUTED, anchor="w").pack(anchor="w", padx=26, pady=(0, 24))
            self.stat[key] = v

        prog = card(p)
        prog.grid(row=2, column=0, sticky="ew", pady=(0, 22))
        prog.grid_columnconfigure(0, weight=1)
        self.prog_title = ctk.CTkLabel(prog, text="Idle", font=font(16, "bold"), text_color=TEXT, anchor="w")
        self.prog_title.grid(row=0, column=0, sticky="ew", padx=26, pady=(18, 4))
        self.prog_bar = ctk.CTkProgressBar(prog, height=10, corner_radius=5, fg_color=CARD2, progress_color=VIOLET)
        self.prog_bar.set(0)
        self.prog_bar.grid(row=1, column=0, sticky="ew", padx=26)
        self.prog_sub = ctk.CTkLabel(prog, text="Nothing running. Start a job from “New job”.", font=font(12), text_color=MUTED, anchor="w")
        self.prog_sub.grid(row=2, column=0, sticky="ew", padx=26, pady=(6, 16))

        lower = ctk.CTkFrame(p, fg_color="transparent")
        lower.grid(row=3, column=0, sticky="nsew")
        lower.grid_columnconfigure(0, weight=1, uniform="l")
        lower.grid_columnconfigure(1, weight=1, uniform="l")
        lower.grid_rowconfigure(0, weight=1)
        left, right = card(lower), card(lower)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 11))
        right.grid(row=0, column=1, sticky="nsew", padx=(11, 0))
        for c in (left, right):
            c.grid_columnconfigure(0, weight=1)
            c.grid_rowconfigure(1, weight=1)
        self._card_title(left, "Now processing", "live")
        self.dash_queue = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.dash_queue.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self._card_title(right, "Top clips", "best first")
        self.dash_clips = ctk.CTkScrollableFrame(right, fg_color="transparent")
        self.dash_clips.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

    def _page_job(self, p) -> None:
        p.grid_columnconfigure(0, weight=1)
        p.grid_rowconfigure(1, weight=1)
        self._heading(p, "New job", "Add links, pick the song for each, then analyze. Every clip is named after its video, song and number.")
        c = card(p)
        c.grid(row=1, column=0, sticky="nsew")
        c.grid_columnconfigure(0, weight=1)
        c.grid_rowconfigure(4, weight=1)
        head = self._card_title(c, "Video links", "YouTube · Twitch · Kick · or video files — one per line")
        button(head, "＋  Video files", self._add_video_files, width=140, height=36).pack(side="right")
        self.links = ctk.CTkTextbox(c, fg_color=CARD2, border_width=1, border_color=LINE, corner_radius=14,
                                    font=font(14), text_color=TEXT, wrap="none", height=80)
        self.links.grid(row=1, column=0, sticky="ew", padx=28, pady=(4, 12))
        opts = ctk.CTkFrame(c, fg_color="transparent")
        opts.grid(row=2, column=0, sticky="ew", padx=28, pady=(0, 14))
        self.song_choice: tuple[str | None, str] = (None, "Automatic")
        self.song_btn = button(opts, "♪  Song: Automatic", self._pick_song, width=210)
        self.song_btn.pack(side="left")
        self.types = [("auto", "Auto-detect")] + [(k, p_.label) for k, p_ in self.profiles.PROFILES.items()]
        self.type_var = ctk.StringVar(value="Auto-detect")
        ctk.CTkOptionMenu(opts, variable=self.type_var, values=[l for _, l in self.types], width=200, height=42, corner_radius=14,
                          fg_color=CARD2, button_color=LINE, button_hover_color=VIOLET, dropdown_fg_color=CARD2).pack(side="left", padx=12)
        self.max_clips = ctk.CTkEntry(opts, placeholder_text="Max clips", width=110, height=42, corner_radius=14,
                                      fg_color=CARD2, border_color=LINE)
        self.max_clips.pack(side="left")
        button(opts, "🎨  Style", self._open_style, width=110).pack(side="left", padx=12)
        button(opts, "＋  Add to queue", self._add_to_queue, primary=True, width=160).pack(side="right", padx=(24, 0))
        # ---- AI suggestions: tell the AI brain how to choose and cut (applies to links you add next)
        gbox = ctk.CTkFrame(c, fg_color=CARD2, corner_radius=14)
        gbox.grid(row=3, column=0, sticky="ew", padx=28, pady=(0, 12))
        gbox.grid_columnconfigure(0, weight=1)
        gh = ctk.CTkFrame(gbox, fg_color="transparent")
        gh.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))
        gh.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(gh, text="✦  AI suggestions: how should it choose and cut?", font=font(13, "bold"), text_color=PINK,
                     anchor="w").grid(row=0, column=0, sticky="w")
        button(gh, "💡 Suggest ideas", self._suggest_ideas, width=140, height=30, fg_color=CARD).grid(row=0, column=1, padx=(6, 0))
        button(gh, "Clear", lambda: self.guidance.delete("1.0", "end"), width=64, height=30, fg_color=CARD).grid(row=0, column=2, padx=(6, 0))
        self.guidance = ctk.CTkTextbox(gbox, fg_color=CARD, border_width=1, border_color=LINE, corner_radius=10, font=font(13),
                                       text_color=TEXT, height=58, wrap="word")
        self.guidance.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 6))
        ctk.CTkLabel(gbox, text="Optional. Plain words, e.g. “pick the funniest reactions, skip long setups, no clips about chat”. "
                     "It guides the AI (or, with no AI key, a keyword boost). Click an idea to add it.", font=font(11), text_color=MUTED,
                     anchor="w", justify="left", wraplength=760).grid(row=2, column=0, sticky="w", padx=16)
        self.idea_row = ctk.CTkFrame(gbox, fg_color="transparent")
        self.idea_row.grid(row=3, column=0, sticky="ew", padx=12, pady=(4, 10))
        self._show_ideas()
        self.queue_list = ctk.CTkScrollableFrame(c, fg_color=CARD2, corner_radius=14)
        self.queue_list.grid(row=4, column=0, sticky="nsew", padx=28, pady=(0, 14))
        self.go = button(c, "▶   Analyze & make clips", self._start, primary=True, height=56, corner_radius=28)
        self.go.configure(font=font(16, "bold"))
        self.go.grid(row=5, column=0, sticky="ew", padx=28, pady=(0, 28))
        self.queue: list[dict] = []
        self._render_queue()

    def _show_ideas(self, ideas: list[str] | None = None) -> None:
        from engine.ai.suggest import PRESETS

        for w in self.idea_row.winfo_children():
            w.destroy()
        for k, text in enumerate((ideas or PRESETS)[:8]):
            b = button(self.idea_row, text, lambda t=text: self._add_idea(t), height=28, fg_color=CARD, hover_color=LINE,
                       font=font(12), text_color=TEXT)
            b.grid(row=k // 4, column=k % 4, padx=3, pady=3, sticky="w")

    def _add_idea(self, text: str) -> None:
        cur = self.guidance.get("1.0", "end").strip()
        if text.lower() in cur.lower():
            return
        self.guidance.insert("end", ("" if not cur else (" " if cur.endswith((".", ";")) else ". ")) + text + ".")
        self.guidance.see("end")

    def _suggest_ideas(self) -> None:
        """Ask the AI for fresh instructions (needs an API key); otherwise show the built-in ideas."""
        from engine.ai.llm import get_provider
        from engine.ai.suggest import ideas
        from engine.config import load_settings

        label = self.type_var.get()
        have = self.guidance.get("1.0", "end").strip()
        prov, msg = get_provider(load_settings().ai)
        self._show_ideas(["Thinking…"])

        def work() -> None:
            out = ideas(prov, label if label != "Auto-detect" else "streamer video", have)
            self.after(0, lambda: (self._show_ideas(out), self._log("AI ideas ready." if prov else
                                                                  "Showing built-in ideas. Add an API key in Settings for AI-written ones.")))
        threading.Thread(target=work, daemon=True).start()

    def _add_to_queue(self) -> None:
        text = self.links.get("1.0", "end").strip()
        lines = [x.strip().strip('"') for x in text.replace("\r", "").split("\n") if x.strip()]
        if not lines:
            messagebox.showinfo("Nothing to add", "Paste a link (or add a video file) first.")
            return
        for ln in lines:
            self.queue.append({"link": ln, "song": self.song_choice[0], "label": self.song_choice[1],
                               "focus": self.guidance.get("1.0", "end").strip() or None})
        self.links.delete("1.0", "end")
        self._render_queue()

    def _render_queue(self) -> None:
        for w in self.queue_list.winfo_children():
            w.destroy()
        if not self.queue:
            self._empty(self.queue_list, "The queue is empty.\nPaste links above, choose a song, and press Add to queue.")
            return
        for i, it in enumerate(self.queue):
            r = ctk.CTkFrame(self.queue_list, fg_color=CARD, corner_radius=14)
            r.pack(fill="x", pady=5, padx=2)
            r.grid_columnconfigure(0, weight=1)
            f = font(13, "bold")
            lab = ctk.CTkLabel(r, text=it["link"], font=f, text_color=TEXT, anchor="w", width=10)
            lab.grid(row=0, column=0, sticky="ew", padx=(20, 10), pady=16)
            fit_text(self, r, lab, it["link"], f, 20 + 10 + 190 + 16 + 56 + 16)
            pill(r, "♪ " + it["label"][:16] + ("  ·  ✦ guided" if it.get("focus") else ""), VIOLET if it["song"] and it["song"] != "__none__" else CARD2).grid(row=0, column=1, padx=8)
            if not (it["song"] and it["song"] != "__none__"):
                r.winfo_children()[-1].configure(text_color=MUTED)
            button(r, "✕", lambda k=i: self._remove_queued(k), width=40, height=36, corner_radius=18, fg_color=LINE,
                   hover_color=RED).grid(row=0, column=2, padx=(4, 16))

    def _remove_queued(self, i: int) -> None:
        if 0 <= i < len(self.queue):
            del self.queue[i]
            self._render_queue()

    def _dark_caption(self, win) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            hwnd = ctypes.windll.user32.GetParent(win.winfo_id())

            def bgr(h: str) -> ctypes.c_int:
                return ctypes.c_int(int(h[1:3], 16) | int(h[3:5], 16) << 8 | int(h[5:7], 16) << 16)
            for n, v in ((20, ctypes.c_int(1)), (35, bgr(SIDE)), (36, bgr(TEXT)), (34, bgr(SIDE))):
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, n, ctypes.byref(v), ctypes.sizeof(v))
        except Exception:
            pass

    def _open_style(self) -> None:
        """Look-and-feel dialog: caption font/size/colours/animation, title (hook) card, music style, video extras.
        Saved to settings, so it applies to every clip made from now on."""
        from engine.config import CaptionStyle, load_settings, save_settings

        st = load_settings()
        cs, ms, vs = st.captions, st.music, st.video
        colors = {"White": "#FFFFFF", "Yellow": "#FFD400", "Green": "#3DFF6E", "Pink": "#FF4D8D", "Cyan": "#38BDF8",
                  "Orange": "#FF8A1F", "Red": "#FF3B3B", "Purple": "#9B6BFF", "Black": "#000000"}
        styles = {"auto": "Auto (talking: under, action: over)", "under": "Always under the voice", "over": "Always over the scene",
                  "moment": "Big moments (AI aims the song at crashes, explosions, screams)"}

        def cname(hex_: str) -> str:
            return next((k for k, v in colors.items() if v.lower() == hex_.lower()), "White")

        win = ctk.CTkToplevel(self, fg_color=BG)
        win.title("Style")
        win.geometry("620x760")
        win.minsize(520, 420)
        win.transient(self)
        win.after(150, lambda: (self._dark_caption(win), win.grab_set(), win.focus_force()))
        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(win, text="Style", font=font(22, "bold"), text_color=TEXT).grid(row=0, column=0, sticky="w", padx=28, pady=(26, 2))
        ctk.CTkLabel(win, text="How your clips look. Applies to every clip made from now on.", font=font(13),
                     text_color=MUTED).grid(row=1, column=0, sticky="w", padx=28)
        scroll = ctk.CTkScrollableFrame(win, fg_color="transparent")
        scroll.grid(row=2, column=0, sticky="nsew", padx=14, pady=10)
        scroll.grid_columnconfigure(0, weight=1)
        rows = [0]
        vars_: dict = {}

        def section(title: str):
            f = card(scroll)
            f.grid(row=rows[0], column=0, sticky="ew", padx=14, pady=(0, 12))
            f.grid_columnconfigure(1, weight=1)
            rows[0] += 1
            ctk.CTkLabel(f, text=title, font=font(12, "bold"), text_color=PINK).grid(
                row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(14, 4))
            f.n = 1
            return f

        def add(f, label: str, key: str, kind: str, value, values=None) -> None:
            n = f.n
            f.n += 1
            ctk.CTkLabel(f, text=label, font=font(13), text_color=TEXT, anchor="w").grid(row=n, column=0, sticky="w", padx=(20, 10), pady=6)
            if kind == "menu":
                v = ctk.StringVar(value=str(value))
                w = ctk.CTkOptionMenu(f, variable=v, values=values, height=36, corner_radius=12, fg_color=CARD2, button_color=LINE,
                                      button_hover_color=VIOLET, dropdown_fg_color=CARD2)
            elif kind == "combo":
                v = ctk.StringVar(value=str(value))
                w = ctk.CTkComboBox(f, variable=v, values=values, height=36, corner_radius=12, fg_color=CARD2, border_color=LINE,
                                    button_color=LINE, dropdown_fg_color=CARD2)
            elif kind == "switch":
                v = ctk.BooleanVar(value=bool(value))
                w = ctk.CTkSwitch(f, text="", variable=v, progress_color=VIOLET)
            else:
                v = ctk.StringVar(value=str(value))
                w = ctk.CTkEntry(f, textvariable=v, height=36, corner_radius=12, fg_color=CARD2, border_color=LINE)
            w.grid(row=n, column=1, sticky="e" if kind == "switch" else "ew", padx=(0, 20), pady=6)
            vars_[key] = v

        import tkinter.font as tkfont
        popular = ["Arial Black", "Impact", "Arial", "Segoe UI Black", "Bahnschrift", "Verdana", "Trebuchet MS", "Tahoma",
                   "Georgia", "Comic Sans MS", "Courier New"]
        installed = sorted({n for n in tkfont.families(self) if not n.startswith("@")}, key=str.lower)
        fonts = [n for n in popular if n in installed] + [n for n in installed if n not in popular]
        f = section("CAPTIONS")
        add(f, "Font (every installed font)", "font", "combo", cs.font, fonts)
        add(f, "Size", "font_size", "entry", cs.font_size)
        add(f, "Position", "position", "menu", cs.position, ["lower", "seam", "upper"])
        add(f, "Animation", "animation", "menu", cs.animation, ["pop", "highlight", "none"])
        add(f, "Words per line", "max_words_per_line", "menu", str(cs.max_words_per_line), ["1", "2", "3", "4", "5"])
        add(f, "ALL CAPS", "uppercase", "switch", cs.uppercase)
        add(f, "Text colour", "primary_color", "menu", cname(cs.primary_color), list(colors))
        add(f, "Current-word colour", "highlight_color", "menu", cname(cs.highlight_color), list(colors))
        add(f, "Hype-word colour", "emphasis_color", "menu", cname(cs.emphasis_color), list(colors))
        add(f, "Outline colour", "outline_color", "menu", cname(cs.outline_color), list(colors))
        add(f, "Outline thickness", "outline", "menu", str(cs.outline), [str(i) for i in range(0, 13)])
        f = section("TITLE (HOOK TEXT AT THE START)")
        add(f, "Show title card", "show_hook", "switch", cs.show_hook)
        add(f, "Seconds on screen", "hook_seconds", "entry", cs.hook_seconds)
        tstyles = {"box": "White box", "yellow": "Yellow box", "dark": "Dark banner", "red": "Red alert", "outline": "Yellow outline text"}
        add(f, "Title look", "title_style", "menu", tstyles[cs.title_style], list(tstyles.values()))
        f = section("MUSIC")
        add(f, "Music style", "music_style", "menu", styles[ms.style], list(styles.values()))
        add(f, "Let the AI judge where the song lands", "ai_placement", "switch", ms.ai_placement)
        add(f, "Max repeats of one song per 10 clips (0 = no limit)", "rotation_max", "entry", ms.rotation_max)
        add(f, "Show song name tag", "sound_tag", "switch", ms.sound_tag)
        add(f, "Song card at the end", "end_card", "switch", ms.end_card)
        duckings = {"smooth": "Smooth: steady level while they talk on and off", "tight": "Tight: dips under every phrase"}
        add(f, "When people talk over the song", "ducking", "menu", duckings[ms.ducking], list(duckings.values()))
        dips = {None: "Volume preset's default", 6.0: "A little (-6 dB, song stays big)", 9.0: "Some (-9 dB)",
                12.0: "Medium (-12 dB)", 15.0: "A lot (-15 dB, voice very clear)", 20.0: "Almost off (-20 dB)"}
        add(f, "How far the song dips under talking", "talk_duck_db", "menu",
            dips.get(ms.talk_duck_db, dips[None]), list(dips.values()))
        add(f, "Cuts land on the song's beat", "beat_sync_cuts", "switch", ms.beat_sync_cuts)
        f = section("MUSIC PROMO (SONG FIRST)")
        aon = {"auto": "Auto (Music promo videos only)", "on": "On for every clip with a song", "off": "Off"}
        add(f, "Silent beat right before the chorus hits", "pre_drop_dropout", "menu", aon[ms.pre_drop_dropout], list(aon.values()))
        add(f, "Silence length (seconds, 0 = one beat)", "pre_drop_seconds", "entry", ms.pre_drop_seconds)
        add(f, "Song lyrics on screen", "lyrics", "menu", aon[cs.lyrics], list(aon.values()))
        add(f, "Build-up before the chorus (bars)", "promo_build_bars", "menu", str(ms.promo_build_bars),
            [str(i) for i in range(1, 9)])
        ctk.CTkLabel(f, text="Each clip can override the silent beat and lyrics in Review (Song extras).", font=font(11),
                     text_color=MUTED, anchor="w", wraplength=480, justify="left").grid(
            row=f.n, column=0, columnspan=2, sticky="w", padx=20, pady=(0, 12))
        f.n += 1
        f = section("CREDITS (video name and creator)")
        add(f, "Show the creator's name on the clip", "show_attribution", "switch", cs.show_attribution)
        add(f, "Put the creator + link in the post caption", "credit_in_caption", "switch", cs.credit_in_caption)
        fn = {"video": "Use the video's title", "number": "Hide the title (Video 3 - song - Clip 01)"}
        add(f, "Clip folder names", "folder_names", "menu", fn[load_settings().folder_names], list(fn.values()))
        ctk.CTkLabel(f, text="Some creators and clipping programs require credit. Turn these off only if you are allowed to.",
                     font=font(11), text_color=MUTED, anchor="w", wraplength=480, justify="left").grid(
            row=f.n, column=0, columnspan=2, sticky="w", padx=20, pady=(0, 12))
        f.n += 1
        f = section("CLIP LENGTH")
        add(f, "Minimum clip length (seconds)", "min_length", "entry", int(load_settings().clips.min_length))
        add(f, "Longest clip (seconds)", "max_duration", "entry", int(load_settings().clips.max_duration))
        f = section("VIDEO")
        layouts = {"auto": "Auto-detect", "double": "Double camera (cam on top, scene below)", "single": "Single (one full view)"}
        add(f, "Layout", "layout_style", "menu", layouts[vs.layout_style], list(layouts.values()))
        add(f, "Face cam share of the screen (%)", "cam_ratio", "entry", round(vs.cam_ratio * 100))
        add(f, "Punch-in zoom on the payoff", "punch_in", "switch", vs.punch_in)
        add(f, "Divider line between cam and game", "divider", "switch", vs.divider)
        add(f, "Divider colour", "divider_color", "menu", cname(vs.divider_color), list(colors))
        qual = {"standard": "Standard", "high": "High (sharper, larger files)", "max": "Maximum (sharpest, largest files)"}
        cur_q = "max" if vs.crf <= 16 else "high" if vs.crf <= 18 else "standard"
        add(f, "Video quality", "quality", "menu", qual[cur_q], list(qual.values()))
        fvm = {"blur": "Whole picture over a blurred backdrop", "fill": "Zoom to fill the screen (sharp, crops the sides)"}
        add(f, "Clips with no face cam", "full_view_mode", "menu", fvm[vs.full_view_mode], list(fvm.values()))
        add(f, "Face tracking (camera panel follows his face)", "face_tracking", "switch", vs.face_tracking)
        add(f, "Lead with the best part first", "lead_with_best", "switch", vs.lead_with_best)
        add(f, "Best-part length (seconds)", "lead_seconds", "entry", vs.lead_seconds)
        add(f, "Loop-friendly ending", "loop_ending", "switch", vs.loop_ending)
        add(f, "Progress bar", "progress_bar", "switch", vs.progress_bar)
        add(f, "Cut out dead air (jump cuts)", "jump_cuts", "switch", vs.jump_cuts)
        msg = ctk.StringVar()
        ctk.CTkLabel(win, textvariable=msg, font=font(13), text_color=AMBER, wraplength=540, justify="left", anchor="w").grid(
            row=4, column=0, sticky="w", padx=28, pady=(0, 16))

        def num(key: str, lo: float, hi: float, name: str):
            try:
                x = float(vars_[key].get())
            except ValueError:
                x = None
            if x is None or not lo <= x <= hi:
                msg.set(f"{name} must be a number between {lo:g} and {hi:g}.")
                return None
            return x

        def save() -> None:
            size = num("font_size", 20, 200, "Size")
            secs = num("hook_seconds", 0.5, 10, "Title seconds")
            cam = num("cam_ratio", 15, 60, "Face cam share")
            lead = num("lead_seconds", 1, 4, "Best-part length")
            rot = num("rotation_max", 0, 10, "Max repeats of one song")
            if rot is None:
                return
            gap = num("pre_drop_seconds", 0, 0.8, "Silence length")
            if gap is None:
                return
            lo = num("min_length", 5, 120, "Minimum clip length")
            hi = num("max_duration", 15, 180, "Longest clip")
            if size is None or secs is None or cam is None or lead is None or lo is None or hi is None:
                return
            if lo >= hi:
                msg.set("The minimum clip length must be shorter than the longest clip.")
                return
            cfg = load_settings()
            c = cfg.captions
            c.font = vars_["font"].get().strip() or c.font
            c.font_size = int(size)
            c.position = vars_["position"].get()
            c.animation = vars_["animation"].get()
            c.max_words_per_line = int(vars_["max_words_per_line"].get())
            c.uppercase = bool(vars_["uppercase"].get())
            for k in ("primary_color", "highlight_color", "emphasis_color", "outline_color"):
                setattr(c, k, colors[vars_[k].get()])
            c.outline = int(vars_["outline"].get())
            c.show_hook, c.hook_seconds = bool(vars_["show_hook"].get()), float(secs)
            c.title_style = next((k for k, t in tstyles.items() if t == vars_["title_style"].get()), "box")
            cfg.music.style = next((k for k, v in styles.items() if v == vars_["music_style"].get()), "auto")
            cfg.clips.min_length, cfg.clips.max_duration = float(lo), float(hi)
            cfg.music.rotation_max = int(rot)
            c.show_attribution, c.credit_in_caption = bool(vars_["show_attribution"].get()), bool(vars_["credit_in_caption"].get())
            cfg.folder_names = "number" if vars_["folder_names"].get().startswith("Hide") else "video"
            cfg.music.ai_placement = bool(vars_["ai_placement"].get())
            cfg.music.sound_tag, cfg.music.end_card = bool(vars_["sound_tag"].get()), bool(vars_["end_card"].get())
            cfg.music.ducking = next((k for k, t in duckings.items() if t == vars_["ducking"].get()), "smooth")
            cfg.music.talk_duck_db = next((k for k, t in dips.items() if t == vars_["talk_duck_db"].get()), None)
            cfg.music.beat_sync_cuts = bool(vars_["beat_sync_cuts"].get())
            cfg.music.pre_drop_dropout = next((k for k, t in aon.items() if t == vars_["pre_drop_dropout"].get()), "auto")
            cfg.music.pre_drop_seconds = float(gap)
            c.lyrics = next((k for k, t in aon.items() if t == vars_["lyrics"].get()), "auto")
            cfg.music.promo_build_bars = int(vars_["promo_build_bars"].get())
            v = cfg.video
            v.cam_ratio = float(cam) / 100
            v.crf = {"Standard": 20, "High (sharper, larger files)": 18, "Maximum (sharpest, largest files)": 16}.get(vars_["quality"].get(), 18)
            v.full_view_mode = "fill" if vars_["full_view_mode"].get().startswith("Zoom") else "blur"
            v.layout_style = next((k for k, t in layouts.items() if t == vars_["layout_style"].get()), "auto")
            v.punch_in, v.divider = bool(vars_["punch_in"].get()), bool(vars_["divider"].get())
            v.divider_color = colors[vars_["divider_color"].get()]
            v.face_tracking, v.lead_with_best, v.loop_ending = (bool(vars_["face_tracking"].get()), bool(vars_["lead_with_best"].get()),
                                                               bool(vars_["loop_ending"].get()))
            v.lead_seconds = float(lead)
            v.progress_bar, v.jump_cuts = bool(vars_["progress_bar"].get()), bool(vars_["jump_cuts"].get())
            save_settings(cfg)
            self.settings = load_settings()
            self._log("Style saved. New clips will use it.")
            win.destroy()

        def reset() -> None:
            cfg = load_settings()
            cfg.captions = CaptionStyle()
            save_settings(cfg)
            win.destroy()
            self._open_style()

        bar = ctk.CTkFrame(win, fg_color="transparent")
        bar.grid(row=3, column=0, sticky="ew", padx=28, pady=(0, 10))
        button(bar, "Save style", save, primary=True, width=130, height=42).pack(side="left")
        button(bar, "Cancel", win.destroy, width=100, height=42).pack(side="left", padx=8)
        button(bar, "Reset captions", reset, width=140, height=42, fg_color=CARD2, hover_color=RED).pack(side="right")

    def _open_watch(self) -> None:
        """Watch folder: new videos dropped in a folder (or links in links.txt) are queued automatically."""
        from tkinter import filedialog
        from engine.config import load_settings, save_settings
        from engine.watch import resolve_folder

        st = load_settings()
        w = st.watch
        win = ctk.CTkToplevel(self, fg_color=BG)
        win.title("Watch folder")
        win.geometry("600x520")
        win.minsize(520, 420)
        win.transient(self)
        win.after(150, lambda: (self._dark_caption(win), win.grab_set(), win.focus_force()))
        win.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(win, text="Watch folder", font=font(22, "bold"), text_color=TEXT).grid(row=0, column=0, sticky="w", padx=28, pady=(26, 2))
        ctk.CTkLabel(win, text="Drop a video file in the folder, or put links (one per line) in a file called links.txt, and clips "
                     "are made automatically.", font=font(13), text_color=MUTED, wraplength=540, justify="left").grid(row=1, column=0, sticky="w", padx=28)
        f = card(win)
        f.grid(row=2, column=0, sticky="ew", padx=28, pady=18)
        f.grid_columnconfigure(1, weight=1)
        on = ctk.BooleanVar(value=w.enabled)
        folder = ctk.StringVar(value=str(resolve_folder(self.ROOT, w.folder)))
        maxc = ctk.StringVar(value=str(w.max_clips or ""))
        songs = {"Automatic": "", "No song": "__none__"}
        songs.update({f"{x['title']} — {x['artist']}"[:60]: x["slug"] for x in self.songs_cache})
        cur = next((k for k, v in songs.items() if v == w.song), "Automatic")
        song = ctk.StringVar(value=cur)

        def lab(n, t):
            ctk.CTkLabel(f, text=t, font=font(13), text_color=TEXT, anchor="w").grid(row=n, column=0, sticky="w", padx=(20, 10), pady=10)
        lab(0, "Watch this folder")
        ctk.CTkSwitch(f, text="", variable=on, progress_color=VIOLET).grid(row=0, column=1, sticky="e", padx=20)
        lab(1, "Folder")
        ctk.CTkEntry(f, textvariable=folder, height=36, fg_color=CARD2, border_color=LINE).grid(row=1, column=1, sticky="ew", padx=(0, 20))
        button(f, "Browse…", lambda: folder.set(filedialog.askdirectory(initialdir=folder.get()) or folder.get()), width=100, height=32,
               fg_color=CARD2).grid(row=2, column=1, sticky="e", padx=20)
        lab(3, "Max clips per video (blank = default)")
        ctk.CTkEntry(f, textvariable=maxc, width=110, height=36, fg_color=CARD2, border_color=LINE).grid(row=3, column=1, sticky="e", padx=20)
        lab(4, "Song for these videos")
        ctk.CTkOptionMenu(f, variable=song, values=list(songs), height=36, fg_color=CARD2, button_color=LINE, button_hover_color=VIOLET,
                          dropdown_fg_color=CARD2).grid(row=4, column=1, sticky="ew", padx=20, pady=(0, 14))
        msg = ctk.StringVar()
        ctk.CTkLabel(win, textvariable=msg, font=font(13), text_color=AMBER, wraplength=540, justify="left").grid(row=4, column=0, sticky="w", padx=28)

        def save() -> None:
            cfg = load_settings()
            cfg.watch.enabled = bool(on.get())
            cfg.watch.folder = folder.get().strip() or "Watch"
            cfg.watch.max_clips = int(maxc.get()) if maxc.get().strip().isdigit() else 0
            cfg.watch.song = songs.get(song.get(), "")
            save_settings(cfg)
            resolve_folder(self.ROOT, cfg.watch.folder).mkdir(parents=True, exist_ok=True)
            self._log("Watch folder " + ("is ON: " + folder.get() if cfg.watch.enabled else "is off."))
            win.destroy()

        bar = ctk.CTkFrame(win, fg_color="transparent")
        bar.grid(row=3, column=0, sticky="ew", padx=28)
        button(bar, "Save", save, primary=True, width=100, height=42).pack(side="left")
        button(bar, "Open folder", lambda: open_path(Path(folder.get())), width=130, height=42, fg_color=CARD2).pack(side="left", padx=8)
        button(bar, "Cancel", win.destroy, width=90, height=42, fg_color=CARD2).pack(side="right")

    def _set_ai_light(self, state: str, text: str) -> None:
        color = {"ok": GREEN, "off": MUTED, "saved": AMBER, "testing": AMBER, "bad": RED}.get(state, MUTED)
        self.ai_light.configure(text="●  " + text, text_color=color)

    def check_ai(self, done=None) -> None:
        """Ask the AI provider one tiny question in the background and light the indicators: green only when the
        key really works AND the engine is using it."""
        from engine.config import SECRET_NAMES, load_settings, secret_hint

        st = load_settings()
        name = SECRET_NAMES.get(st.ai.provider, "")
        if st.ai.mode != "api":
            self._set_ai_light("off", "AI: off (built-in rules)")
            if done:
                done("off", "AI is off. The engine uses its built-in rules.")
            return
        if not secret_hint(name):
            self._set_ai_light("bad", "AI: no key")
            if done:
                done("bad", "AI is on but no key is saved.")
            return
        self._set_ai_light("testing", "AI: checking…")

        def work() -> None:
            try:
                r = self.server.test_ai()
            except Exception as e:
                r = {"ok": False, "message": str(e)}
            state = "ok" if r["ok"] else "bad"
            text = f"AI: connected · {st.ai.model}" if r["ok"] else "AI: key not working"
            msg = (f"Connected. {st.ai.model} is answering and is being used for your clips." if r["ok"] else r["message"])
            self.after(0, lambda: (self._set_ai_light(state, text), done and done(state, msg)))
        threading.Thread(target=work, daemon=True).start()

    def _open_settings(self) -> None:
        """Gear dialog: choose the AI provider and paste its API key (stored in config/secrets.env)."""
        from engine.config import PROVIDERS, SECRET_NAMES, connect_provider, detect_provider, load_settings, save_settings, secret_hint, set_secret

        st = load_settings()
        win = ctk.CTkToplevel(self, fg_color=BG)
        win.title("Settings")
        win.geometry("600x720")
        win.minsize(520, 420)
        win.transient(self)
        win.after(150, lambda: (self._dark_caption(win), win.grab_set(), win.focus_force()))
        win.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(win, text="Settings", font=font(22, "bold"), text_color=TEXT).grid(row=0, column=0, sticky="w", padx=28, pady=(26, 2))
        ctk.CTkLabel(win, text="AI is optional. Without a key the engine uses its built-in rules.", font=font(13),
                     text_color=MUTED).grid(row=1, column=0, sticky="w", padx=28)
        win.grid_rowconfigure(2, weight=1)        # the form scrolls if the window is short; buttons stay pinned below
        scroll = ctk.CTkScrollableFrame(win, fg_color="transparent")
        scroll.grid(row=2, column=0, sticky="nsew", padx=14, pady=10)
        scroll.grid_columnconfigure(0, weight=1)
        body = card(scroll)
        body.grid(row=0, column=0, sticky="ew", padx=14)
        body.grid_columnconfigure(0, weight=1)
        labels = {v["label"]: k for k, v in PROVIDERS.items()}
        cur = st.ai.provider if st.ai.provider in PROVIDERS else "gemini"

        def lab(text: str, r: int) -> None:
            ctk.CTkLabel(body, text=text, font=font(12, "bold"), text_color=MUTED, anchor="w").grid(
                row=r, column=0, sticky="w", padx=20, pady=(16 if r == 0 else 10, 2))

        lab("AI PROVIDER", 0)
        prov = ctk.StringVar(value=PROVIDERS[cur]["label"])
        model = ctk.StringVar(value=st.ai.model if st.ai.provider == cur else PROVIDERS[cur]["default_model"])
        key = ctk.StringVar()
        url = ctk.StringVar(value=st.ai.openai_base_url)
        hint = ctk.StringVar()
        msg = ctk.StringVar()

        def light(state: str, text: str) -> None:
            led.configure(text_color={"ok": GREEN, "saved": AMBER, "testing": AMBER, "bad": RED}.get(state, MUTED))
            hint.set(text)

        def refresh_hint(*_, check: bool = False) -> None:
            k = labels[prov.get()]
            h = secret_hint(SECRET_NAMES[k])
            cfg = load_settings()
            in_use = cfg.ai.mode == "api" and cfg.ai.provider == k
            if not h:
                entry.configure(placeholder_text="Paste your API key here")
                light("none", "No key saved yet")
                return
            # dots show that a key is stored, without ever showing the key
            entry.configure(placeholder_text="••••••••••••••••••••  saved — paste a new key to replace it")
            if in_use and check:
                light("testing", f"Key saved ({h}) — checking the connection…")
                self.check_ai(lambda st_, m: win.winfo_exists() and light(st_, f"Connected — in use  ({h})" if st_ == "ok" else m))
            elif in_use:
                light("saved", f"Key saved ({h}) — in use")
            else:
                light("saved", f"Key saved ({h}) — press Save to start using it")

        def on_provider(label: str) -> None:
            k = labels[label]
            model.set(PROVIDERS[k]["default_model"])
            refresh_hint(check=True)

        ctk.CTkOptionMenu(body, values=list(labels), variable=prov, command=on_provider, height=40, corner_radius=12,
                          fg_color=CARD2, button_color=LINE, text_color=TEXT).grid(row=1, column=0, sticky="ew", padx=20)
        lab("API KEY", 2)
        entry = ctk.CTkEntry(body, textvariable=key, show="•", height=42, corner_radius=12, fg_color=CARD2, border_color=LINE,
                             text_color=TEXT, placeholder_text="Paste your API key here")
        entry.grid(row=3, column=0, sticky="ew", padx=20)
        srow = ctk.CTkFrame(body, fg_color="transparent")
        srow.grid(row=4, column=0, sticky="w", padx=20, pady=(6, 0))
        led = ctk.CTkLabel(srow, text="●", font=font(18), text_color=MUTED, width=22)
        led.pack(side="left")
        ctk.CTkLabel(srow, textvariable=hint, font=font(13, "bold"), text_color=TEXT, anchor="w").pack(side="left", padx=(4, 0))

        def on_key(*_) -> None:
            d = detect_provider(key.get())          # a pasted key picks its own provider
            if d and labels[prov.get()] != d:
                prov.set(PROVIDERS[d]["label"])
                model.set(PROVIDERS[d]["default_model"])
                refresh_hint()
        key.trace_add("write", on_key)
        lab("MODEL", 5)
        ctk.CTkEntry(body, textvariable=model, height=40, corner_radius=12, fg_color=CARD2, border_color=LINE,
                     text_color=TEXT).grid(row=6, column=0, sticky="ew", padx=20)
        lab("API ADDRESS (only for OpenAI-compatible / custom hosts)", 7)
        ctk.CTkEntry(body, textvariable=url, height=40, corner_radius=12, fg_color=CARD2, border_color=LINE,
                     text_color=TEXT).grid(row=8, column=0, sticky="ew", padx=20, pady=(0, 18))
        refresh_hint(check=True)
        ctk.CTkLabel(win, textvariable=msg, font=font(13), text_color=AMBER, wraplength=500, justify="left", anchor="w").grid(
            row=4, column=0, sticky="w", padx=28, pady=(0, 20))

        def save(close: bool = False, check: bool = True) -> bool:
            k = labels[prov.get()]
            typed = key.get().strip()
            if typed and (len(typed) < 12 or " " in typed):
                msg.set("That does not look like an API key.")
                return False
            if not typed and not secret_hint(SECRET_NAMES[k]):
                msg.set("Paste an API key first.")
                return False
            if not model.get().strip():
                msg.set("Enter a model name.")
                return False
            cfg = load_settings()
            cfg.ai.mode, cfg.ai.provider = "api", k
            cfg.ai.model = model.get().strip()
            if url.get().strip():
                cfg.ai.openai_base_url = url.get().strip().rstrip("/")
            save_settings(cfg)
            if typed:
                set_secret(SECRET_NAMES[k], typed)
                key.set("")
            self.settings = load_settings()
            msg.set("")
            refresh_hint(check=check)
            if close:
                win.destroy()
            return True

        def test() -> None:
            if not save(close=False, check=False):
                return
            msg.set("")
            light("testing", "Testing the connection…")
            self.check_ai(lambda st_, m: win.winfo_exists() and (light(st_, "Connected — key in use" if st_ == "ok" else "Not working"),
                                                                 msg.set(("✓ " if st_ == "ok" else "✗ ") + m)))

        def off() -> None:
            cfg = load_settings()
            cfg.ai.mode = "heuristic"
            save_settings(cfg)
            set_secret(SECRET_NAMES[labels[prov.get()]], "")
            refresh_hint()
            self._set_ai_light("off", "AI: off (built-in rules)")
            msg.set("Key removed. The engine will use its built-in rules.")

        row = ctk.CTkFrame(win, fg_color="transparent")
        row.grid(row=3, column=0, sticky="ew", padx=28, pady=(0, 12))
        button(row, "Save", save, primary=True, width=100, height=42).pack(side="left")
        button(row, "Test connection", test, width=150, height=42, fg_color=CARD2, hover_color=LINE).pack(side="left", padx=8)
        button(row, "Remove key", off, width=110, height=42, fg_color=CARD2, hover_color=RED).pack(side="left")
        button(row, "Close", win.destroy, width=90, height=42, fg_color=CARD2).pack(side="right")

    def _pick_song(self) -> None:
        """Searchable song chooser (works with hundreds of songs)."""
        songs = self.songs_cache
        win = ctk.CTkToplevel(self, fg_color=BG)
        win.title("Choose a song")
        win.geometry("620x680")
        win.transient(self)
        win.after(150, lambda: (self._dark_caption(win), win.grab_set(), win.focus_force()))
        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(2, weight=1)
        head = ctk.CTkFrame(win, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=28, pady=(26, 4))
        ctk.CTkLabel(head, text="Choose a song", font=font(22, "bold"), text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(head, text=f"{len(songs)} analyzed songs — type to search by title, artist or mood", font=font(13),
                     text_color=MUTED).pack(anchor="w", pady=(4, 0))
        q = ctk.StringVar()
        entry = ctk.CTkEntry(win, textvariable=q, placeholder_text=f"Search {len(songs)} songs by title, artist or mood…", height=44,
                             corner_radius=14, fg_color=CARD2, border_color=LINE, placeholder_text_color=MUTED, text_color=TEXT)
        entry.grid(row=1, column=0, sticky="ew", padx=28, pady=(8, 12))
        lst = ctk.CTkScrollableFrame(win, fg_color=CARD, corner_radius=16)
        lst.grid(row=2, column=0, sticky="nsew", padx=28, pady=(0, 24))

        def choose(slug, label) -> None:
            self.song_choice = (slug, label)
            self.song_btn.configure(text=f"♪  Song: {label[:24]}")
            win.destroy()

        def row(title: str, sub: str, slug, label, accent: bool = False) -> None:
            base, hover = (VIOLET, PINK) if accent else (CARD2, LINE)
            f = ctk.CTkFrame(lst, fg_color=base, corner_radius=12, cursor="hand2")
            f.pack(fill="x", pady=4, padx=4)
            t = ctk.CTkLabel(f, text=title, font=font(14, "bold"), text_color="white" if accent else TEXT, anchor="w")
            t.pack(fill="x", padx=18, pady=(12, 0 if sub else 12))
            kids = [t]
            if sub:
                u = ctk.CTkLabel(f, text=sub, font=font(12), text_color=IVORY if accent else MUTED, anchor="w")
                u.pack(fill="x", padx=18, pady=(0, 12))
                kids.append(u)
            for w in (f, *kids):
                w.bind("<Button-1>", lambda e: choose(slug, label))
                w.bind("<Enter>", lambda e: f.configure(fg_color=hover))
                w.bind("<Leave>", lambda e: f.configure(fg_color=base))

        def refill(*_) -> None:
            for w in lst.winfo_children():
                w.destroy()
            row("Automatic", "Talking clips get the song under the voice, action clips get it over the scene", None, "Automatic", True)
            row("No song", "No music at all: just a batch of clean clips", "__none__", "No song")
            term = q.get().strip().lower()
            hits = [x for x in songs if not term or term in f"{x['title']} {x['artist']} {' '.join(x.get('mood') or [])}".lower()]
            for x in hits[:150]:
                sub = "  ·  ".join(v for v in (x["artist"] or "", f"{round(x['bpm'])} BPM" if x.get("bpm") else "",
                                               ", ".join((x.get("mood") or [])[:3])) if v)
                row("♪  " + x["title"], sub, x["slug"], x["title"])
            if len(hits) > 150:
                ctk.CTkLabel(lst, text=f"{len(hits) - 150} more — keep typing to narrow it down", text_color=MUTED, font=font(12)).pack(pady=10)
            if term and not hits:
                ctk.CTkLabel(lst, text="No songs match.", text_color=MUTED, font=font(13)).pack(pady=18)
        q.trace_add("write", refill)
        refill()
        entry.focus_set()

    def _page_songs(self, p) -> None:
        p.grid_columnconfigure(0, weight=1)
        p.grid_rowconfigure(1, weight=1)
        h = self._heading(p, "Songs", "Songs are analyzed automatically (BPM, hook, mood).")
        bar = ctk.CTkFrame(h, fg_color="transparent")
        bar.grid(row=0, column=1, rowspan=2, sticky="e", padx=(20, 0))
        button(bar, "＋  Add songs", self._add_songs, primary=True, width=150).pack(side="left", padx=(0, 12))
        button(bar, "Open folder", lambda: open_path(self.songs_dir), width=140).pack(side="left")
        c = card(p)
        c.grid(row=1, column=0, sticky="nsew")
        c.grid_columnconfigure(0, weight=1)
        c.grid_rowconfigure(0, weight=1)
        self.songs_list = ctk.CTkScrollableFrame(c, fg_color="transparent")
        self.songs_list.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        self.rights = ctk.CTkCheckBox(c, text="I own or control the rights to the songs in this folder (required to place them in clips)",
                                      text_color=MUTED, fg_color=VIOLET, hover_color=PINK, border_color=LINE, font=font(13))
        self.rights.grid(row=1, column=0, sticky="w", padx=30, pady=(0, 8))
        ctk.CTkLabel(c, text="Tip: name files “Artist - Title.mp3” to fill the artist automatically.", text_color=MUTED,
                     font=font(12)).grid(row=2, column=0, sticky="w", padx=30, pady=(0, 24))

    def _page_clips(self, p) -> None:
        p.grid_columnconfigure(0, weight=1)
        p.grid_rowconfigure(1, weight=1)
        h = self._heading(p, "Clips", "Finished vertical clips, best first. Double-check, then export.")
        bar = ctk.CTkFrame(h, fg_color="transparent")
        bar.grid(row=0, column=1, rowspan=2)
        button(bar, "Open clips folder", lambda: open_path(self.library), width=170).pack(side="left", padx=(0, 12))
        button(bar, "Export approved", self._export, primary=True, width=170).pack(side="left")
        c = card(p)
        c.grid(row=1, column=0, sticky="nsew")
        c.grid_columnconfigure(0, weight=1)
        c.grid_rowconfigure(0, weight=1)
        self.clips_list = ctk.CTkScrollableFrame(c, fg_color="transparent")
        self.clips_list.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)

    def _page_activity(self, p) -> None:
        p.grid_columnconfigure(0, weight=1)
        p.grid_rowconfigure(1, weight=1, uniform="a")
        p.grid_rowconfigure(2, weight=1, uniform="a")
        h = self._heading(p, "Activity", "Every video's progress, and the engine's live log.")
        button(h, "Open error log", self._open_error_log, width=150, height=38, fg_color=CARD2).grid(row=0, column=1, rowspan=2)
        c1 = card(p)
        c1.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
        c1.grid_columnconfigure(0, weight=1)
        c1.grid_rowconfigure(0, weight=1)
        self.streams_list = ctk.CTkScrollableFrame(c1, fg_color="transparent")
        self.streams_list.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        c2 = card(p)
        c2.grid(row=2, column=0, sticky="nsew")
        c2.grid_columnconfigure(0, weight=1)
        c2.grid_rowconfigure(0, weight=1)
        self.log = ctk.CTkTextbox(c2, fg_color="#0B0908", text_color="#B7AEA0", corner_radius=12,
                                  font=ctk.CTkFont(family="Consolas", size=12), state="disabled", wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)

    def _ffmpeg(self) -> bool:
        from engine import runtime
        return runtime.ffmpeg_ok()

    # ------------------------------------------------------------------ row builders
    def _redraw(self, key: str, sig, frame, build) -> None:
        if self.sigs.get(key) == sig:
            return
        self.sigs[key] = sig
        for w in frame.winfo_children():
            w.destroy()
        build()

    def _empty(self, frame, text: str) -> None:
        ctk.CTkLabel(frame, text=text, text_color=MUTED, font=font(13), justify="center").pack(pady=36)

    def _stream_row(self, parent, s: dict, n: int = 90, more: int = 0) -> None:
        r = ctk.CTkFrame(parent, fg_color=CARD2, corner_radius=14)
        r.pack(fill="x", pady=6, padx=2)
        r.grid_columnconfigure(0, weight=1)
        name = s["title"] or s["source_url"] or f"video {s['id']}"
        f = font(13, "bold")
        lab = ctk.CTkLabel(r, text=name, font=f, text_color=TEXT, anchor="w", width=10)
        lab.grid(row=0, column=0, sticky="ew", padx=(22, 8), pady=(18, 0))
        fit_text(self, r, lab, name, f, 22 + 8 + 120 + 22 + 8)
        active = s["status"] not in DONE
        sub = f"{s['clips']} clip(s)" + (f" · {s['error'][:80]}" if s.get("error") else "")
        ctk.CTkLabel(r, text=sub, font=font(11), text_color=RED if s.get("error") else MUTED, anchor="w", width=10).grid(
            row=1, column=0, sticky="ew", padx=(22, 8), pady=(2, 18))
        color = AMBER if active else RED if s.get("error") or s["status"] in ("failed", "error") else GREEN
        pill(r, s["status"], color).grid(row=0, column=1, rowspan=2, padx=(0, 12) if more else 22)
        if more:
            busy = s["id"] in self.more_busy
            button(r, "Rendering…" if busy else f"＋ {min(10, more)} more clips  ·  {more} left", lambda i=s["id"]: self._more_clips(i),
                   width=210, height=36, corner_radius=18, fg_color=LINE, hover_color=VIOLET,
                   state="disabled" if busy else "normal").grid(row=0, column=2, rowspan=2, padx=(0, 22))
        if active:
            bar = ctk.CTkProgressBar(r, mode="indeterminate", height=4, progress_color=VIOLET, fg_color=LINE)
            bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=22, pady=(0, 18))
            bar.start()

    def _clip_row(self, parent, c: dict, n: int = 80, compact: bool = False) -> None:
        r = ctk.CTkFrame(parent, fg_color=CARD2, corner_radius=14)
        r.pack(fill="x", pady=6, padx=2)
        r.grid_columnconfigure(1, weight=1)
        score = round(c["score"] or 0)
        badge(r, str(score), GREEN if score >= 70 else AMBER if score >= 50 else RED).grid(row=0, column=0, rowspan=2, padx=(20, 16), pady=16)
        name = c.get("title") or c.get("hook_choice") or f"Clip {c['id']}"
        f = font(13, "bold")
        lab = ctk.CTkLabel(r, text=name, font=f, text_color=TEXT, anchor="w", width=10)
        lab.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(18, 0))
        right = 44 + 20 if compact else 100 + 6 + 84 + 20
        fit_text(self, r, lab, name, f, 20 + 52 + 16 + 10 + right + 12)
        m = c.get("music") or {}
        song = (m.get("title") or m.get("song") or "") if isinstance(m, dict) else ""
        ctk.CTkLabel(r, text=f"{round(c['duration'] or 0)} s" + (f"  ·  ♪ {song}" if song else "  ·  no song"), font=font(11),
                     text_color=MUTED, anchor="w", width=10).grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(2, 18))
        play = lambda i=c["id"]: self._play_clip(i)
        if compact:
            button(r, "▶", play, primary=True, width=44, height=44, corner_radius=22).grid(row=0, column=2, rowspan=2, padx=(0, 20))
            return
        button(r, "▶  Play", play, primary=True, width=100, height=40, corner_radius=20).grid(row=0, column=2, rowspan=2, padx=6)
        button(r, "Folder", lambda i=c["id"]: self._open_clip_folder(i), fg_color=LINE, hover_color=VIOLET, width=84, height=40,
               corner_radius=20).grid(row=0, column=3, rowspan=2, padx=(0, 20))

    # ------------------------------------------------------------------ actions
    # ------------------------------------------------------------------ errors
    def report_callback_exception(self, exc, val, tb) -> None:
        """Called by Tk when anything in a button / timer / event handler crashes: keep the app alive and explain."""
        from engine import errors
        errors.log_exception("window", val, tb)
        self.report_error("window", val)

    def report_error(self, where: str, exc: BaseException) -> None:
        from engine import errors

        msg = errors.friendly(exc)

        def show() -> None:
            try:
                self._log(f"⚠ {msg}")
                if errors.should_notify(exc):
                    messagebox.showwarning("Something went wrong", f"{msg}\n\nThe app is still running. Full details are saved in:\n{errors.LOG_PATH}")
            except Exception:
                pass
        try:
            self.after(0, show)
        except Exception:
            pass

    def _open_error_log(self) -> None:
        from engine import errors
        if not errors.LOG_PATH.exists():
            messagebox.showinfo("Error log", "No errors have been recorded. 🎉")
            return
        open_path(errors.LOG_PATH.parent)

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _add_video_files(self) -> None:
        files = filedialog.askopenfilenames(title="Pick video files",
                                            filetypes=[("Video", "*.mp4 *.mkv *.mov *.webm *.flv *.ts *.m4v *.avi"), ("All", "*.*")])
        for f in files:
            self.links.insert("end", f + "\n")

    def _add_songs(self) -> None:
        files = filedialog.askopenfilenames(title="Pick songs",
                                            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg"), ("All", "*.*")])
        for f in files:
            src = Path(f)
            dest = self.songs_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
        if files:
            self._log(f"Copied {len(files)} song(s) into the Songs folder.")

    def _start(self) -> None:
        if self.links.get("1.0", "end").strip():          # anything typed but not yet added goes in with the chosen song
            self._add_to_queue()
        if not self.queue:
            messagebox.showinfo("Nothing to do", "Paste at least one YouTube link (or add a video file) first.")
            return
        ctype = {l: k for k, l in self.types}.get(self.type_var.get(), "auto")
        mc = self.max_clips.get().strip()
        max_clips = int(mc) if mc.isdigit() else None
        items, self.queue = list(self.queue), []
        self._render_queue()
        self.go.configure(state="disabled")
        self._log("Queuing videos (downloading starts in the background)…")

        def work() -> None:
            ok, errs = 0, []
            for it in items:
                try:
                    self.server._queue_source(it["link"], None, None, it.get("focus"), max_clips, ctype, it["song"])
                    ok += 1
                except Exception as e:
                    errs.append(f"{it['link']}: {e}")
                    self.queue.append(it)               # keep failed ones so nothing is lost
            if ok:
                self.server.worker.wake()
            self.after(0, lambda: self._queued(ok, errs))

        threading.Thread(target=work, daemon=True).start()

    def _queued(self, ok: int, errs: list[str]) -> None:
        self.go.configure(state="normal")
        self._render_queue()
        if ok:
            self._log(f"Queued {ok} video(s).")
            if not errs:
                self.show("Dashboard")
        for e in errs:
            self._log("Could not queue " + e)
        if errs:
            messagebox.showerror("Could not queue", "\n".join(errs)[:800])

    def _remaining(self, s: dict) -> int:
        """Moments that passed review but were left out of the first batch (renderable on demand)."""
        if s["status"] in ("failed", "error") or s["status"] not in DONE:
            return 0
        row = self.db.one("SELECT COUNT(*) FROM candidates WHERE stream_id=? AND status='selected' "
                          "AND id NOT IN (SELECT candidate_id FROM clips WHERE candidate_id IS NOT NULL)", (s["id"],))
        return int(row[0]) if row else 0

    def _more_clips(self, stream_id: int) -> None:
        if stream_id in self.more_busy:
            return
        self.more_busy.add(stream_id)
        self._log("Rendering the next 10 best moments…")

        def work() -> None:
            try:
                from engine.config import load_settings
                from engine.pipeline import Engine

                ids = Engine(load_settings(), log=lambda m: None).render_more(stream_id, 10)
                self.after(0, lambda: self._log(f"Made {len(ids)} more clip(s)."))
            except Exception as e:
                self.after(0, lambda m=str(e): self._log(f"Could not render more clips: {m}"))
            finally:
                self.more_busy.discard(stream_id)
        threading.Thread(target=work, daemon=True).start()

    def _clip_path(self, clip_id: int) -> Path | None:
        row = self.db.one("SELECT video_path FROM clips WHERE id=?", (clip_id,))
        if not row or not row["video_path"]:
            return None
        p = Path(row["video_path"])
        return p if p.is_absolute() else self.ROOT / p

    def _play_clip(self, clip_id: int) -> None:
        p = self._clip_path(clip_id)
        if p is None or not p.exists():
            messagebox.showinfo("Not ready", "That clip's file isn't available.")
            return
        os.startfile(str(p)) if hasattr(os, "startfile") else subprocess.Popen(["xdg-open", str(p)])

    def _open_clip_folder(self, clip_id: int) -> None:
        from engine.db import loads

        row = self.db.one("SELECT layout_json FROM clips WHERE id=?", (clip_id,))
        folder = (loads(row["layout_json"], {}) or {}).get("_folder") if row else None
        if folder and Path(folder).exists():
            open_path(Path(folder))
        else:
            p = self._clip_path(clip_id)
            open_path(p.parent if p else self.library)

    def _export(self) -> None:
        def work() -> None:
            try:
                r = self.server.export(self.server.ExportReq())
                self.after(0, lambda: (self._log(f"Exported {len(r['exported'])} clip(s) to {r['folder']}"), open_path(Path(r["folder"]))))
            except Exception as e:
                self.after(0, lambda m=str(e): self._log(f"Export failed: {m}"))
        threading.Thread(target=work, daemon=True).start()

    def _web(self) -> None:
        import webbrowser

        if self.web_port is None:
            import socket

            port = 8765
            while port < 8800:
                with socket.socket() as s:
                    if s.connect_ex(("127.0.0.1", port)) != 0:
                        break
                port += 1
            self.web_port = port

            def run() -> None:
                import uvicorn
                uvicorn.run(self.server.app, host="127.0.0.1", port=port, log_level="warning")
            threading.Thread(target=run, daemon=True).start()
            time.sleep(1.2)
        webbrowser.open(f"http://127.0.0.1:{self.web_port}")

    def _quit(self) -> None:
        if self.server.worker.busy and not messagebox.askyesno("Quit", "Clips are still being made. Quit anyway?\n"
                                                              "(Unfinished videos resume next time.)"):
            return
        try:
            self.review_page.leave()         # stop the clip sound and delete its temp file
            self.watcher.stop_flag.set()
        except Exception:
            pass
        self.destroy()
        os._exit(0)

    # ------------------------------------------------------------------ polling
    def _poll(self) -> None:
        try:
            self._scan_songs()
            songs = self.songs_cache = self.server.songs()
            streams = self.server.streams()[:30]
            clips = [c for c in self.server.clips(sort="new", limit=400) if c["has_video"]]       # newest first, never cut by score
            self._refresh_songs(songs)
            self._refresh_streams(streams)
            self._refresh_clips(clips)
            self._refresh_dashboard(songs, streams, clips)
            self._refresh_review_badge()
            self._refresh_log()
        except Exception as e:      # never let a refresh error kill the window
            self._log(f"(refresh error: {e})")
        self.after(POLL_MS, self._poll)

    def _scan_songs(self) -> None:
        if not self.rights.get():
            return
        for f in sorted(self.songs_dir.iterdir()):
            if f.suffix.lower() not in SONG_EXTS or str(f) in self.seen_songs:
                continue
            self.seen_songs[str(f)] = "Analyzing…"
            stem = f.stem
            artist, title = ("Unknown", stem)
            if " - " in stem:
                artist, title = [x.strip() for x in stem.split(" - ", 1)]
            threading.Thread(target=self._import_song, args=(f, title or stem, artist or "Unknown"), daemon=True).start()

    def _import_song(self, f: Path, title: str, artist: str) -> None:
        from engine.db import Database
        from engine.music.library import import_song

        s = self.settings
        try:
            import_song(Database(s.db_path), s.library / "songs", f, title, artist,
                        "Uploader confirmed ownership/control of rights in the desktop app", "desktop", False, 3, {},
                        s.transcription, log=lambda m: self.db.log("info", f"[music] {m}"))
            self.seen_songs[str(f)] = "Ready"
        except Exception as e:
            self.seen_songs[str(f)] = "Error: " + str(e)[:60]
            self.db.log("error", f"[music] import failed for '{title}': {e}")

    def confirm(self, title: str, message: str, yes_label: str, on_yes, danger: bool = True) -> None:
        """Themed "Are you sure?" dialog (same look as the rest of the app) instead of a grey system box."""
        win = ctk.CTkToplevel(self, fg_color=BG)
        win.title(title)
        win.geometry("520x300")
        win.resizable(False, False)
        win.transient(self)
        win.after(150, lambda: (self._dark_caption(win), win.grab_set(), win.focus_force()))
        win.grid_columnconfigure(0, weight=1)
        box = card(win)
        box.grid(row=0, column=0, sticky="nsew", padx=24, pady=(24, 12))
        box.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(box, text="⚠", font=font(30), text_color=RED if danger else AMBER).grid(row=0, column=0, pady=(18, 0))
        ctk.CTkLabel(box, text=title, font=font(19, "bold"), text_color=TEXT).grid(row=1, column=0, pady=(2, 6))
        ctk.CTkLabel(box, text=message, font=font(13), text_color=MUTED, wraplength=440, justify="center").grid(
            row=2, column=0, padx=20, pady=(0, 18))
        bar = ctk.CTkFrame(win, fg_color="transparent")
        bar.grid(row=1, column=0, pady=(0, 20))

        def yes() -> None:
            win.destroy()
            on_yes()
        no = button(bar, "Cancel", win.destroy, width=130, height=44, fg_color=CARD2)
        no.pack(side="left", padx=8)
        button(bar, yes_label, yes, width=170, height=44, fg_color=RED if danger else VIOLET,
               hover_color="#C23B40" if danger else PINK, text_color="white").pack(side="left", padx=8)
        win.bind("<Escape>", lambda e: win.destroy())

    def _remove_song(self, sid: int, title: str) -> None:
        self.confirm("Remove this song?",
                     f"“{title}” will be removed from your library and will no longer be used on new clips.\n\n"
                     "Clips you already made keep their sound. The original file is moved to Songs/_removed, so you can get it back.",
                     "Yes, remove it", lambda: self._do_remove_song(sid, title))

    def _do_remove_song(self, sid: int, title: str) -> None:
        from engine.music.library import file_hash

        try:
            row = self.db.one("SELECT file_hash FROM songs WHERE id=?", (sid,))
            h = row["file_hash"] if row else None
            self.server.delete_song(sid)             # database rows + the library copy
            moved = 0
            if h:                                    # the original in Songs/ would be re-imported on the next scan
                trash = self.songs_dir / "_removed"
                for f in self.songs_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in SONG_EXTS:
                        try:
                            same = file_hash(f) == h
                        except OSError:
                            same = False
                        if same:
                            trash.mkdir(exist_ok=True)
                            dest = trash / f.name
                            k = 1
                            while dest.exists():
                                dest = trash / f"{f.stem}_{k}{f.suffix}"
                                k += 1
                            f.replace(dest)
                            self.seen_songs.pop(str(f), None)
                            moved += 1
            self.songs_cache = [s for s in self.songs_cache if s.get("id") != sid]
            self._refresh_songs(self.server.songs())
            self._log(f"Removed “{title}”." + (" Original moved to Songs/_removed." if moved else ""))
        except Exception as e:
            self._log(f"Could not remove “{title}”: {e}")
            messagebox.showerror("Could not remove", str(e))

    def _edit_hook(self, sid: int) -> None:
        from engine.song_hook_ui import open_hook_editor
        open_hook_editor(self, sid)

    def _edit_song(self, sid: int, title: str, artist: str, explicit: bool) -> None:
        win = ctk.CTkToplevel(self)
        win.title("Edit song")
        win.geometry("480x440")
        win.minsize(480, 440)
        win.configure(fg_color=BG)
        win.transient(self)
        win.after(150, win.grab_set)
        win.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(win, text="Edit song", font=font(20, "bold"), text_color=TEXT).grid(row=0, column=0, sticky="w", padx=28, pady=(24, 12))
        fields = {}
        for i, (label, val) in enumerate((("Title", title), ("Artist", artist))):
            ctk.CTkLabel(win, text=label, font=font(12), text_color=MUTED).grid(row=1 + 2 * i, column=0, sticky="w", padx=28)
            e = ctk.CTkEntry(win, height=38, fg_color=CARD2, border_color=LINE, text_color=TEXT, font=font(13))
            e.insert(0, val)
            e.grid(row=2 + 2 * i, column=0, sticky="ew", padx=28, pady=(2, 10))
            fields[label] = e
        ex = ctk.CTkCheckBox(win, text="Explicit", text_color=MUTED, fg_color=VIOLET, hover_color=PINK, border_color=LINE, font=font(13))
        if explicit:
            ex.select()
        ex.grid(row=5, column=0, sticky="w", padx=28, pady=(0, 14))
        msg = ctk.CTkLabel(win, text="", text_color=RED, font=font(12))
        msg.grid(row=6, column=0, sticky="w", padx=28)
        bar = ctk.CTkFrame(win, fg_color="transparent")
        bar.grid(row=7, column=0, sticky="e", padx=28, pady=(6, 20))

        def save() -> None:
            t, a = fields["Title"].get().strip(), fields["Artist"].get().strip()
            if not t or not a:
                msg.configure(text="Title and artist can't be empty.")
                return
            self.server.patch_song(sid, self.server.SongPatch(title=t, artist=a, explicit=bool(ex.get())))
            win.destroy()
            self._refresh_songs(self.server.songs())

        button(bar, "Cancel", win.destroy, width=100).pack(side="left", padx=(0, 10))
        button(bar, "Save", save, primary=True, width=100).pack(side="left")

    def _refresh_songs(self, songs: list[dict]) -> None:
        rows = [(r["title"], r["artist"] or "", r["bpm"], ", ".join(r.get("mood") or [])[:40], "Error" if r["error"] else "Ready",
                 r["id"], bool(r.get("explicit"))) for r in songs]
        for path, state in self.seen_songs.items():
            if state != "Ready":
                rows.append((Path(path).stem, "", None, "", state, None, False))
        waiting = (not self.rights.get()) and any(p.suffix.lower() in SONG_EXTS and str(p) not in self.seen_songs
                                                  for p in self.songs_dir.iterdir())

        def build() -> None:
            if waiting:
                ctk.CTkLabel(self.songs_list, text="New songs are waiting — tick the rights box below to analyze them.",
                             text_color=AMBER, font=font(13)).pack(pady=(10, 4))
            if not rows and not waiting:
                self._empty(self.songs_list, "No songs yet.\nAdd some, then tick the rights box.")
            for title, artist, bpm, mood, state, sid, explicit in rows:
                r = ctk.CTkFrame(self.songs_list, fg_color=CARD2, corner_radius=14)
                r.pack(fill="x", pady=6, padx=2)
                r.grid_columnconfigure(1, weight=1)
                badge(r, "♪", VIOLET, 52, IVORY, 20).grid(row=0, column=0, rowspan=2, padx=(20, 16), pady=16)
                ctk.CTkLabel(r, text=title, font=font(14, "bold"), text_color=TEXT, anchor="w").grid(row=0, column=1, sticky="w", pady=(18, 0))
                sub = "  ·  ".join(x for x in (artist, f"{round(bpm)} BPM" if bpm else "", mood) if x)
                ctk.CTkLabel(r, text=sub or " ", font=font(12), text_color=MUTED, anchor="w").grid(row=1, column=1, sticky="w", pady=(2, 18))
                color = GREEN if state == "Ready" else RED if state.startswith("Error") else AMBER
                pill(r, state.split(":")[0], color).grid(row=0, column=2, rowspan=2, padx=(0, 12))
                if sid is not None:
                    button(r, "Hook…", lambda i=sid: self._edit_hook(i), width=80, height=34).grid(row=0, column=3, rowspan=2, padx=(0, 8))
                    button(r, "Edit", lambda i=sid, t=title, a=artist, e=explicit: self._edit_song(i, t, a, e),
                           width=70, height=34).grid(row=0, column=4, rowspan=2, padx=(0, 8))
                    button(r, "Remove", lambda i=sid, t=title: self._remove_song(i, t), width=84, height=34, fg_color=CARD,
                           hover_color=RED, text_color=RED).grid(row=0, column=5, rowspan=2, padx=(0, 22))
        self._redraw("songs", (tuple(rows), waiting), self.songs_list, build)

    def _refresh_streams(self, streams: list[dict]) -> None:
        busy = sum(1 for s in streams if s["status"] not in DONE)
        working = self.server.worker.busy or busy
        self.status.configure(text=f"●  Working · {busy or 1}" if working else "●  Idle", text_color=AMBER if working else GREEN)
        left = {s["id"]: self._remaining(s) for s in streams}
        sig = tuple((s["id"], s["status"], s["clips"], s.get("error"), left[s["id"]], s["id"] in self.more_busy) for s in streams)

        def build() -> None:
            if not streams:
                self._empty(self.streams_list, "Nothing yet — start a job from “New job”.")
            for s in streams:
                self._stream_row(self.streams_list, s, more=left[s["id"]])
        self._redraw("streams", sig, self.streams_list, build)

    def _refresh_clips(self, clips: list[dict]) -> None:
        sig = tuple((c["id"], c["score"], c["updated_at"]) for c in clips)

        def build() -> None:
            if not clips:
                self._empty(self.clips_list, "No clips yet. Finished clips appear here.")
            for c in clips:
                self._clip_row(self.clips_list, c)
        self._redraw("clips", sig, self.clips_list, build)

    def _refresh_review_badge(self) -> None:
        """Show how many clips wait in Review on the tab itself, and pull newly finished clips into an open Review tab."""
        from engine import review

        n = review.waiting_count(self.db)
        last = getattr(self, "_last_waiting", None)
        if last is not None and n > last:
            self._log(f"✦ {n - last} new clip(s) ready: open the Review tab (newest are first).")
        self._last_waiting = n
        b = self.nav_btns.get("Review")
        if b is not None:
            b.configure(text=f"   ⇄    Review" + (f"   ·   {n}" if n else ""))
        rp = getattr(self, "review_page", None)
        if rp is not None and self.current == "Review":
            rp.check_new(n)

    def _refresh_progress(self) -> None:
        from engine import progress
        snap = progress.snapshot(self.db)
        self.prog_title.configure(text=snap["headline"])
        self.prog_bar.set(snap["pct"] if snap["active"] else 0)
        self.prog_sub.configure(text=("Now: " + snap["detail"]) if snap["active"] and snap["detail"]
                                else "Nothing running. Start a job from “New job”.")
        if snap["active"]:
            tail = progress.fmt_eta(snap["eta_s"]) if snap["eta_s"] is not None else "estimating"
            n = f"{snap['clips_done']}/{snap['clips_total']} · " if snap["clips_total"] else ""
            self.status.configure(text=f"●  {n}{tail}", text_color=AMBER)

    def _refresh_dashboard(self, songs: list[dict], streams: list[dict], clips: list[dict]) -> None:
        self._refresh_progress()
        self.stat["clips"].configure(text=str(len(clips)))
        self.stat["top"].configure(text=str(round(max((c["score"] or 0) for c in clips))) if clips else "–")
        self.stat["videos"].configure(text=str(len(streams)))
        self.stat["songs"].configure(text=str(len(songs)))
        active = [s for s in streams if s["status"] not in DONE]
        queue = active or streams[:4]
        top = sorted(clips, key=lambda c: c["score"] or 0, reverse=True)[:5]

        def build_q() -> None:
            if not queue:
                self._empty(self.dash_queue, "Idle. Start a job from “New job”.")
            for s in queue:
                self._stream_row(self.dash_queue, s, 24)

        def build_c() -> None:
            if not top:
                self._empty(self.dash_clips, "No clips yet.")
            for c in top:
                self._clip_row(self.dash_clips, c, 17, True)
        self._redraw("dq", tuple((s["id"], s["status"], s["clips"]) for s in queue), self.dash_queue, build_q)
        self._redraw("dc", tuple((c["id"], c["score"], c["updated_at"]) for c in top), self.dash_clips, build_c)

    def _refresh_log(self) -> None:
        for r in self.server.logs(self.log_after, 100):
            self.log_after = max(self.log_after, r["id"])
            self._log(f"[{r['level']}] {r['message']}"[:400])


def main() -> None:
    from engine import errors

    try:
        app = App()
    except Exception as e:            # the window itself could not start: say why instead of vanishing
        errors.log_exception("startup", e)
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        messagebox.showerror("GTA 6 Clip Engine could not start", f"{errors.friendly(e)}\n\nDetails: {errors.LOG_PATH}")
        raise SystemExit(1)
    errors.install_hooks(app.report_error)
    app.mainloop()


if __name__ == "__main__":
    main()
