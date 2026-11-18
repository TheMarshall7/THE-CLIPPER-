"""The Review tab: clips one at a time, swipe style.

    ←  reject (filed under the selected reason; keys 1-9 pick a reason and reject in one go)
    →  keep        ↑  keep AND mark "going to post" (the engine learns from these)
    Z  undo        Space  replay

Plays the finished clip inside the app (picture through OpenCV, sound through Windows' own player).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import customtkinter as ctk

from engine import review
from engine.db import loads

try:
    import winsound
except ImportError:                     # not Windows: picture only
    winsound = None

PANEL_H = 560                           # on-screen height of the clip (9:16)


class Player:
    """Loops a clip in a label: frames paced by the wall clock, audio as an async looping WAV."""

    def __init__(self, app, label: ctk.CTkLabel, h: int = PANEL_H):
        self.app, self.label, self.h = app, label, h
        self.w = int(h * 9 / 16)
        self.cap = None
        self.wav: Path | None = None
        self.t0 = 0.0
        self.dur = 1.0
        self.fps = 30.0
        self.job = None
        self.gen = 0
        self.last_idx = -1
        self.muted = False
        self.playing = False

    def load(self, path: Path) -> None:
        import cv2

        self.stop()
        self.gen += 1
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
        self.dur = max(0.5, n / self.fps)
        vw, vh = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 9, self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 16
        self.w = int(self.h * vw / vh)
        self.last_idx = -1
        self.wav = None
        gen = self.gen

        def make_audio() -> None:       # decode the soundtrack once, off the UI thread
            if winsound is None:
                return
            out = Path(tempfile.gettempdir()) / f"gta6_review_{os.getpid()}_{gen}.wav"
            try:
                subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "2",
                                "-ar", "44100", "-c:a", "pcm_s16le", str(out)], check=True, timeout=60)
            except Exception:
                return
            if gen == self.gen and self.playing:
                self.wav = out
                self.app.after(0, lambda: self._start_audio(gen))
            else:
                out.unlink(missing_ok=True)
        threading.Thread(target=make_audio, daemon=True).start()
        self.playing = True
        self.t0 = time.perf_counter()
        self._tick(gen)

    def _start_audio(self, gen: int) -> None:
        if gen != self.gen or not self.wav or self.muted or winsound is None:
            return
        try:
            winsound.PlaySound(str(self.wav), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            self.t0 = time.perf_counter()        # re-sync the picture to the moment the sound starts
            if self.cap is not None:
                self.cap.set(1, 0)
                self.last_idx = -1
        except Exception:
            pass

    def _tick(self, gen: int) -> None:
        if gen != self.gen or self.cap is None:
            return
        import cv2
        from PIL import Image

        t = (time.perf_counter() - self.t0) % self.dur
        idx = int(t * self.fps)
        if idx < self.last_idx:                  # looped
            self.cap.set(1, 0)
            self.last_idx = -1
        frame = None
        while self.last_idx < idx:               # skip ahead if we fell behind
            ok, fr = self.cap.read()
            if not ok:
                self.cap.set(1, 0)
                self.last_idx = -1
                self.t0 = time.perf_counter()
                break
            self.last_idx += 1
            frame = fr
        if frame is not None:
            img = Image.fromarray(cv2.cvtColor(cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            self.photo = ctk.CTkImage(light_image=img, dark_image=img, size=(self.w, self.h))
            self.label.configure(image=self.photo, text="")
        self.job = self.label.after(12, lambda: self._tick(gen))

    def stop(self) -> None:
        self.playing = False
        self.gen += 1
        if self.job is not None:
            try:
                self.label.after_cancel(self.job)
            except Exception:
                pass
            self.job = None
        if winsound is not None:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.wav is not None:
            self.wav.unlink(missing_ok=True)
            self.wav = None


class ReviewPage:
    def __init__(self, app, page: ctk.CTkFrame):
        from engine import gui as G

        self.G, self.app, self.p = G, app, page
        self.db = app.db
        self.queue: list[dict] = []
        self.i = 0
        self.history: list[int] = []
        self.reason = review.REASONS[-1]
        self.view = "Review"
        self.busy = False
        self._build()
        for key, fn in (("<Left>", lambda: self.reject()), ("<Right>", lambda: self.decide("keep")), ("<Up>", lambda: self.decide("post")),
                        ("z", self.undo), ("<BackSpace>", self.undo), ("<space>", self.replay)):
            app.bind(key, lambda e, f=fn: self._key(e, f))
        for n in range(1, len(review.REASONS) + 1):
            app.bind(str(n), lambda e, k=n: self._key(e, lambda: self.reject(review.REASONS[k - 1])))

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        G, p = self.G, self.p
        p.grid_columnconfigure(0, weight=1)
        p.grid_rowconfigure(2, weight=1)
        h = G.ctk.CTkFrame(p, fg_color="transparent")
        h.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        h.grid_columnconfigure(0, weight=1)
        t = ctk.CTkFrame(h, fg_color="transparent")
        t.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(t, text="Review", font=G.font(30, "bold"), text_color=G.TEXT).pack(anchor="w")
        self.sub = ctk.CTkLabel(t, text="", font=G.font(13), text_color=G.MUTED)
        self.sub.pack(anchor="w")
        bar = ctk.CTkFrame(h, fg_color="transparent")
        bar.grid(row=0, column=1, sticky="e")
        self.tabs = ctk.CTkSegmentedButton(bar, values=["Review", "Rejected", "Kept"], command=self.set_view, height=38,
                                           fg_color=G.CARD2, selected_color=G.VIOLET, selected_hover_color=G.PINK,
                                           unselected_color=G.CARD2, text_color=G.IVORY)
        self.tabs.set("Review")
        self.tabs.pack(side="left", padx=(0, 10))
        G.button(bar, "🧠  Learning", self.learn_dialog, width=130, height=38).pack(side="left", padx=(0, 8))
        G.button(bar, "⚙  Auto-reject…", self.auto_dialog, width=150, height=38).pack(side="left")

        self.insight = ctk.CTkLabel(p, text="", font=G.font(12), text_color=G.MUTED, anchor="w", justify="left")
        self.insight.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        self.body = ctk.CTkFrame(p, fg_color="transparent")
        self.body.grid(row=2, column=0, sticky="nsew")
        self.body.grid_columnconfigure(1, weight=1)
        self.body.grid_rowconfigure(0, weight=1)

        # left: the clip + big buttons
        left = ctk.CTkFrame(self.body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="n", padx=(0, 22))
        self.screen = ctk.CTkLabel(left, text="", width=int(PANEL_H * 9 / 16), height=PANEL_H, fg_color="#000000", corner_radius=14)
        self.screen.pack()
        self.player = Player(self.app, self.screen)
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.pack(pady=14, fill="x")
        row.grid_columnconfigure((0, 1, 2), weight=1, uniform="b")
        G.button(row, "✕  Reject  ←", lambda: self.reject(), height=50, fg_color="#3A1618", hover_color=G.RED).grid(row=0, column=0, sticky="ew", padx=3)
        G.button(row, "✓  Keep  →", lambda: self.decide("keep"), height=50, fg_color="#1D3A22", hover_color="#2E7D45").grid(row=0, column=1, sticky="ew", padx=3)
        G.button(row, "★  Post  ↑", lambda: self.decide("post"), primary=True, height=50).grid(row=0, column=2, sticky="ew", padx=3)
        # after every decision a big UNDO bar appears for a few seconds, so a slip of the arrow key is one click to fix
        self.toast_box = ctk.CTkFrame(left, fg_color="#2A1E0E", corner_radius=14, border_width=1, border_color=G.AMBER)
        self.toast_box.grid_columnconfigure(0, weight=1)
        self.toast_text = ctk.CTkLabel(self.toast_box, text="", font=G.font(13, "bold"), text_color=G.IVORY, anchor="w")
        self.toast_text.grid(row=0, column=0, sticky="ew", padx=(14, 6), pady=10)
        G.button(self.toast_box, "↶  UNDO", self.undo, width=110, height=36, fg_color=G.AMBER, hover_color="#FFD36B",
                 text_color="#1A1205").grid(row=0, column=1, padx=(0, 10), pady=8)
        self.toast_job = None
        row2 = self.row2 = ctk.CTkFrame(left, fg_color="transparent")
        row2.pack(fill="x")
        G.button(row2, "↶ Undo last  (Z)", self.undo, width=150, height=34, fg_color=G.CARD2, text_color=G.AMBER).pack(side="left")
        G.button(row2, "⟲ Replay", self.replay, width=100, height=34, fg_color=G.CARD2).pack(side="left", padx=6)
        G.button(row2, "Open with sound", self.open_external, width=140, height=34, fg_color=G.CARD2).pack(side="right")

        # right: info + reasons + title options
        right = G.card(self.body)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        self.info = ctk.CTkLabel(right, text="", font=G.font(15, "bold"), text_color=G.TEXT, anchor="w", justify="left", wraplength=520)
        self.info.grid(row=0, column=0, sticky="ew", padx=24, pady=(22, 4))
        self.detail = ctk.CTkLabel(right, text="", font=G.font(13), text_color=G.MUTED, anchor="w", justify="left", wraplength=520)
        self.detail.grid(row=1, column=0, sticky="ew", padx=24)
        ctk.CTkLabel(right, text="TITLE OPTIONS  (click to re-render with that title)", font=G.font(12, "bold"), text_color=G.PINK,
                     anchor="w").grid(row=2, column=0, sticky="w", padx=24, pady=(18, 4))
        self.titles = ctk.CTkFrame(right, fg_color="transparent")
        self.titles.grid(row=3, column=0, sticky="ew", padx=20)
        ctk.CTkLabel(right, text="IF YOU REJECT, FILE IT UNDER  (press 1-9 to reject straight into a category)", font=G.font(12, "bold"),
                     text_color=G.PINK, anchor="w").grid(row=4, column=0, sticky="w", padx=24, pady=(18, 4))
        self.chips = ctk.CTkFrame(right, fg_color="transparent")
        self.chips.grid(row=5, column=0, sticky="ew", padx=20)
        self.chip_btns: dict[str, ctk.CTkButton] = {}
        for n, r in enumerate(review.REASONS, start=1):
            b = G.button(self.chips, f"{n}  {r}", lambda r=r: self.pick_reason(r), height=34, fg_color=G.CARD2)
            b.grid(row=(n - 1) // 3, column=(n - 1) % 3, sticky="ew", padx=4, pady=4)
            self.chips.grid_columnconfigure((n - 1) % 3, weight=1, uniform="c")
            self.chip_btns[r] = b
        self.pick_reason(self.reason)

        self.list = ctk.CTkScrollableFrame(self.body, fg_color=G.CARD, corner_radius=16)   # Rejected / Kept views

    # ------------------------------------------------------------------ data
    def refresh(self) -> None:
        """Called whenever the tab is opened."""
        if self.view != "Review":
            self.render_list()
            return
        keep_id = self.queue[self.i]["id"] if self.queue and self.i < len(self.queue) else None
        self.queue = review.to_review(self.db)
        self.i = next((n for n, c in enumerate(self.queue) if c["id"] == keep_id), 0)
        self.show_current()

    def set_view(self, name: str) -> None:
        self.view = name
        if name == "Review":
            self.list.grid_forget()
            self.body.winfo_children()[0].grid(row=0, column=0, sticky="n", padx=(0, 22))
            self.body.winfo_children()[1].grid(row=0, column=1, sticky="nsew")
            self.refresh()
        else:
            self.player.stop()
            self.body.winfo_children()[0].grid_forget()
            self.body.winfo_children()[1].grid_forget()
            self.list.grid(row=0, column=0, columnspan=2, sticky="nsew")
            self.render_list()

    def leave(self) -> None:
        self.player.stop()

    def _counts_line(self) -> None:
        c = review.counts(self.db)
        self.sub.configure(text=f"{c['waiting']} waiting  ·  {c['keep']} kept  ·  {c['post']} to post  ·  {c['reject']} rejected")
        self.insight.configure(text="   ".join(review.insights(self.db)))

    # ------------------------------------------------------------------ the swipe card
    def show_current(self) -> None:
        G = self.G
        self._counts_line()
        if self.i >= len(self.queue):
            self.player.stop()
            self.screen.configure(image=None, text="All caught up\n\nNothing left to review.\nNew clips will show up here.",
                                  text_color=G.MUTED, font=G.font(15))
            self.info.configure(text="Review finished")
            c = review.counts(self.db)
            lines = [f"{n} × {r}" for r, n in sorted(c["reasons"].items(), key=lambda kv: -kv[1])]
            self.detail.configure(text=("Rejected: " + "  ·  ".join(lines)) if lines else "")
            self._titles(None)
            return
        c = self.queue[self.i]
        cd = self.db.one("SELECT category, summary FROM candidates WHERE id=?", (c["candidate_id"],)) if c["candidate_id"] else None
        qc = loads(c["qc_json"], {}) or {}
        music = qc.get("music", {}) or {}
        meta = loads(c["metadata_json"], {}) or {}
        hooks = loads(c["hooks_json"], []) or []
        title = hooks[min(c["hook_choice"], len(hooks) - 1)] if hooks else meta.get("title", "")
        rv = self.db.one("SELECT ai_rating, ai_note FROM clip_review WHERE clip_id=?", (c["id"],))
        song = music.get("song")
        lines = [f"Score {c['score'] or 0:.0f}   ·   {(c['end'] - c['start']):.0f}s   ·   "
                 f"{(cd['category'] or '').replace('_', ' ') if cd else ''}",
                 f"Song: {song} ({music.get('mode', '')})" if song else ("No song" + (f" — {music['blocked']}" if music.get("blocked") else ""))]
        if qc.get("blocks"):
            lines.append("Checks failed: " + "; ".join(qc["blocks"])[:200])
        if rv and rv["ai_rating"] is not None:
            lines.append(f"AI: {rv['ai_rating']:.0f}/10 — {rv['ai_note'] or ''}")
        if cd and cd["summary"]:
            lines.append(cd["summary"][:240])
        self.info.configure(text=f"{self.i + 1} of {len(self.queue)}   ·   {title or '(no title)'}")
        self.detail.configure(text="\n".join(lines))
        self._titles(c)
        path = Path(c["video_path"])
        path = path if path.is_absolute() else self.app.ROOT / path
        if path.exists():
            self.player.load(path)
        else:
            self.player.stop()
            self.screen.configure(image=None, text="File missing", text_color=self.G.RED)

    def _titles(self, c) -> None:
        for w in self.titles.winfo_children():
            w.destroy()
        if not c:
            return
        hooks = loads(c["hooks_json"], []) or []
        for n, h in enumerate(hooks[:3]):
            on = n == c["hook_choice"]
            self.G.button(self.titles, h, lambda n=n: self.retitle(n), height=34, anchor="w",
                          fg_color=self.G.VIOLET if on else self.G.CARD2, text_color=self.G.IVORY if on else self.G.TEXT
                          ).pack(fill="x", padx=4, pady=3)
        if not hooks:
            ctk.CTkLabel(self.titles, text="(no title options saved for this clip)", text_color=self.G.MUTED).pack(anchor="w", padx=4)

    def retitle(self, n: int) -> None:
        if self.busy or self.i >= len(self.queue):
            return
        c = self.queue[self.i]
        self.busy = True
        self.player.stop()
        self.info.configure(text="Re-rendering with that title…")
        self.db.update("clips", c["id"], {"hook_choice": n})

        def work() -> None:
            try:
                from engine.config import load_settings
                from engine.pipeline import Engine
                Engine(load_settings(), log=lambda m: None).render_clip(c["id"])
            except Exception as e:
                self.app.after(0, lambda m=str(e): self.app._log(f"Re-render failed: {m}"))
            self.busy = False
            self.app.after(0, self.refresh)
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ actions
    def _key(self, event, fn) -> None:
        if self.app.current != "Review" or self.view != "Review":
            return
        if "entry" in str(event.widget).lower() or "text" in str(event.widget).lower():
            return
        fn()

    def pick_reason(self, r: str) -> None:
        self.reason = r
        for k, b in self.chip_btns.items():
            b.configure(fg_color=self.G.VIOLET if k == r else self.G.CARD2, text_color=self.G.IVORY if k == r else self.G.TEXT)

    def _current(self):
        return self.queue[self.i] if self.view == "Review" and 0 <= self.i < len(self.queue) and not self.busy else None

    def reject(self, reason: str | None = None) -> None:
        c = self._current()
        if not c:
            return
        if reason:
            self.pick_reason(reason)
        review.decide(self.db, c["id"], "reject", self.reason)
        self._advance(c)

    def decide(self, what: str) -> None:
        c = self._current()
        if not c:
            return
        review.decide(self.db, c["id"], what)
        self._advance(c)

    def _advance(self, c: dict) -> None:
        self.history.append(c["id"])
        rv = self.db.one("SELECT decision, reason FROM clip_review WHERE clip_id=?", (c["id"],))
        what = {"reject": "✕ Rejected" + (f" · {rv['reason']}" if rv and rv["reason"] else ""), "keep": "✓ Kept",
                "post": "★ Marked to post"}.get(rv["decision"] if rv else "", "Done")
        self.i += 1
        self.show_current()
        self._toast(what)

    def _toast(self, text: str) -> None:
        """Show the UNDO bar for 8 seconds."""
        self.toast_text.configure(text=text)
        self.toast_box.pack(fill="x", pady=(0, 10), before=self.row2)
        if self.toast_job is not None:
            try:
                self.app.after_cancel(self.toast_job)
            except Exception:
                pass
        self.toast_job = self.app.after(8000, self._hide_toast)

    def _hide_toast(self) -> None:
        self.toast_job = None
        self.toast_box.pack_forget()

    def undo(self) -> None:
        """Take back the last decision (reject / keep / post). Works even after you left the tab or restarted the app."""
        if self.view != "Review" or self.busy:
            return
        cid = self.history.pop() if self.history else review.last_manual(self.db)
        if cid is None:
            self.app._log("Nothing to undo.")
            return
        review.undo(self.db, cid)
        self.queue = review.to_review(self.db)
        self.i = next((n for n, c in enumerate(self.queue) if c["id"] == cid), 0)
        self.show_current()
        self._toast("↶ Undone — back on this clip")

    def replay(self) -> None:
        c = self._current()
        if c:
            self.show_current()

    def open_external(self) -> None:
        c = self._current()
        if c:
            self.app._play_clip(c["id"])

    # ------------------------------------------------------------------ Rejected / Kept lists
    def render_list(self) -> None:
        G = self.G
        self._counts_line()
        for w in self.list.winfo_children():
            w.destroy()
        rows = review.decisions(self.db, "reject") if self.view == "Rejected" else \
            [r for r in review.decisions(self.db) if r["decision"] in ("keep", "post") and not (r["auto"] and r["decision"] == "keep")]
        if not rows:
            ctk.CTkLabel(self.list, text="Nothing here yet.", text_color=G.MUTED, font=G.font(14)).pack(pady=40)
            return
        groups: dict[str, list[dict]] = {}
        for r in rows:
            key = (r["reason"] or "Other") if self.view == "Rejected" else ("★ Going to post" if r["decision"] == "post" else "✓ Kept")
            groups.setdefault(key, []).append(r)
        for key, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            ctk.CTkLabel(self.list, text=f"{key}   ({len(items)})", font=G.font(14, "bold"), text_color=G.PINK, anchor="w").pack(
                fill="x", padx=14, pady=(14, 4))
            for r in items[:200]:
                row = ctk.CTkFrame(self.list, fg_color=G.CARD2, corner_radius=12)
                row.pack(fill="x", padx=10, pady=3)
                name = Path(r["video_path"] or "").parent.name or f"Clip {r['id']}"
                ctk.CTkLabel(row, text=f"{name}   ·   score {r['score'] or 0:.0f}", anchor="w", text_color=G.TEXT).pack(
                    side="left", padx=14, pady=10)
                G.button(row, "Restore", lambda cid=r["id"]: self.restore(cid), width=90, height=30, fg_color=G.LINE).pack(side="right", padx=8)
                G.button(row, "▶", lambda cid=r["id"]: self.app._play_clip(cid), width=40, height=30, fg_color=G.LINE).pack(side="right")

    def restore(self, cid: int) -> None:
        review.undo(self.db, cid)
        self.render_list()

    # ------------------------------------------------------------------ what the engine has learned from your swipes
    def learn_dialog(self) -> None:
        from engine import learn
        from engine.config import load_settings, save_settings

        G = self.G
        st = learn.status(self.db)
        cfg = load_settings()
        win = ctk.CTkToplevel(self.app, fg_color=G.BG)
        win.title("Learning")
        win.geometry("560x560")
        win.minsize(500, 420)
        win.transient(self.app)
        win.after(150, lambda: (self.app._dark_caption(win), win.grab_set(), win.focus_force()))
        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(win, text="What the engine learned from you", font=G.font(22, "bold"), text_color=G.TEXT).grid(
            row=0, column=0, sticky="w", padx=28, pady=(26, 6))
        body = G.card(win)
        body.grid(row=1, column=0, sticky="nsew", padx=24, pady=8)
        body.grid_columnconfigure(0, weight=1)
        led = G.GREEN if st["active"] else G.AMBER
        head = (f"●  Learning is ON and shaping new picks" if st["active"] else
                f"●  Not enough yet: {st['need']} more swipe(s) to start (needs a few keeps and a few rejects)")
        ctk.CTkLabel(body, text=head, font=G.font(14, "bold"), text_color=led, anchor="w", wraplength=480, justify="left").grid(
            row=0, column=0, sticky="w", padx=22, pady=(20, 6))
        lines = [f"{st['decisions']} decisions so far:  {st['keeps']} kept / posted,  {st['rejects']} rejected"]
        if st["likes"]:
            lines.append("You tend to keep clips with: " + ", ".join(st["likes"]))
        if st["dislikes"]:
            lines.append("You tend to reject clips with: " + ", ".join(st["dislikes"]))
        lines += review.insights(self.db)
        ctk.CTkLabel(body, text="\n".join(lines), font=G.font(13), text_color=G.TEXT, anchor="w", justify="left", wraplength=480).grid(
            row=1, column=0, sticky="w", padx=22, pady=(0, 12))
        ctk.CTkLabel(body, text="Every Keep / Post / Reject you do by hand teaches it. New moments that look like the ones you keep "
                     "score higher, and ones like your rejects score lower, the next time you make clips. Auto-rejects never teach it.",
                     font=G.font(12), text_color=G.MUTED, anchor="w", justify="left", wraplength=480).grid(row=2, column=0, sticky="w", padx=22)
        use = ctk.BooleanVar(value=cfg.review.learn)
        ctk.CTkSwitch(body, text="Use what it learned when choosing clips", variable=use, progress_color=G.VIOLET, font=G.font(13)).grid(
            row=3, column=0, sticky="w", padx=22, pady=(14, 6))
        strengths = {"Gentle": 0.5, "Normal": 1.0, "Strong": 1.5}
        cur = next((k for k, v in strengths.items() if abs(v - cfg.review.learn_strength) < 0.01), "Normal")
        sv = ctk.StringVar(value=cur)
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.grid(row=4, column=0, sticky="w", padx=22, pady=(0, 18))
        ctk.CTkLabel(row, text="How strongly", font=G.font(13), text_color=G.TEXT).pack(side="left")
        ctk.CTkOptionMenu(row, variable=sv, values=list(strengths), width=120, height=34, fg_color=G.CARD2, button_color=G.LINE,
                          button_hover_color=G.VIOLET, dropdown_fg_color=G.CARD2).pack(side="left", padx=10)
        msg = ctk.StringVar()
        ctk.CTkLabel(win, textvariable=msg, font=G.font(13), text_color=G.AMBER, wraplength=500).grid(row=3, column=0, sticky="w", padx=28)

        def save() -> None:
            c2 = load_settings()
            c2.review.learn = bool(use.get())
            c2.review.learn_strength = strengths[sv.get()]
            save_settings(c2)
            self.app.settings = c2
            win.destroy()

        def forget() -> None:
            self.app.confirm("Forget everything it learned?", "The engine will start learning from scratch. Your clips and "
                             "decisions are not deleted; they just stop teaching it.", "Yes, forget it",
                             lambda: (learn.forget(self.db), msg.set("Forgotten. It will learn again from your next swipes."),
                                      self._counts_line()), danger=True)
        bar = ctk.CTkFrame(win, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", padx=28, pady=(8, 20))
        G.button(bar, "Save", save, primary=True, width=100, height=42).pack(side="left")
        G.button(bar, "Forget what it learned", forget, width=190, height=42, fg_color=G.CARD2, hover_color=G.RED).pack(side="left", padx=8)
        G.button(bar, "Close", win.destroy, width=90, height=42, fg_color=G.CARD2).pack(side="right")

    # ------------------------------------------------------------------ auto-reject rules
    def auto_dialog(self) -> None:
        from engine.config import load_settings, save_settings

        G = self.G
        st = load_settings()
        rs = st.review
        win = ctk.CTkToplevel(self.app, fg_color=G.BG)
        win.title("Auto-reject")
        win.geometry("600x720")
        win.minsize(520, 420)
        win.transient(self.app)
        win.after(150, lambda: (self.app._dark_caption(win), win.grab_set(), win.focus_force()))
        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(win, text="Auto-reject", font=G.font(22, "bold"), text_color=G.TEXT).grid(row=0, column=0, sticky="w", padx=28, pady=(26, 2))
        ctk.CTkLabel(win, text="Pick the rules. Clips that break one are rejected for you (filed as “Auto: …”) so you only swipe the good ones.",
                     font=G.font(13), text_color=G.MUTED, wraplength=540, justify="left").grid(row=1, column=0, sticky="w", padx=28)
        scroll = ctk.CTkScrollableFrame(win, fg_color="transparent")
        scroll.grid(row=2, column=0, sticky="nsew", padx=14, pady=10)
        scroll.grid_columnconfigure(0, weight=1)
        f = G.card(scroll)
        f.grid(row=0, column=0, sticky="ew", padx=14)
        f.grid_columnconfigure(1, weight=1)
        vars_: dict = {}

        def add(n: int, label: str, key: str, kind: str, value, tip: str = "") -> None:
            ctk.CTkLabel(f, text=label, font=G.font(13), text_color=G.TEXT, anchor="w", wraplength=300, justify="left").grid(
                row=n, column=0, sticky="w", padx=(20, 10), pady=8)
            v = ctk.BooleanVar(value=bool(value)) if kind == "switch" else ctk.StringVar(value=str(value))
            w = ctk.CTkSwitch(f, text="", variable=v, progress_color=G.VIOLET) if kind == "switch" else \
                ctk.CTkEntry(f, textvariable=v, width=110, height=34, fg_color=G.CARD2, border_color=G.LINE)
            w.grid(row=n, column=1, sticky="e", padx=20, pady=8)
            vars_[key] = v

        add(0, "Reject clips with a score under (0 = off)", "min_score", "entry", rs.min_score)
        add(1, "Reject clips shorter than … seconds (0 = off)", "min_seconds", "entry", rs.min_seconds)
        add(2, "Reject clips longer than … seconds (0 = off)", "max_seconds", "entry", rs.max_seconds)
        add(3, "Reject clips that failed a quality check (sound, sync, loudness)", "reject_blocked", "switch", rs.reject_blocked)
        add(4, "Reject clips where the song was dropped (speech not clear)", "reject_music_dropped", "switch", rs.reject_music_dropped)
        add(5, "Reject clips with no clear face / camera", "reject_no_face", "switch", rs.reject_no_face)
        add(6, "Reject clips with more dead air than … (0-1, 0 = off)", "max_silence", "entry", rs.max_silence)
        add(7, "Let the AI rate each clip (uses your API key)", "use_ai", "switch", rs.use_ai)
        add(8, "Reject if the AI rates it under (1-10)", "ai_min_rating", "entry", rs.ai_min_rating)
        add(9, "Run these rules automatically on every new batch", "auto_run", "switch", rs.auto_run)
        msg = ctk.StringVar()
        ctk.CTkLabel(win, textvariable=msg, text_color=G.AMBER, font=G.font(13), wraplength=540, justify="left", anchor="w").grid(
            row=4, column=0, sticky="w", padx=28, pady=(0, 14))

        def save() -> bool:
            try:
                new = dict(min_score=float(vars_["min_score"].get() or 0), min_seconds=float(vars_["min_seconds"].get() or 0),
                           max_seconds=float(vars_["max_seconds"].get() or 0), max_silence=float(vars_["max_silence"].get() or 0),
                           ai_min_rating=float(vars_["ai_min_rating"].get() or 6))
            except ValueError:
                msg.set("The number boxes need numbers.")
                return False
            cfg = load_settings()
            for k, v in new.items():
                setattr(cfg.review, k, v)
            for k in ("reject_blocked", "reject_music_dropped", "reject_no_face", "use_ai", "auto_run"):
                setattr(cfg.review, k, bool(vars_[k].get()))
            save_settings(cfg)
            self.app.settings = cfg
            return True

        def run() -> None:
            if not save():
                return
            msg.set("Checking clips…")

            def work() -> None:
                try:
                    res = review.auto_review(self.db, load_settings(), log=lambda m: None)
                    txt = f"Checked {res['checked']} clips, rejected {res['rejected']}." + \
                        ("  " + ", ".join(f"{n} × {r}" for r, n in res["by_rule"].items()) if res["by_rule"] else "")
                except Exception as e:
                    txt = f"Could not run: {e}"
                self.app.after(0, lambda: (msg.set(txt), self.refresh()))
            threading.Thread(target=work, daemon=True).start()

        bar = ctk.CTkFrame(win, fg_color="transparent")
        bar.grid(row=3, column=0, sticky="ew", padx=28, pady=(0, 8))
        G.button(bar, "Save", lambda: save() and win.destroy(), primary=True, width=100, height=42).pack(side="left")
        G.button(bar, "Save & run now", run, width=150, height=42).pack(side="left", padx=8)
        G.button(bar, "Close", win.destroy, width=90, height=42, fg_color=G.CARD2).pack(side="right")
