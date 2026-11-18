"""The Post tab: clips you marked "Post" in Review, with captions to copy, export folders, and a place to log results."""
from __future__ import annotations

from pathlib import Path

import customtkinter as ctk

from engine import posting, review


class PostPage:
    def __init__(self, app, page: ctk.CTkFrame):
        from engine import gui as G

        self.G, self.app, self.p, self.db = G, app, page, app.db
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)
        t = ctk.CTkFrame(page, fg_color="transparent")
        t.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(t, text="Post", font=G.font(30, "bold"), text_color=G.TEXT).pack(anchor="w")
        self.sub = ctk.CTkLabel(t, text="", font=G.font(13), text_color=G.MUTED, anchor="w", justify="left")
        self.sub.pack(anchor="w")
        self.times = ctk.CTkLabel(page, text="", font=G.font(12), text_color=G.MUTED, anchor="w", justify="left")
        self.times.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.list = ctk.CTkScrollableFrame(page, fg_color=G.CARD, corner_radius=16)
        self.list.grid(row=2, column=0, sticky="nsew")

    def refresh(self) -> None:
        G = self.G
        for w in self.list.winfo_children():
            w.destroy()
        todo, done = posting.to_post(self.db), posting.to_post(self.db, posted=True)
        self.sub.configure(text=f"{len(todo)} ready to post  ·  {len(done)} posted.  Mark clips with ★ Post in the Review tab.")
        bt = posting.best_times(self.db)
        self.times.configure(text="Best times:   " + "     ".join(f"{posting.PLATFORMS[k]}: {v}" for k, v in bt.items()))
        if not todo and not done:
            ctk.CTkLabel(self.list, text="Nothing marked to post yet.\n\nIn Review, press ↑ (or ★ Post) on the clips you will post.",
                         text_color=G.MUTED, font=G.font(14)).pack(pady=60)
            return
        for c in todo:
            self._row(c, False)
        if done:
            ctk.CTkLabel(self.list, text="POSTED", font=G.font(12, "bold"), text_color=G.PINK, anchor="w").pack(fill="x", padx=16, pady=(18, 4))
            for c in done:
                self._row(c, True)

    def _row(self, c: dict, posted: bool) -> None:
        G = self.G
        r = ctk.CTkFrame(self.list, fg_color=G.CARD2, corner_radius=14)
        r.pack(fill="x", padx=10, pady=5)
        r.grid_columnconfigure(0, weight=1)
        hooks = posting.loads(c.get("hooks_json"), []) or []
        title = hooks[min(c["hook_choice"], len(hooks) - 1)] if hooks else Path(c["video_path"] or "").parent.name
        ctk.CTkLabel(r, text=f"{title}   ·   score {c['score'] or 0:.0f}", font=G.font(14, "bold"), text_color=G.TEXT, anchor="w").grid(
            row=0, column=0, sticky="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(r, text=posting.caption(c, "tiktok")[:170], font=G.font(12), text_color=G.MUTED, anchor="w", justify="left",
                     wraplength=620).grid(row=1, column=0, sticky="w", padx=16)
        bar = ctk.CTkFrame(r, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="w", padx=12, pady=(8, 12))
        for key, label in posting.PLATFORMS.items():
            G.button(bar, f"Copy {label.split()[0]} caption", lambda k=key: self.copy(c, k), width=160, height=32, fg_color=G.LINE).pack(side="left", padx=3)
        G.button(bar, "▶", lambda: self.app._play_clip(c["id"]), width=40, height=32, fg_color=G.LINE).pack(side="left", padx=3)
        if not posted:
            G.button(bar, "Export for TikTok / Reels / Shorts", lambda: self.export(c), width=240, height=32).pack(side="left", padx=(10, 3))
            G.button(bar, "✓ Mark posted", lambda: self.mark(c), width=130, height=32, primary=True).pack(side="left", padx=3)
        else:
            views = ctk.StringVar()
            plat = ctk.StringVar(value="TikTok")
            ctk.CTkOptionMenu(bar, variable=plat, values=list(v.split()[0] if v != "YouTube Shorts" else "Shorts" for v in posting.PLATFORMS.values()),
                              width=110, height=32, fg_color=G.CARD, button_color=G.LINE).pack(side="left", padx=(10, 3))
            ctk.CTkEntry(bar, textvariable=views, placeholder_text="views", width=90, height=32, fg_color=G.CARD,
                         border_color=G.LINE).pack(side="left", padx=3)
            G.button(bar, "Save result", lambda: self.result(c, plat.get(), views.get()), width=110, height=32).pack(side="left", padx=3)

    def copy(self, c: dict, platform: str) -> None:
        self.app.clipboard_clear()
        self.app.clipboard_append(posting.caption(c, platform))
        self.app._log(f"Copied the {posting.PLATFORMS[platform]} caption.")

    def export(self, c: dict) -> None:
        from engine.config import load_settings
        try:
            folders = [posting.export(self.db, load_settings(), c["id"], k) for k in posting.PLATFORMS]
            self.app._log(f"Exported to {folders[0].parent.parent}")
            from engine.gui import open_path
            open_path(folders[0].parent.parent)
        except Exception as e:
            self.app._log(f"Export failed: {e}")

    def mark(self, c: dict) -> None:
        review.mark_posted(self.db, c["id"], "")
        self.refresh()

    def result(self, c: dict, plat_label: str, views: str) -> None:
        if not views.strip().isdigit():
            self.app._log("Enter the number of views as a number.")
            return
        key = {"TikTok": "tiktok", "Instagram": "instagram", "Shorts": "youtube_shorts"}.get(plat_label, "tiktok")
        posting.record_result(self.db, c["id"], key, int(views))
        self.app._log("Result saved. The engine will use it to suggest better posting times.")
        self.refresh()
