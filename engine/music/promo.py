"""Song-first clipping for music promotion.

The usual engine picks a clip and then looks for a song that fits it. For music marketing the song is the product, so
this module turns that around:

  - choose_song / window_lengths: which song the batch is promoting and how long its build-up and hook are
  - biggest_event / reshape:      find the biggest hit in a moment (explosion, crash, scream) and cut the clip AROUND it,
                                  so a few bars of song build up and the chorus downbeat lands exactly on that hit
  - music_fit:                    how well a moment can carry a song (big hit, little talking over the hook, no game music)
  - snap_cuts_to_beats:           move each jump cut onto the song's beat grid without moving anything else in the clip
  - dropout_window:               a short silence right before the chorus hits (the classic "drop")
  - lyric_lines:                  the song's timed lyrics on the clip's output timeline, split into short caption lines

Everything here is plain functions over numbers so it can be tested without rendering anything.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from engine.db import Database
from engine.music.library import effective_analysis
from engine.music.moments import meaningful, talk_share


# ------------------------------------------------------------------ which song, how long
def choose_song(db: Database, settings, slug: Optional[str] = None):
    """The song a promo batch is cut for: the job's song, else the campaign song, else the highest-priority song
    with confirmed rights and a beat grid. Returns the songs row or None."""
    for want in (slug, getattr(settings.music, "campaign_song", None)):
        if want and want != "__none__":
            row = db.one("SELECT * FROM songs WHERE slug=?", (want,))
            if row is not None and "bars" in effective_analysis(row):
                return row
    rows = db.query("SELECT s.* FROM songs s JOIN song_rights_confirmations r ON r.song_id=s.id "
                    "ORDER BY COALESCE(s.campaign_priority, 3) DESC, s.id DESC")
    return next((r for r in rows if "bars" in effective_analysis(r)), None)


def window_lengths(a: dict, build_bars: int = 4) -> tuple[float, float]:
    """(seconds of song before the chorus downbeat, seconds of hook after it) for one song analysis."""
    bar = float(a.get("bar_s") or 2.0)
    build = float(np.clip(build_bars * bar, 3.0, 10.0))
    hook = a.get("hook") or {}
    db = float(hook.get("downbeat", hook.get("start", 0.0)))
    if hook.get("user") and hook.get("end"):
        after = float(hook["end"]) - db
    else:
        nxt = [s["end"] for s in a.get("sections", []) if s["start"] <= db + 0.05 < s["end"]]
        after = (nxt[0] - db) if nxt else 8 * bar
    return round(build, 3), round(float(np.clip(after, 6.0, 20.0)), 3)


# ------------------------------------------------------------------ the biggest hit, and the window around it
def biggest_event(t: np.ndarray, score: np.ndarray, start: float, end: float, smooth: int = 3) -> Optional[tuple[float, float]]:
    """Time and strength (0-1) of the loudest excitement spike inside [start, end]."""
    sel = np.flatnonzero((t >= start) & (t <= end))
    if sel.size == 0:
        return None
    s = score[sel]
    if smooth > 1 and s.size >= smooth:            # one-hop clicks shouldn't beat a real explosion
        s = np.convolve(s, np.ones(smooth) / smooth, mode="same")
    i = int(np.argmax(s))
    # the hit itself is the onset: walk back to where the spike starts rising
    j = i
    while j > 0 and s[j - 1] < s[j] and s[j] - s[j - 1] > 0.02 and i - j < 4:
        j -= 1
    return float(t[sel[j]]), float(min(1.0, max(0.0, score[sel[i]])))


def refine_onset(wav_path, t_coarse: float, before: float = 1.2, after: float = 0.8, frame_ms: float = 10.0) -> float:
    """The exact moment the hit starts (to ~10 ms): the sharpest jump in loudness near the coarse 0.5 s estimate.
    The excitement curve is pooled over half-second windows and smeared, which is far too loose for a chorus to land on."""
    import soundfile as sf

    try:
        info = sf.info(str(wav_path))
        sr = info.samplerate
        a = max(0.0, t_coarse - before)
        x, _ = sf.read(str(wav_path), start=int(a * sr), stop=int((t_coarse + after) * sr), dtype="float32", always_2d=True)
    except Exception:
        return t_coarse
    x = x.mean(axis=1)
    hop = max(1, int(sr * frame_ms / 1000))
    n = x.size // hop
    if n < 8:
        return t_coarse
    e = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    # loudness jump between the 50 ms after and the 50 ms before each frame: an onset, not just a loud frame
    k = 5
    jump = np.full(n, -np.inf)
    for i in range(k, n - k):
        jump[i] = e[i:i + k].mean() - e[i - k:i].mean()
    i = int(np.argmax(jump))
    if not np.isfinite(jump[i]) or jump[i] < 6.0:      # no clear attack: keep the coarse estimate
        return t_coarse
    return round(a + i * hop / sr, 3)


def reshape(anchor: float, build_s: float, after_s: float, media_dur: float) -> tuple[float, float]:
    """Clip window that puts `anchor` build_s into the clip and plays after_s of hook after it.
    Near the start/end of the video the window slides instead of shrinking."""
    start, end = anchor - build_s, anchor + after_s
    if start < 0:
        start = 0.0
    if end > media_dur:
        end = media_dur
    return round(start, 3), round(end, 3)


def music_fit(anchor: float, strength: float, words: list[dict], conflicts: list[tuple[float, float]],
              window: tuple[float, float], build_s: float, after_s: float) -> tuple[float, dict]:
    """0-100: how well this moment carries a song. Source-time inputs.
      hit      - how big the hit is (the chorus lands on it)
      clear    - how little meaningful talking there is over the hook (music would bury it)
      no_music - the game's own music doesn't play inside the window
      room     - the full build-up and hook fit inside the video"""
    spans = meaningful(words)
    clear = 1.0 - talk_share(spans, anchor - 0.5, anchor + after_s)
    a, b = window
    clash = sum(max(0.0, min(c1, b) - max(c0, a)) for c0, c1 in conflicts) / max(1e-6, b - a)
    room = min(1.0, (anchor - a) / max(1e-6, build_s)) * min(1.0, (b - anchor) / max(1e-6, after_s))
    parts = {"hit": round(100 * strength, 1), "clear": round(100 * clear, 1),
             "no_music": round(100 * (1 - min(1.0, clash * 2)), 1), "room": round(100 * room, 1)}
    total = 0.40 * parts["hit"] + 0.35 * parts["clear"] + 0.15 * parts["no_music"] + 0.10 * parts["room"]
    return round(total, 1), parts


def blend(viral: float, fit: float, fit_share: float = 0.45) -> float:
    return round(max(0.0, min(100.0, (1 - fit_share) * viral + fit_share * fit)), 1)


# ------------------------------------------------------------------ beat grid on the clip timeline
def out_beats(a: dict, placement: dict, subdivide: int = 2) -> np.ndarray:
    """Beat (and off-beat) times on the clip's OUTPUT timeline while the song plays."""
    beats = np.array(a.get("beats") or [], dtype=float)
    if beats.size < 2:
        bars = np.array(a.get("bars") or [], dtype=float)
        if bars.size < 2:
            return np.zeros(0)
        beats = np.concatenate([np.linspace(x, y, 5)[:-1] for x, y in zip(bars, bars[1:])])
    if subdivide > 1:
        mids = [beats[:-1] + (beats[1:] - beats[:-1]) * k / subdivide for k in range(1, subdivide)]
        beats = np.sort(np.concatenate([beats, *mids]))
    s_in, s_out = float(placement["song_in"]), float(placement["song_out"])
    sel = beats[(beats >= s_in) & (beats <= s_out)]
    return float(placement["clip_in"]) + (sel - s_in)


def snap_cuts_to_beats(keep_abs: list[tuple[float, float]], beats_out: np.ndarray, words: list[dict],
                       region: tuple[float, float], max_shift: float = 0.12,
                       fixed: list[float] = ()) -> tuple[list[tuple[float, float]], list[float]]:
    """Move every jump cut that falls inside `region` (output time) onto the nearest beat in `beats_out`.

    A cut between kept pieces k and k+1 moves by d when piece k gains d at its end AND piece k+1 loses d at its start.
    The total length before every other cut stays the same, so nothing else in the clip moves: words, the payoff and the
    chorus downbeat stay exactly where they were. Only the silent padding around the cut changes, and a cut is skipped
    if moving it would clip a word, a `fixed` source time (the payoff), or eat more than the gap that was removed."""
    keep = [list(x) for x in keep_abs]
    shifts: list[float] = []
    if len(keep) < 2 or beats_out.size == 0:
        return [tuple(x) for x in keep], shifts
    acc = 0.0
    for k in range(len(keep) - 1):
        acc += keep[k][1] - keep[k][0]
        t = acc                                            # output time of the cut after piece k
        if not (region[0] <= t <= region[1]):
            continue
        nb = float(beats_out[np.argmin(np.abs(beats_out - t))])
        d = nb - t
        if abs(d) < 0.005 or abs(d) > max_shift:
            continue
        b_k, a_n = keep[k][1], keep[k + 1][0]
        if a_n - b_k < abs(d) + 0.02:                      # not enough removed gap to slide through
            continue
        if d > 0:      # piece k plays d longer into the gap; piece k+1 starts d later
            lo, hi = a_n, a_n + d
            clipped = any(w["end"] > lo and w["start"] < hi for w in words)
            if keep[k + 1][1] - hi < 0.1:
                continue
        else:          # piece k ends |d| earlier; piece k+1 starts |d| earlier, out of the gap
            lo, hi = b_k + d, b_k
            clipped = any(w["end"] > lo and w["start"] < hi for w in words)
            if lo - keep[k][0] < 0.1:
                continue
        if clipped or any(lo <= f <= hi for f in fixed):
            continue
        keep[k][1] = round(b_k + d, 4)
        keep[k + 1][0] = round(a_n + d, 4)
        acc += d                                           # piece k grew by d (already counted at its old length)
        shifts.append(round(d, 4))
    return [tuple(x) for x in keep], shifts


# ------------------------------------------------------------------ the drop
def dropout_window(a: dict, placement: dict, length: Optional[float] = None) -> Optional[tuple[float, float]]:
    """(start, end) on the output timeline of the silence right before the chorus downbeat: one beat, at most 0.5 s.
    None when the placement has no synced drop or the song hasn't played long enough to drop out of."""
    sync = placement.get("payoff_sync")
    if not sync:
        return None
    hit = float(sync["clip_time"])
    if length is None:
        bpm = float(a.get("bpm") or 120.0)
        length = min(0.5, 60.0 / max(40.0, bpm))
    length = float(np.clip(length, 0.1, 0.8))
    if hit - float(placement["clip_in"]) < length + 1.0:
        return None
    return round(hit - length, 4), round(hit, 4)


# ------------------------------------------------------------------ lyrics on screen
def lyric_lines(lyrics: list[dict], placement: dict, max_words: int = 6) -> list[tuple[str, float, float]]:
    """Timed lyric lines that play inside the placement, mapped to output time and split into short lines.
    Long Whisper segments are split evenly by word count over their duration."""
    s_in, s_out, c_in = float(placement["song_in"]), float(placement["song_out"]), float(placement["clip_in"])
    out: list[tuple[str, float, float]] = []
    for l in lyrics or []:
        a, b = float(l["start"]), float(l["end"])
        if b <= s_in or a >= s_out or not str(l.get("text", "")).strip():
            continue
        words = str(l["text"]).split()
        chunks = [words[i:i + max_words] for i in range(0, len(words), max_words)]
        step = (b - a) / len(chunks)
        for n, ch in enumerate(chunks):
            t0, t1 = a + n * step, a + (n + 1) * step
            t0, t1 = max(t0, s_in), min(t1, s_out)
            if t1 - t0 < 0.25:
                continue
            out.append((" ".join(ch), round(c_in + t0 - s_in, 3), round(c_in + t1 - s_in, 3)))
    return out


def ai_brief(title: str, artist: str, a: dict, build_s: float, after_s: float, settings, song_first: bool = True) -> str:
    """What the engine will do with the song, in plain words, for the AI that judges and ranks moments and writes copy.
    Built from the live settings so the AI's picture of the edit always matches what is actually rendered."""
    ms, cs = settings.music, settings.captions
    mood = ", ".join((a.get("mood") or {}).get("tags") or []) or "unknown"
    parts = [f"MUSIC PLAN: this batch promotes the song '{title}' by {artist} ({a.get('bpm', 0):.0f} BPM, mood: {mood}). "
             f"Each clip is cut around ONE hit: about {build_s:.0f}s of the song building up, then the chorus downbeat lands on the "
             f"exact instant the hit starts (explosion, crash, impact, gunshot, scream), then about {after_s:.0f}s of the chorus "
             "plays over what happens next."]
    if wants(ms.pre_drop_dropout, song_first):
        parts.append("All sound cuts out for one beat right before the hit, so the hit and the chorus land together out of silence.")
    if ms.beat_sync_cuts:
        parts.append("Jump cuts are moved onto the song's beat.")
    if ms.ducking == "smooth":
        parts.append("When people talk, the song sits at one steady lower level under the whole stretch of talking.")
    else:
        parts.append("The song dips under each spoken phrase and comes back up in between.")
    if wants(cs.lyrics, song_first):
        parts.append("The song's lyrics appear on screen in the top part of the frame after the opening title card.")
    parts.append("Judge each moment by how well it carries the song: best is a single dominant, clearly visible hit with an "
                 "aftermath that still looks good under the chorus (wreckage, a getaway, a reaction) and no line the viewer must "
                 "hear. Score low when the point of the moment is something said, when several equal hits compete, when the "
                 "hit comes in the first 3 seconds (no room for a build-up), or when the game's own music is playing. "
                 "Set payoff_time to the instant the hit STARTS, not its loudest point or the reaction after it.")
    return " ".join(parts)


def wants(setting: str, song_first: bool) -> bool:
    """auto = only for song-first (music promo) clips."""
    return setting == "on" or (setting == "auto" and song_first)


def song_lyrics(db: Database, song_id: int) -> list[dict]:
    row = db.one("SELECT * FROM songs WHERE id=?", (song_id,))
    if row is None:
        return []
    lyr = effective_analysis(row).get("lyrics") or []
    if not lyr:
        lyr = [dict(r) for r in db.query("SELECT start, end, text FROM song_lyrics WHERE song_id=? ORDER BY start", (song_id,))]
    return lyr


__all__ = ["choose_song", "window_lengths", "biggest_event", "reshape", "music_fit", "blend", "out_beats",
           "snap_cuts_to_beats", "dropout_window", "lyric_lines", "wants", "song_lyrics"]
