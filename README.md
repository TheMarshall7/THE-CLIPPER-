# GTA 6 Viral Clip Intelligence Engine

Local engine that finds the strongest moments in GTA 6 streams, edits them into 9:16 Shorts with
word-by-word captions and hooks, and places your own songs on the clips where they fit.
Media processing is 100% local (FFmpeg, Faster-Whisper, OpenCV, librosa).

## Install from GitHub (Windows): the quick way

Download the repo (green **Code** button, then **Download ZIP**, or `git clone`), unzip it, and double-click
**`Launch GTA6 Clipper.bat`**. It installs everything the first time (Python 3.11+ is the only thing you must have
already), then opens the app. Put your own songs in the `Songs` folder it creates.

### The manual way

You need **Python 3.11+** and **FFmpeg**. Then, in the folder you cloned:

```
winget install Gyan.FFmpeg                  # once; restart the terminal afterwards so ffmpeg is on PATH
python -m venv .venv
.venv\Scripts\pip install -e .
.venv\Scripts\python -m engine.gui         # opens the desktop app
```

- First run picks settings for your PC and downloads the small face model. The speech model (about 1.6 GB) downloads the
  first time a video is transcribed.
- **NVIDIA GPU** (much faster transcription): also run `.venv\Scripts\pip install -e ".[gpu]"`.
- Your songs go in the `Songs` folder it creates; clips appear in `Clips`.
- Add an AI key (optional) with the gear **Settings** button inside the app. Keys are stored only on your PC.
- To build the standalone `GTA6Clipper.exe` yourself: `.venv\Scripts\pip install -e ".[build]"` then
  `.venv\Scripts\python scripts/build_standalone.py` (needs ffmpeg on PATH; produces about 3 GB).
- The `.exe` files are not stored in this repo (too large); everything above builds from source.

## Easiest way once built: the dashboard

Double-click **`GTA6Clipper.exe`** (in this folder). A small window starts the engine and your browser
opens the dashboard at http://127.0.0.1:8765 (only reachable from this computer). Keep the window open
while you work; close it to stop.

- **Dashboard**: stats, the stream being processed with live step progress, live log, AI brain switch.
- **New job**: paste one or many Twitch/YouTube/Kick links (one per line) or file paths, or drag in video files.
- **Music**: upload songs (rights confirmation required; analyzed automatically), set priority, play,
  remove; edit artist brand profiles (good-fit moments, avoid-list, CTA, sound tag, volume).
- **Creators**: add creators, set permission status and whether added music is allowed.
- **Streams**: every stream, its status, per-step timings, retry failed ones.
- **Clips**: video previews, score + sub-scores, hook, why it was picked, music decision,
  approve/reject, "no-song version", export approved clips.
- **Music / Creators**: song library analysis; creator permission settings.

The exe needs no Claude Code. First run sets itself up (hardware detection, face model); it checks FFmpeg,
picks another port if 8765 is busy, and the page has an **Engine health** panel and a **Quit** button.
Video type on **New job**: *Streamer stream* or *Gameplay/trailer/cutscene*. Music volume per clip: Trailer (default
for cinematic), Featured, Balanced, Subtle.

Rebuild the launcher after moving the folder (it finds `.venv` next to itself):
`.venv/Scripts/pyinstaller --onefile --console --name GTA6Clipper --distpath . --workpath storage/build --specpath storage/build scripts/launcher.py`

## Desktop app: songs per link, readable clip folders

- **New job** is a queue: paste links, pick the song for them (searchable list, or *Automatic* / *No song*), press
  **Add to queue**. Each link keeps its own song; press **Analyze & make clips** to run them all. A chosen song bypasses the
  viral-score gate for that job's clips; safety rules (creator permission, source music, singing along) still apply.
- Every finished clip is also copied to `Clips/<Video title> - <Song> - Clip 03/` next to the exe (clip number counts
  inside its video; `No song` when none was mixed in). It holds the clip, `thumbnail.jpg`, and a `(no song)` version.
  Re-rendering with another song renames the folder. Exports use the same names.

## Command line (Windows)

```bash
cd gta6-clip-engine
.venv/Scripts/clipper setup                       # detects hardware, writes config, downloads face model
.venv/Scripts/clipper streamer add "SomeStreamer" --platform twitch --permission explicitly_authorized
.venv/Scripts/clipper process "https://www.twitch.tv/videos/123456" --streamer SomeStreamer
.venv/Scripts/clipper clips list --min-score 70
.venv/Scripts/clipper clips show 4
.venv/Scripts/clipper clips approve 4
.venv/Scripts/clipper export
```

Music:

```bash
.venv/Scripts/clipper music import song.wav --title "Tweakin" --artist "Moqemae" --confirm-rights --tiktok-sound "Tweakin - Moqemae"
.venv/Scripts/clipper music artist "Moqemae" --good-fit "chase,getaway,crazy" --avoid "tragedy,slur,sexual"
.venv/Scripts/clipper clips music 4 --mode payoff_drop      # or --song <slug>, --off
```

## How a clip is chosen (the "brain")

1. **Signals** over the whole stream, chunked (never loads 4 h into RAM):
   audio excitement vs *this streamer's* baseline (loudness, rise, pitch, laughter-like bursts),
   chat velocity (if a chat log is available), transcript cues ("wait what", "no way", "clip that").
   Detected source music (car radio/BGM) is discounted so it can't fake excitement.
2. **Proposal**: well-separated peaks; clip starts on the first line that hooks, setup capped at 14 s.
3. **Evaluation** against a viral rubric (0-100 each): hook strength, entertainment, payoff, emotional
   intensity, unexpectedness, contextual clarity, shareability, + audio excitement and chat.
   Penalties: outside the 18-45 s sweet spot, slow start, mid-sentence start.
4. **Rejection rules** (dead air, no payoff, repetitive, AI "keep: false"), **dedupe**, then an
   **editor's final cut**: the AI ranks survivors against each other (model modes only).
5. Clips under 70 stay in the library but are never publish-ready; unknown creator permission is never publish-ready.

All weights live in `config/settings.json`.

## Shorts output

1080x1920, face cam top / gameplay bottom (YuNet face detection + static-border search; manual crop
via `clips edit --cam-box x,y,w,h`), or gameplay band with blurred fill when there is no face cam
(never invents one). Jump cuts remove pauses > 0.6 s (payoff and wordless screams are protected),
punch-in zoom on the payoff (face cam layouts), hook card for the first 2.8 s, word-by-word captions
with highlight + emphasis colours inside TikTok/Shorts safe zones, streamer credit, loudness -14 LUFS / -1 dBTP.

## Music module

- **Song analysis** (once, cached by hash): BPM, fitted beat grid + downbeats, key, sections, hook, mood, timed lyrics, bar-aligned promo segments.
- **Source-music detection**: beat periodicity + harmonic stability + tonal continuity in speech gaps; per-stream BGM baseline; "singing along" detection.
- **Placement**: payoff drop (hook downbeat on the payoff), reaction tail, underscore, cold open; bar-aligned; never over source music.
- **Mix**: speech-driven ducking (15 dB, 120/400 ms), 1-4 kHz carve under speech, beat-aligned fades, loudness to target, re-transcription intelligibility check (auto-lowers music or drops it).
- **Rules**: rights confirmation, campaign dates, explicit flag, creator `allows_added_music`, brand avoid-list, rotation limit, campaign mode.
- **Exports**: mixed `video.mp4` + `video_clean_no_song.mp4` + `sound_sheet.txt` (in-app official sound instructions).

## AI is optional (dashboard → Settings)

| mode | needs | what you get | cost / privacy |
|---|---|---|---|
| Rules only (default) | nothing | signal-based picks (audio, surprise words, action, cuts); basic hooks | free, offline, nothing leaves your PC |
| AI with my API key | an Anthropic key, or an OpenAI-compatible key + address | reads the transcript for context and payoff, ranks clips against each other, writes clip-specific hooks/captions | billed to your account; short transcript excerpts are sent to the provider |

The key is saved only in `config/secrets.env` (never in `settings.json`, never shown again, never logged).
**Test connection** checks it. If the key is missing/rejected/rate-limited the engine says so and uses the rules
for that moment, labelling every result with what judged it. The log shows tokens used per run.
There is no Ollama or Claude CLI mode any more; old settings files load as "rules only".

## Video types (New job → Video type)

What the video is decides how its moments are ranked and cut. Pick one, or leave **Auto-detect** (title cues, then how
much of the audio is music; the reason is shown on the Streams tab, where you can change the type and re-rank).

| Type | What ranks highest | Pacing / layout / music |
|---|---|---|
| Streamer stream / VOD | the streamer's reaction, laughter, "clip that", chat spikes | 18-45 s, pauses cut, face cam + game, song volume "featured" |
| Gameplay footage | action spikes, payoff, surprise, sound intensity | 12-35 s, whole frame, trailer-style music |
| Game trailer / cutscene | iconic shots and reveals, hook, payoff, quotable lines | 20-50 s, whole frame, trailer-style music |
| Movie / TV trailer | reveals, star moments, the quotable line (speech and spectacle) | 15-40 s, whole frame, trailer-style music |
| Podcast / interview | quotable, self-contained claims and stories | 30-75 s, pauses trimmed, face shown, subtle music |
| Sports / highlights | the play, emotion, crowd/commentator excitement | 12-35 s, whole frame, "featured" music |

Each type sets: ranking weights, whether speech is required, clip length, jump cuts, layout, music style and volume,
the guidance given to the AI judge, and the tone of rules-only hooks. See `engine/profiles.py`; tweak one type's weights in
`settings.json`: `"profile_overrides": {"movie_trailer": {"payoff": 0.3}}`. CLI: `clipper process URL --type game_trailer`.
Auto-detect is a guess from the title and audio; a trailer with no game cue in its title is ranked as a movie/TV trailer.

## Verified (tests/ — `.venv/Scripts/python -m pytest`)

25 tests on authorized synthetic media (TTS speech + generated songs): scoring/redistribution,
boundaries, jump-cut remapping, caption sync, song BPM/bars (<3 ms) and hook, car radio on/off,
constant BGM, singing-along, placement + rendered drop onset within 50 ms, ducking depth >= 12 dB,
loudness +-1 LU, rotation, brand avoid-list, program restrictions, queue resume, metadata attribution.

## Trends, recipes, end card, source-music removal

- **Trends tab / `clipper trends ...`**: import reference clips (links or files you may analyze, or a YouTube Data API search with
  `YOUTUBE_API_KEY`). Each gets measured layout, cuts per 10 s, hook frame, caption placement, music entry, loudness and
  sound-effect hits. The report ranks patterns by views/day relative to account size, marks **rising** and **saturated** patterns,
  and labels anything under 8 clips *early signal*. Metrics a platform does not give (shares, saves) stay empty.
- **Recipes tab**: a recipe is style settings only (layout, pacing, captions, music timing). `clipper trends recipe`, promote one to
  be the default, or pick one per clip in the clip editor. Enter post-performance (Performance button) and each recipe is scored
  against the baseline. Each recipe carries 2-3 "Our Twist" ideas. Recipes cannot hold another account's assets or songs.
- **Clip editor** (Clips tab, Edit): start/end, layout, face-cam crop, gameplay focus, hook text, captions, title, description, recipe.
- **Discover tab**: Twitch/YouTube search with permission and status, one-click Process. Settings tab: clip length, sensitivity,
  Whisper model, resolution, captions, storage folder, concurrency, priority, end card, source-music removal.
- **End card** (Settings, off by default): 2 s card with title, artist, CTA after clips that have a song. The no-song file never has it.
- **Source-music removal** (Settings, off by default): needs `.venv/Scripts/pip install demucs` (about 2 GB). Keeps only the voice
  stem under our song; the no-song export keeps the original sound. Never used when the music is the joke (singing along).
- **Validate on your footage**: `clipper validate facecam stream.mp4` saves frames with the crop drawn on;
  `clipper validate music stream.mp4 --truth "0-30:music,30-95:none"` scores the detector against ranges you label by ear.

## Not yet verified / not built

- Face-cam detection has not been tested on real webcam footage (none was available here). Run `clipper validate facecam` on a real stream.
- Music detection is calibrated on synthetic audio and the trailer. Run `clipper validate music` with your own labels.
- Demucs removal and the trend engine's analysis of real short-form clips were not run end to end on real footage (Demucs is not installed here).
- Caption words-per-line, animation and emoji use in reference clips are not measured (needs OCR); they are reported as unavailable.
- Recipe fields `sfx` and `pre_payoff_dropout_s` are recorded but not yet applied by the renderer.
- Discovery needs Twitch/YouTube API keys. Kick has no public discovery API configured: add Kick creators manually.
