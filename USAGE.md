# How to use the GTA 6 Clip Engine

This guide covers the dashboard and the command line. Run everything from the `gta6-clip-engine` folder.

---

## 1. One-time setup

1. **FFmpeg** must be installed and on PATH. Check it with `ffmpeg -version`. To install: `winget install Gyan.FFmpeg`.
2. **Ollama** (free local AI). Install from https://ollama.com, then start it and pull the model:
   ```bash
   ollama serve
   ollama pull qwen2.5:7b
   ```
   (Skip this if you use Claude mode, see section 6.)
3. **Run the setup check.** It detects your hardware, writes the config, creates the database and downloads the face-detection model:
   ```bash
   .venv/Scripts/clipper setup
   ```
   Read the **"Needs your action"** list it prints. It is also saved to `logs/setup_report.md`.
4. **Optional discovery keys.** To search for streams, add keys to `config/secrets.env`:
   ```
   TWITCH_CLIENT_ID=...
   TWITCH_CLIENT_SECRET=...
   YOUTUBE_API_KEY=...
   ```
   Without them you can still process any URL or file you give the engine. Discovery just won't search.

---

## 2. Start the app (easiest way)

Double-click **`GTA6Clipper.exe`** in the `gta6-clip-engine` folder.

- A small window starts the engine. Your browser opens the dashboard at http://127.0.0.1:8765. It only works on this computer.
- Keep the window open while you work. Close it to stop the engine.
- If the window says it can't find `.venv`, keep the exe inside the `gta6-clip-engine` folder.
- To rebuild the exe after moving the folder:
  ```bash
  .venv/Scripts/pyinstaller --onefile --console --name GTA6Clipper --distpath . --workpath storage/build --specpath storage/build scripts/launcher.py
  ```

### Dashboard pages

| Page | What you do there |
|---|---|
| **Dashboard** | See stats, the stream being processed with live step progress, the live log, and the AI brain switch. |
| **New job** | Paste a Twitch, YouTube or Kick link, or a local video file path. Set the creator's permission. Start. |
| **Streams** | See every stream, its status and per-step timings. Retry failed ones. |
| **Clips** | Preview clips, read the score, sub-scores, hook, "why it was picked" and music decision. Approve or reject. Use the no-song version. Export approved clips. |
| **Music / Creators** | Review the song library analysis. Set each creator's permissions. |

---

## 3. Process a stream (command line)

```bash
.venv/Scripts/clipper process "https://www.twitch.tv/videos/123456" --streamer SomeStreamer --permission explicitly_authorized
```

Useful options:

| Option | Meaning |
|---|---|
| `--streamer NAME` | Creator name. Defaults to the channel name. |
| `--permission STATUS` | Creator permission. See section 4. |
| `--chat FILE` | Chat log (Twitch/YouTube/CSV) to add the chat signal. |
| `--focus "funny reactions"` | Bias the picks toward a type of moment. |
| `--max-clips N` | Clips to produce per stream. |
| `--workers N` | How many streams run at once. Default is 1, which is safest on most PCs. |

You can pass several URLs or files at once. Long streams take a while. Progress shows in the dashboard.

To queue now and run later:

```bash
.venv/Scripts/clipper queue add "https://www.twitch.tv/videos/123456" --streamer SomeStreamer
.venv/Scripts/clipper queue run
.venv/Scripts/clipper queue list
```

`queue run` resumes any job that was interrupted. Failed jobs retry up to 3 times.

To rerun analysis for a stream, reusing the cached transcript:

```bash
.venv/Scripts/clipper reanalyze 4 --focus "shocking discoveries"
```

---

## 4. Creator permissions (important)

Every creator needs a permission status. Clips from a creator with the status `unknown` are **never publish-ready**.

```bash
.venv/Scripts/clipper streamer add "SomeStreamer" --platform twitch --url "https://twitch.tv/somestreamer" --permission explicitly_authorized
.venv/Scripts/clipper streamer permission "SomeStreamer" explicitly_authorized --monetize --attribution "Clip: {streamer} - {url}"
.venv/Scripts/clipper streamer list
```

Statuses:

| Status | Publish-ready? |
|---|---|
| `explicitly_authorized` | Yes |
| `official_clipping_program` | Yes |
| `permission_requested` | No |
| `unknown` | No |
| `not_authorized` | No |

Other flags: `--added-music` (allows your song on their clips), `--promotion`, `--rules "program rules or link"`. Use `streamer favorite` and `streamer exclude` to flag creators.

Only process content you have permission to clip. Do not remove watermarks or bypass access controls.

---

## 5. Review and export clips

```bash
.venv/Scripts/clipper streams                          # all streams and their status
.venv/Scripts/clipper clips candidates 4               # every moment considered, including rejects and why
.venv/Scripts/clipper clips list --min-score 70        # clips, best first
.venv/Scripts/clipper clips show 12                    # full detail: scores, reasoning, hooks, metadata
.venv/Scripts/clipper clips open 12                    # play the rendered clip
.venv/Scripts/clipper clips approve 12 13 15
.venv/Scripts/clipper clips reject 14
.venv/Scripts/clipper export                           # export all approved clips
```

Clips scoring under **70** stay in the library but are never publish-ready.

### Fix a clip

```bash
.venv/Scripts/clipper clips edit 12 --start 310.5 --end 352.0     # change the boundaries (seconds in the source)
.venv/Scripts/clipper clips edit 12 --hook 1                      # use hook 2 (hooks are 0, 1, 2)
.venv/Scripts/clipper clips edit 12 --hook 1 --hook-text "He did NOT see that coming"
.venv/Scripts/clipper clips edit 12 --layout gameplay_only        # split | fullcam | gameplay_only
.venv/Scripts/clipper clips edit 12 --cam-box 0,0,640,360         # manual webcam crop (source pixels)
.venv/Scripts/clipper clips edit 12 --highlight "#FFD400" --words-per-line 3 --animation pop
.venv/Scripts/clipper clips edit 12 --new-hooks                   # regenerate hooks and metadata
```

Every `clips edit` re-renders the clip automatically.

Export is only for the clips you approve. Output goes to the folder shown in the `exported ->` lines.

---

## 6. Choose the AI mode

Set with `clipper setup --ai-mode <mode>`, or edit `config/settings.json`.

| Mode | Needs | Quality | Notes |
|---|---|---|---|
| `local` | Ollama running with `qwen2.5:7b` | Good | Free, offline. Default on this PC. |
| `claude` | Claude Code CLI installed and signed in | Best judgement | Uses your subscription. Transcript excerpts go to Anthropic. |
| `heuristic` | Nothing | Signal-based, generic hooks | Free, offline. Fallback. |

The engine never silently switches to a paid API. If the model is unavailable, it uses `heuristic` and labels every result.

Check your hardware and what the engine picked:

```bash
.venv/Scripts/clipper doctor
.venv/Scripts/clipper stats
```

---

## 7. Music (your songs in clips)

Only add songs you own or have rights to. The import requires `--confirm-rights`.

**Import and analyze a song** (analysis runs once and is cached):

```bash
.venv/Scripts/clipper music import "C:\path\to\song.wav" --title "Tweakin" --artist "Moqemae" --confirm-rights --priority 4 --tiktok-sound "Tweakin - Moqemae"
.venv/Scripts/clipper music list
.venv/Scripts/clipper music show tweakin
```

WAV is preferred. MP3, FLAC and AAC also work. Stems are optional. Timed lyrics are transcribed by default.

**Correct the analysis** if BPM, the hook or a section is wrong. Your corrections override the automatic values:

```bash
.venv/Scripts/clipper music correct tweakin --bpm 142 --hook-start 40.6
```

**Set the artist brand profile** (the AI reads this before choosing music):

```bash
.venv/Scripts/clipper music artist "Moqemae" --good-fit "chase,getaway,crazy" --avoid "tragedy,slur,sexual,mocking" --sound-tag "♪ {artist} – {title}" --cta "Out now everywhere,Pre-save in bio" --volume balanced
```

**Control music on a clip:**

```bash
.venv/Scripts/clipper clips music 12                              # let the engine choose
.venv/Scripts/clipper clips music 12 --song tweakin --mode payoff_drop
.venv/Scripts/clipper clips music 12 --mode reaction_tail
.venv/Scripts/clipper clips music 12 --off                        # export without music
```

Placement modes: `payoff_drop` (song hits on the payoff), `reaction_tail` (song fades in at the end), `underscore` (quiet bed under the clip), `cold_open` (hook at the start).

Music is never placed over music already in the source (car radio, stream BGM). If it can't be placed cleanly, the clip stays available with no song and is marked "Not music-eligible."

Each music export gives you:
- `video.mp4`: mixed with your song.
- `video_clean_no_song.mp4`: no song, for adding the official sound inside the app.
- `sound_sheet.txt`: which sound to add, the start offset and the volume levels.

Rotation limits (no song over-used across a batch) and campaign mode (one song pushed across a batch) are built in. Campaign mode is not exposed as a CLI flag yet; check `engine/music/match.py` for how it is switched on.

---

## 8. Maintenance

```bash
.venv/Scripts/clipper logs                # recent processing log
.venv/Scripts/clipper perf                # timings per step
.venv/Scripts/clipper transcript 4 --fmt srt --out "stream4.srt"
.venv/Scripts/clipper cleanup             # delete temp files (transcripts are kept)
.venv/Scripts/clipper cleanup --audio     # also delete cached WAVs of finished streams
```

Settings to know, in `config/settings.json`:
- `max_concurrent_jobs`: how many streams run at once. Keep this at 1 unless you have a lot of GPU memory.
- `transcription.model`: the Whisper model used for transcripts.
- `video.encoder`: `h264_nvenc` uses your NVIDIA GPU. Change to `libx264` if you have no NVIDIA GPU.

---

## 9. Running the tests

```bash
.venv/Scripts/python -m pytest
```

This runs 25 tests on synthetic audio. A pass shows the logic works. It does not prove the output is good on real footage.

---

## 10. Known limits

- **Face-cam detection** has not been tested on real webcam footage. Check the layout on your first real clips and fix it with `clips edit --cam-box` if needed.
- **Music detection** is calibrated on synthetic audio. Check the music decisions on real GTA radio and BGM.
- **Discovery** needs the Twitch and YouTube keys. Kick has no discovery.
- **Not built yet:** source-music removal, end card, the full clip editor in the dashboard, performance tracking, trend engine.
- **Local AI** needs Ollama running. Without it, the engine uses `heuristic` mode.
- Only publish clips from creators with a publishable permission status. The engine flags risk but does not guarantee platform monetization.

---

## Quick start checklist

1. Install FFmpeg. Install and start Ollama. Pull `qwen2.5:7b`.
2. `clipper setup`. Read "Needs your action".
3. `clipper streamer add ... --permission ...` for each creator.
4. Double-click `GTA6Clipper.exe`, or run `clipper process <url> --streamer <name>`.
5. `clipper clips list --min-score 70`, then `clips show` on the good ones.
6. `clips approve <ids>`, then `clipper export`.
7. Optional: `clipper music import` your songs and set `clipper music artist`.
