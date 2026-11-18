"""Synthesize a royalty-free test song with a known structure (for repeatable tests).

128 BPM (or --bpm), 4/4: intro 4 bars, verse 8, chorus 8 (the hook), verse 8, chorus 8, outro 4.
Ground truth (bars, sections) is written next to the WAV as JSON so tests can
check beat/section detection against it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def tone(f, dur, sr=SR, kind="saw"):
    t = np.arange(int(dur * sr)) / sr
    if kind == "sine":
        return np.sin(2 * np.pi * f * t)
    return 2 * ((f * t) % 1.0) - 1.0


def env(n, a=0.005, r=0.2, sr=SR):
    e = np.ones(n)
    ka, kr = int(a * sr), int(r * sr)
    e[:ka] = np.linspace(0, 1, ka) if ka else 1
    if kr and kr < n:
        e[-kr:] *= np.linspace(1, 0, kr)
    return e


def make(bpm: float = 128.0, key_root: float = 110.0, seed: int = 1) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    bar = 4 * beat
    plan = [("intro", 4, 0.35), ("verse", 8, 0.55), ("chorus", 8, 1.0), ("verse", 8, 0.55), ("chorus", 8, 1.0), ("outro", 4, 0.3)]
    total_bars = sum(b for _, b, _ in plan)
    n = int(total_bars * bar * SR) + SR
    mix = np.zeros(n)
    prog = [0, 5, 3, 7]  # semitone offsets: i - VI - iv - VII-ish minor feel
    sections, b0 = [], 0
    for name, nb, energy in plan:
        sections.append({"label": name, "start": round(b0 * bar, 3), "end": round((b0 + nb) * bar, 3), "is_hook": name == "chorus"})
        for b in range(b0, b0 + nb):
            t_bar = b * bar
            root = key_root * 2 ** (prog[b % 4] / 12)
            for k in range(4):                                 # kick on every beat
                s = int((t_bar + k * beat) * SR)
                kd = 0.25
                tt = np.arange(int(kd * SR)) / SR
                kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12)
                mix[s:s + kick.size] += 0.9 * kick * (0.6 + 0.4 * energy)
                if k in (1, 3) and name != "intro":            # snare
                    sn = rng.normal(0, 1, int(0.15 * SR)) * np.exp(-np.arange(int(0.15 * SR)) / SR * 25)
                    mix[s:s + sn.size] += 0.35 * sn * energy
                for h in (0, 0.5):                             # hats
                    hs = int((t_bar + (k + h) * beat) * SR)
                    hat = rng.normal(0, 1, int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / SR * 120)
                    mix[hs:hs + hat.size] += 0.12 * hat * energy
            bs = int(t_bar * SR)                               # bass, one note per beat
            for k in range(4):
                note = tone(root / 2, beat * 0.9) * env(int(beat * 0.9 * SR), r=0.05)
                s = bs + int(k * beat * SR)
                mix[s:s + note.size] += 0.25 * note * (0.5 + 0.5 * energy)
            chord = sum(tone(root * 2 ** (i / 12), bar, kind="sine") for i in (0, 3, 7)) / 3   # pad
            mix[bs:bs + chord.size] += 0.18 * chord * env(chord.size, a=0.05, r=0.3) * energy
            if name == "chorus":                               # lead hook melody
                mel = [0, 3, 7, 10, 7, 3, 5, 3]
                for k, m in enumerate(mel):
                    f = root * 4 * 2 ** (m / 12)
                    d = bar / 8
                    nt = tone(f, d * 0.95, kind="sine") * env(int(d * 0.95 * SR), r=0.04)
                    s = bs + int(k * d * SR)
                    mix[s:s + nt.size] += 0.22 * nt
        b0 += nb
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.89
    stereo = np.stack([mix, mix], axis=1)
    truth = {"bpm": bpm, "bar_s": bar, "bars": [round(i * bar, 4) for i in range(total_bars + 1)], "sections": sections}
    return stereo, truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--bpm", type=float, default=128.0)
    ap.add_argument("--root", type=float, default=110.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    audio, truth = make(a.bpm, a.root, a.seed)
    sf.write(a.out, audio, SR, subtype="PCM_16")
    a.out.with_suffix(".truth.json").write_text(json.dumps(truth, indent=1))
    print(a.out)
