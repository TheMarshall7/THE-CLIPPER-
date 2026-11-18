"""Webcam / layout detection with OpenCV (PRD §10).

Returns one of:
- "split":         webcam overlay found -> face cam on top, gameplay below
- "fullcam":       the footage is mostly the camera (face is large)
- "gameplay_only": no stable face -> never invent a face cam

The webcam rectangle is located by looking for border edges that stay put
across sampled frames (a static overlay frame) around the detected face.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class Layout:
    layout: str
    cam_box: Optional[list[int]]       # x, y, w, h in source pixels
    face_box: Optional[list[int]]
    confidence: float
    frame_size: list[int]
    method: str
    manual: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


YUNET_PATH = Path(__file__).resolve().parents[2] / "models" / "vision" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_LOCAL = threading.local()      # OpenCV's detector keeps per-call state: one instance per thread (renders run in parallel)


def ensure_face_model() -> bool:
    """Download the small (230 KB) face model once if it is missing. True when it is available afterwards."""
    if YUNET_PATH.exists():
        return True
    try:
        import requests

        r = requests.get(YUNET_URL, timeout=60)
        r.raise_for_status()
        YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = YUNET_PATH.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(YUNET_PATH)
        return True
    except Exception:
        return False


def _detector():
    """YuNet DNN face detector (OpenCV model zoo). OpenCV 5 dropped Haar cascades."""
    det = getattr(_LOCAL, "det", None)
    if det is None:
        if not ensure_face_model():
            raise RuntimeError(f"Face model missing: {YUNET_PATH}. Run `clipper setup` (downloads it from {YUNET_URL}).")
        cv2.setLogLevel(2) if hasattr(cv2, "setLogLevel") else None
        det = _LOCAL.det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.75, 0.3, 50)
    return det


def sample_frames(path: Path, start: float, end: float, n: int = 10) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        for t in np.linspace(start, end, n + 2)[1:-1]:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
            ok, fr = cap.read()
            if ok and fr is not None:
                frames.append(fr)
    finally:
        cap.release()
    return frames


def detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Faces as (x, y, w, h) in frame pixels. Runs at <=1280 px wide for speed."""
    h, w = frame.shape[:2]
    scale = min(1.0, 1280 / w)
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
    det = _detector()
    det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = det.detect(small)
    if faces is None:
        return []
    min_side = max(12, h // 40)
    out = []
    for f in faces:
        x, y, fw, fh = (float(v) / scale for v in f[:4])
        if min(fw, fh) >= min_side:
            out.append((int(x), int(y), int(fw), int(fh)))
    return out


def _stable_face(frames: list[np.ndarray]) -> tuple[Optional[list[int]], float]:
    boxes = []
    for fr in frames:
        f = detect_faces(fr)
        if f:
            boxes.append(max(f, key=lambda b: b[2] * b[3]))
    if not boxes:
        return None, 0.0
    arr = np.array(boxes, dtype=float)
    med = np.median(arr, axis=0)
    # keep detections near the median position (webcam doesn't jump around)
    tol = max(med[2], med[3]) * 0.8
    close = arr[np.hypot(arr[:, 0] - med[0], arr[:, 1] - med[1]) < tol]
    support = len(close) / len(frames)
    box = np.median(close, axis=0) if len(close) else med
    return [int(v) for v in box], support


def _border_search(frames: list[np.ndarray], face: list[int]) -> tuple[list[int], float]:
    H, W = frames[0].shape[:2]
    fx, fy, fw, fh = face
    # search region: generous area around the face
    x0, x1 = max(0, fx - 4 * fw), min(W, fx + 5 * fw)
    y0, y1 = max(0, fy - 3 * fh), min(H, fy + 4 * fh)
    gx = np.zeros((y1 - y0, x1 - x0), np.float32)
    gy = np.zeros_like(gx)
    for fr in frames:
        g = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx += np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
        gy += np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    gx /= len(frames)
    gy /= len(frames)
    cols = gx.mean(axis=0)
    rows = gy.mean(axis=1)
    cfx, cfy = fx - x0, fy - y0

    def best(profile: np.ndarray, lo: int, hi: int) -> tuple[Optional[int], float]:
        lo, hi = max(0, lo), min(len(profile), hi)
        if hi - lo < 3:
            return None, 0.0
        seg = profile[lo:hi]
        k = int(np.argmax(seg))
        strength = float(seg[k] / (np.median(profile) + 1e-6))
        return (lo + k, strength) if strength > 2.2 else (None, strength)

    left, sl = best(cols, 0, cfx - fw // 4)
    right, sr = best(cols, cfx + fw + fw // 4, len(cols))
    top, st = best(rows, 0, cfy - fh // 4)
    bottom, sb = best(rows, cfy + fh + fh // 4, len(rows))
    found = sum(v is not None for v in (left, right, top, bottom))
    # Fill missing sides with a 16:9-ish box around the face; frame edges count as borders.
    face_cx, face_cy = fx + fw / 2, fy + fh / 2
    bw = fw * 3.4
    bh = bw * 9 / 16
    L = x0 + left if left is not None else face_cx - bw / 2
    R = x0 + right if right is not None else face_cx + bw / 2
    T = y0 + top if top is not None else face_cy - bh * 0.45
    B = y0 + bottom if bottom is not None else face_cy + bh * 0.55
    L, T = max(0, L), max(0, T)
    R, B = min(W, R), min(H, B)
    conf = 0.4 + 0.15 * found
    return [int(L), int(T), int(R - L), int(B - T)], min(conf, 0.95)


def detect_layout(path: Path, start: float, end: float) -> Layout:
    frames = sample_frames(path, start, end)
    if not frames:
        raise RuntimeError("Could not decode frames for webcam detection.")
    H, W = frames[0].shape[:2]
    face, support = _stable_face(frames)
    if face is None or support < 0.4:
        return Layout("gameplay_only", None, face, round(1 - support, 2), [W, H], "yunet:no_stable_face")
    area = face[2] * face[3] / (W * H)
    if area > 0.035:  # big face -> the frame is the camera itself
        return Layout("fullcam", None, face, round(support, 2), [W, H], "yunet:large_face")
    cam, conf = _border_search(frames, face)
    return Layout("split", cam, face, round(conf * support, 2), [W, H], "yunet+static_border")


def too_blurry_cam(found: dict, cfg, sw: int, sh: int) -> Optional[float]:
    """The zoom the camera panel would need, if that is past the sharpness limit; otherwise None."""
    from engine.video.compose import cam_zoom_needed

    if found.get("layout") != "split" or not found.get("cam_box"):
        return None
    z = cam_zoom_needed(found, cfg, sw, sh)
    return z if z > cfg.max_cam_zoom * 1.4 else None


def force_layout(path: Path, start: float, end: float, style: str, found: dict) -> dict:
    """Apply the user's layout choice on top of what was auto-detected.

    "double": two stacked views - the streamer's camera close-up on top, the whole scene below. Used even when the
              footage is one big camera (no separate cam overlay): the top panel is then a close-up around the face.
    "single": one full-frame view, no cam panel.
    anything else ("auto"): keep the detection."""
    if style == "single":
        if found.get("layout") == "gameplay_only":
            return found
        W, H = found.get("frame_size") or [0, 0]
        return {"layout": "gameplay_only", "cam_box": None, "face_box": found.get("face_box"), "confidence": 1.0,
                "frame_size": [W, H], "method": "user:single", "manual": False}
    if style != "double" or (found.get("layout") == "split" and found.get("cam_box")):
        return found
    frames = sample_frames(path, start, end)
    if not frames:
        return found
    H, W = frames[0].shape[:2]
    face = found.get("face_box")
    if not face:
        face, support = _stable_face(frames)
        if face is None or support < 0.25:
            return found                       # no face to centre a camera view on: leave the detection as it was
    fx, fy, fw, fh = face
    bw = fw * 3.4                               # head, hair and shoulders around the face
    bh = bw * 9 / 16
    L = max(0, min(fx + fw / 2 - bw / 2, W - bw))
    T = max(0, min(fy + fh / 2 - bh * 0.45, H - bh))
    return {"layout": "split", "cam_box": [int(L), int(T), int(min(bw, W)), int(min(bh, H))], "face_box": [int(v) for v in face],
            "confidence": 0.9, "frame_size": [W, H], "method": "user:double", "manual": False}


def track_face(path: Path, src_start: float, dur: float, keep: list[tuple[float, float]], ref_face: list[int],
               max_points: int = 24) -> list[list[float]]:
    """Follow the streamer's face through the clip: [[t_out, cx, cy], ...] in source pixels on the OUTPUT timeline
    (after jump cuts). Sampled sparsely (<= max_points), smoothed, gaps filled by holding the last position."""
    keep = keep or [(0.0, dur)]
    out_len = sum(b - a for a, b in keep)
    step = max(0.6, out_len / max_points)
    cap = cv2.VideoCapture(str(path))
    pts: list[list[float]] = []
    last = (ref_face[0] + ref_face[2] / 2, ref_face[1] + ref_face[3] / 2)
    try:
        t_out = 0.0
        for a, b in keep:
            t = a
            while t < b:
                cap.set(cv2.CAP_PROP_POS_MSEC, (src_start + t) * 1000)
                ok, fr = cap.read()
                if ok and fr is not None:
                    faces = detect_faces(fr)
                    if faces:
                        # the face nearest the last known position (ignores faces of people on screen in the game)
                        f = min(faces, key=lambda b_: (b_[0] + b_[2] / 2 - last[0]) ** 2 + (b_[1] + b_[3] / 2 - last[1]) ** 2)
                        if ((f[0] + f[2] / 2 - last[0]) ** 2 + (f[1] + f[3] / 2 - last[1]) ** 2) ** 0.5 < max(ref_face[2], ref_face[3]) * 4:
                            last = (f[0] + f[2] / 2, f[1] + f[3] / 2)
                pts.append([round(t_out + (t - a), 3), last[0], last[1]])
                t += step
            t_out += b - a
    finally:
        cap.release()
    if len(pts) < 2:
        return []
    arr = np.array(pts, dtype=float)
    for k in (1, 2):                                   # median of 3 kills single-frame glitches, then a light smooth
        v = arr[:, k].copy()
        m = v.copy()
        for i in range(1, len(v) - 1):
            m[i] = np.median(v[i - 1:i + 2])
        s = m.copy()
        for i in range(1, len(m)):
            s[i] = 0.6 * m[i] + 0.4 * s[i - 1]
        arr[:, k] = s
    arr[-1, 0] = max(arr[-1, 0], out_len)              # hold the last position to the end
    return [[round(float(t), 3), round(float(x), 1), round(float(y), 1)] for t, x, y in arr]
