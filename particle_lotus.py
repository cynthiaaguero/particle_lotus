"""
particle_lotus.py
------------------
Webcam hand-tracking demo: open your hand in front of the camera and a
particle-based lotus flower blooms open; close your hand and it folds
back into a bud.

Sections:
  1. Imports
  2. Camera / display config
  3. Lotus geometry + particle construction (NumPy, vectorized)
  4. Rendering (glow compositing)
  5. Bloom control (set_bloom)
  6. Hand-tracking stubs for you to fill in
  7. Main loop
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# This version of mediapipe only ships the newer Tasks API (mp.solutions,
# the old Hands() class, was removed). The Tasks API needs an explicit
# model file on disk — downloaded once into models/hand_landmarker.task.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")


# ---------------------------------------------------------------------------
# 2. Camera / display configuration
# ---------------------------------------------------------------------------
CAM_WIDTH, CAM_HEIGHT = 960, 720
CENTER = (CAM_WIDTH // 2, CAM_HEIGHT // 2)
BACKDROP_DARKEN = 0.50   # how much of the raw camera feed shows through behind the flower


# ---------------------------------------------------------------------------
# 3. Lotus geometry + particle construction
#
# Each particle gets a "home" position (fully bloomed) and a "closed"
# position (tight bud), both computed once with NumPy. Every frame we
# just interpolate between the two based on the current bloom amount —
# no per-particle Python loop needed.
# ---------------------------------------------------------------------------

# Petals are organized in layers, outer to inner. Outer layers are larger
# and open first; inner layers are smaller and open last, which gives the
# bloom a gentle cascading look instead of popping open all at once.
LAYER_DEFS = [
    dict(n=750, petals=8, length=230.0, half_width=42.0, rot=0.0,        open_start=0.00, open_end=0.55, depth=0),
    dict(n=650, petals=8, length=195.0, half_width=36.0, rot=np.pi / 8,  open_start=0.15, open_end=0.70, depth=1),
    dict(n=550, petals=6, length=155.0, half_width=30.0, rot=np.pi / 6,  open_start=0.30, open_end=0.85, depth=2),
    dict(n=450, petals=5, length=105.0, half_width=24.0, rot=np.pi / 5,  open_start=0.45, open_end=1.00, depth=3),
]
N_CENTER = 350          # warm gold particles at the flower's core
CENTER_RADIUS = 34.0
BUD_SCALE = 0.10        # fraction of full petal size shown while closed

PETAL_PINK = np.array([170.0, 90.0, 225.0], dtype=np.float32)   # BGR soft rose pink
TIP_WHITE = np.array([255.0, 255.0, 255.0], dtype=np.float32)   # BGR white
GOLD_COLOR = np.array([60.0, 180.0, 255.0], dtype=np.float32)   # BGR warm gold


def _smoothstep(x):
    """Smooth 0->1 easing curve used for both bloom opening and glow falloff."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _build_petal_layer(cfg, rng):
    """Vectorized construction of one petal layer's particles."""
    n = cfg["n"]
    t = rng.random(n)                       # 0 = petal base, 1 = petal tip
    lateral = rng.uniform(-1.0, 1.0, n)      # -1..1 across petal width
    petal_idx = rng.integers(0, cfg["petals"], n)
    theta = cfg["rot"] + petal_idx * (2 * np.pi / cfg["petals"])

    width_profile = np.sin(np.pi * t) * cfg["half_width"]   # narrow at base+tip, wide in middle
    local_x = t * cfg["length"]
    local_y = lateral * width_profile

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    home_x = CENTER[0] + local_x * cos_t - local_y * sin_t
    home_y = CENTER[1] + local_x * sin_t + local_y * cos_t

    closed_x = CENTER[0] + (local_x * BUD_SCALE) * cos_t - (local_y * BUD_SCALE) * sin_t
    closed_y = CENTER[1] + (local_x * BUD_SCALE) * sin_t + (local_y * BUD_SCALE) * cos_t

    depth_saturation = 1.0 - cfg["depth"] * 0.12          # outer layers a touch paler
    tip_mix = t ** (1.0 + 0.3 * cfg["depth"])              # how far toward white at the tip
    color = (PETAL_PINK * depth_saturation)[None, :] * (1 - tip_mix)[:, None] \
        + TIP_WHITE[None, :] * tip_mix[:, None]

    open_start = np.full(n, cfg["open_start"], dtype=np.float32)
    open_end = np.full(n, cfg["open_end"], dtype=np.float32)

    return home_x, home_y, closed_x, closed_y, open_start, open_end, color


def _build_center(rng):
    """Small warm-gold particle cluster at the flower's core (always visible)."""
    n = N_CENTER
    r = np.sqrt(rng.random(n)) * CENTER_RADIUS
    ang = rng.uniform(0, 2 * np.pi, n)
    x = CENTER[0] + r * np.cos(ang)
    y = CENTER[1] + r * np.sin(ang)
    color = np.tile(GOLD_COLOR, (n, 1)) * rng.uniform(0.8, 1.15, (n, 1)).astype(np.float32)
    # The core doesn't "open" — closed position == home position.
    open_start = np.zeros(n, dtype=np.float32)
    open_end = np.full(n, 1e-3, dtype=np.float32)
    return x, y, x.copy(), y.copy(), open_start, open_end, color


def _build_lotus():
    rng = np.random.default_rng(42)
    parts = [_build_petal_layer(cfg, rng) for cfg in LAYER_DEFS]
    parts.append(_build_center(rng))

    home_x, home_y, closed_x, closed_y, open_start, open_end, color = (
        np.concatenate(cols) for cols in zip(*parts)
    )
    n_total = home_x.shape[0]

    # Per-particle random phase/frequency for gentle idle shimmer.
    shimmer_rng = np.random.default_rng(7)
    phase_x = shimmer_rng.uniform(0, 2 * np.pi, n_total)
    phase_y = shimmer_rng.uniform(0, 2 * np.pi, n_total)
    phase_b = shimmer_rng.uniform(0, 2 * np.pi, n_total)
    freq_jitter = shimmer_rng.uniform(0.8, 1.3, n_total)

    return dict(
        home_x=home_x.astype(np.float32), home_y=home_y.astype(np.float32),
        closed_x=closed_x.astype(np.float32), closed_y=closed_y.astype(np.float32),
        open_start=open_start, open_end=open_end,
        color=color.astype(np.float32),
        phase_x=phase_x, phase_y=phase_y, phase_b=phase_b, freq_jitter=freq_jitter,
    )


_LOTUS = _build_lotus()
print(f"Particle lotus ready: {_LOTUS['home_x'].shape[0]} particles")


# ---------------------------------------------------------------------------
# 4. Rendering — draw all particles at once, then add a blurred glow pass
# ---------------------------------------------------------------------------
SHIMMER_AMPLITUDE = 1.6     # pixels
SHIMMER_SPEED = 1.5         # radians/sec
BRIGHT_FLICKER = 0.18       # +/- fractional brightness shimmer
GLOW_SIGMA = 7.0            # Gaussian blur radius for the glow pass


def _render_lotus_layer(width, height, t_time):
    """Compute every particle's current position/color with NumPy, scatter
    them onto a dark layer, then blur-and-add that layer onto itself for a
    soft glow."""
    L = _LOTUS
    denom = np.maximum(L["open_end"] - L["open_start"], 1e-6)
    local_t = (_bloom - L["open_start"]) / denom
    ease = _smoothstep(local_t)   # how "open" each particle is right now, 0..1

    cur_x = L["closed_x"] * (1 - ease) + L["home_x"] * ease
    cur_y = L["closed_y"] * (1 - ease) + L["home_y"] * ease

    # Gentle idle shimmer so the flower never looks perfectly static.
    shimmer_x = SHIMMER_AMPLITUDE * np.sin(t_time * SHIMMER_SPEED * L["freq_jitter"] + L["phase_x"])
    shimmer_y = SHIMMER_AMPLITUDE * np.cos(t_time * SHIMMER_SPEED * L["freq_jitter"] + L["phase_y"])
    flicker = 1.0 + BRIGHT_FLICKER * np.sin(t_time * 2.0 * L["freq_jitter"] + L["phase_b"])

    px = np.clip((cur_x + shimmer_x).astype(np.int32), 0, width - 1)
    py = np.clip((cur_y + shimmer_y).astype(np.int32), 0, height - 1)
    lit_color = L["color"] * flicker[:, None]

    layer = np.zeros((height, width, 3), dtype=np.float32)
    np.add.at(layer, (py, px), lit_color)   # scatter all particles in one shot

    glow = cv2.GaussianBlur(layer, (0, 0), sigmaX=GLOW_SIGMA, sigmaY=GLOW_SIGMA)
    layer = layer + glow                    # blur-and-add-back = soft glow

    return np.clip(layer, 0, 255)


def _compose(frame, particle_layer):
    """Dark overlay look: dim the camera feed, then add the glowing particles on top."""
    dark = frame.astype(np.float32) * BACKDROP_DARKEN
    return np.clip(dark + particle_layer, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. Bloom control
# ---------------------------------------------------------------------------
_bloom = 0.0   # 0.0 = closed bud, 1.0 = fully open lotus


def set_bloom(value):
    """Set how open the lotus is, from 0.0 (closed bud) to 1.0 (fully open).

    Each particle layer has its own open_start/open_end range (see
    LAYER_DEFS), so raising this value smoothly cascades the bloom from
    outer petals to inner petals rather than snapping everything open at
    once.
    """
    global _bloom
    _bloom = float(np.clip(value, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 6. Hand-tracking logic — stubs for you to fill in
# ---------------------------------------------------------------------------
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def is_finger_open(landmarks, finger):
    """
    STUB: always returns False.

    `landmarks` is the list of 21 MediaPipe hand landmarks (each with
    .x, .y, .z in normalized [0,1] image coordinates). `finger` is one of
    "thumb", "index", "middle", "ring", "pinky".

    MediaPipe landmark indices per finger (TIP, DIP, PIP, MCP):
        wrist  : 0
        thumb  : 4 (TIP), 3 (IP),  2 (MCP), 1 (CMC)
        index  : 8 (TIP), 7 (DIP), 6 (PIP), 5 (MCP)
        middle : 12 (TIP), 11 (DIP), 10 (PIP), 9 (MCP)
        ring   : 16 (TIP), 15 (DIP), 14 (PIP), 13 (MCP)
        pinky  : 20 (TIP), 19 (DIP), 18 (PIP), 17 (MCP)

    Note: the thumb behaves differently from the other four fingers — it
    folds sideways across the palm rather than curling down, so a
    TIP-above-PIP (y-coordinate) check that works for the other fingers
    won't work for the thumb. It usually needs an x-coordinate /
    across-the-palm comparison instead, and depends on which hand
    (left/right) and handedness is being tracked.
    """

    if finger == "thumb":
        result = landmarks[4].x < landmarks[3].x < landmarks[2].x < landmarks[1].x or landmarks[4].x > landmarks[3].x > landmarks[2].x > landmarks[1].x
        print(result)
        return result
    if finger == "index":
        result = landmarks[8].y < landmarks[7].y < landmarks[6].y < landmarks[5].y
        print(result)
        return result
    if finger == "middle":
        result = landmarks[12].y < landmarks[11].y < landmarks[10].y < landmarks[9].y
        print(result)
        return result
    if finger == "ring":
        result = landmarks[16].y < landmarks[15].y < landmarks[14].y < landmarks[13].y
        print(result)
        return result
    if finger == "pinky":
        result = landmarks[20].y < landmarks[19].y < landmarks[18].y < landmarks[17].y
        print(result)
        return result
    
    return False


def get_target_bloom(open_fingers):
    """
    STUB: always returns 0.0 (closed bud).

    `open_fingers` is a dict like {"thumb": False, "index": True, ...}
    as produced by is_finger_open() above. This should map however many
    fingers are open to a bloom value in [0.0, 1.0] — e.g. a closed fist
    (0 open fingers) -> 0.0, a fully open hand (5 open fingers) -> 1.0,
    with something reasonable for the values in between.
    """
    counter = 0 

    for finger in open_fingers:
        if open_fingers[finger] == True:
            counter+=1
            print(counter)

    return counter/5


# ---------------------------------------------------------------------------
# 7. Main loop
# ---------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,   # frame-by-frame video, not a single IMAGE
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )

    start_time = time.time()
    current_bloom = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)                         # mirror
            frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            landmarks = None
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]   

            if landmarks is not None:
                open_fingers = {name: is_finger_open(landmarks, name) for name in FINGER_NAMES}
            else:
                open_fingers = {name: False for name in FINGER_NAMES}

            target_bloom = get_target_bloom(open_fingers)

            current_bloom += (target_bloom - current_bloom) * .15

            set_bloom(current_bloom)

            t_time = time.time() - start_time
            particle_layer = _render_lotus_layer(CAM_WIDTH, CAM_HEIGHT, t_time)
            output = _compose(frame, particle_layer)

            cv2.imshow("Particle Lotus", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
