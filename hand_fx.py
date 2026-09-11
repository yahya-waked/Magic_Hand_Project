"""Hand FX demo: gesture-triggered "action movie" style overlays.

    Gesture.POINTING  ( ☝️ / 👈 )  -> pulsing energy ball that gradually
                                       charges up just off the fingertip
                                       of the extended index finger, in
                                       whichever direction it's pointing.
    Gesture.OPEN_PALM ( ✋ )        -> a small charge-orb in the palm with
                                       flickering lightning bolts that grow
                                       out from it as the palm opens.
    Two hands held close together   -> a shared "combined energy" orb that
    ( 🙏 -style, proximity-based )     builds up between them (detected by
                                       hand distance, not a specific
                                       gesture value).

All three effects ease in smoothly while their trigger is held and ease
back out when it stops, instead of popping on/off instantly.

Like `ghost_mode.py`, this is a thin orchestration script: it does NOT
modify `hand_tracker.py`, `background_remover.py`, or `utils.py`. It only
consumes HandTracker's public `HandData` contract (landmarks, center,
bounding_box, gesture) and draws on top with OpenCV.

WHY a separate file from ghost_mode.py: these are two independent
features (one drives background compositing, this one draws a VFX
overlay) triggered by different gestures. Keeping them separate means
each can be run, tested, and tuned on its own. Merging them later is just
importing this module's draw functions into ghost_mode.py's loop.

WHY the "glow layer" approach: drawing shapes directly onto the camera
frame looks flat -- an ordinary colored circle or line has none of the
"glowing energy" look. Instead every effect is drawn onto a separate
black canvas (the glow layer), which is then Gaussian-blurred and added
back onto the frame with `cv2.add` (which saturates at 255 instead of
wrapping/overflowing). Blur + saturating-add is what produces the
soft bloom look with a bright core, cheaply, in real time.

Usage:
    python hand_fx.py
    python hand_fx.py --camera-index 1 --no-skeleton
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from hand_tracker import (
    Gesture,
    HandData,
    HandTracker,
    HandTrackerError,
    InvalidFrameError,
)

logger = logging.getLogger("hand_fx")

WINDOW_TITLE = "Hand FX"
QUIT_KEYS = frozenset({ord("q"), ord("Q"), 27})
TOGGLE_SKELETON_KEY = ord("s")

# --------------------------------------------------------------------------- #
# Standard 21-point MediaPipe Hands topology + landmark index for the
# fingertip we anchor the energy ball to. Same fixed topology used in
# ghost_mode.py -- not imported from hand_tracker.py because that module
# deliberately doesn't expose a public "draw" function.
# --------------------------------------------------------------------------- #
INDEX_FINGER_TIP = 8
_WRIST = 0
_INDEX_MCP = 5
_MIDDLE_FINGER_TIP = 12
_MIDDLE_MCP = 9
_RING_FINGER_TIP = 16
_RING_MCP = 13
_PINKY_FINGER_TIP = 20
_PINKY_MCP = 17

_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
)
SKELETON_JOINT_COLOR = (0, 215, 255)
SKELETON_BONE_COLOR = (60, 200, 60)
SKELETON_JOINT_RADIUS_PX = 3
SKELETON_BONE_THICKNESS_PX = 1

# --------------------------------------------------------------------------- #
# Effect tunables. All sizes are RATIOS of the hand's bounding-box diagonal,
# not fixed pixel counts -- so effects scale naturally as a hand moves
# closer to / farther from the camera, instead of staying a fixed size.
# --------------------------------------------------------------------------- #
ENERGY_BALL_RADIUS_RATIO = 0.55
ENERGY_BALL_PULSE_SPEED = 6.0          # radians/sec for the breathing pulse
ENERGY_BALL_PULSE_DEPTH = 0.12         # +/- fraction of radius
ENERGY_BALL_ORBIT_PARTICLES = 5
ENERGY_BALL_ORBIT_SPEED = 3.5          # radians/sec
ENERGY_BALL_ORBIT_RADIUS_RATIO = 1.35  # relative to ball radius
ENERGY_BALL_OFFSET_RATIO = 0.42        # how far off the fingertip the ball floats, relative to hand scale
ENERGY_BALL_GROW_SECONDS = 1.4         # gradual charge-up, like an action-movie power build
ENERGY_BALL_DECAY_SECONDS = 1.0        # fades back down a bit quicker than it charges

LIGHTNING_BOLT_COUNT = 6
LIGHTNING_LENGTH_RATIO = 1.4
LIGHTNING_JITTER_RATIO = 0.22
LIGHTNING_RECURSION_DEPTH = 4
LIGHTNING_CORE_THICKNESS_PX = 2
LIGHTNING_GLOW_THICKNESS_PX = 5
LIGHTNING_GROW_SECONDS = 1.1           # bolts grow out gradually as the palm opens
LIGHTNING_DECAY_SECONDS = 0.85         # and shrink back gradually as it closes

PALM_ORB_RADIUS_RATIO = 0.22           # small charge-orb the bolts radiate from
PALM_ORB_PULSE_SPEED = 5.0
PALM_ORB_PULSE_DEPTH = 0.15

BETWEEN_HANDS_PROXIMITY_RATIO = 1.2    # trigger distance, relative to avg hand scale
BETWEEN_HANDS_GROW_SECONDS = 1.8       # a two-handed "combined power" orb -- slower, weightier build
BETWEEN_HANDS_DECAY_SECONDS = 1.2
BETWEEN_HANDS_RADIUS_RATIO = 0.8       # relative to the two hands' average scale
BETWEEN_HANDS_PULSE_SPEED = 4.0
BETWEEN_HANDS_PULSE_DEPTH = 0.15
BETWEEN_HANDS_ORBIT_PARTICLES = 6
BETWEEN_HANDS_ORBIT_SPEED = 2.2
BETWEEN_HANDS_ORBIT_RADIUS_RATIO = 1.3

GLOW_BLUR_SIGMA = 9.0
GLOW_INTENSITY = 1.1

# All colors are BGR (OpenCV convention). The energy-ball and lightning
# palettes are kept close to each other -- a warm soft-white core fading
# into a muted blue-violet -- so both effects read as "the same kind of
# energy" instead of clashing, oversaturated neon colors.
COLOR_WHITE = (255, 255, 255)
COLOR_ENERGY_CORE = (255, 248, 225)     # warm, soft near-white
COLOR_ENERGY_MID = (250, 205, 150)      # soft powder blue
COLOR_ENERGY_OUTER = (235, 150, 95)     # muted blue-violet
COLOR_LIGHTNING = (250, 195, 130)       # sibling of ENERGY_MID, slightly cooler
COLOR_LIGHTNING_CORE = (255, 250, 235)  # warm near-white, softer than pure white


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture-triggered hand VFX.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--no-skeleton", action="store_true",
        help="Hide the hand skeleton overlay, show effects only.",
    )
    return parser.parse_args(argv)


def _open_capture(index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {index}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def _draw_hand_skeleton(frame: np.ndarray, hands: List[HandData]) -> None:
    for hand in hands:
        for start_index, end_index in _HAND_CONNECTIONS:
            start = hand.landmarks[start_index]
            end = hand.landmarks[end_index]
            cv2.line(
                frame, (start.pixel_x, start.pixel_y),
                (end.pixel_x, end.pixel_y),
                SKELETON_BONE_COLOR, SKELETON_BONE_THICKNESS_PX,
            )
        for landmark in hand.landmarks:
            cv2.circle(
                frame, (landmark.pixel_x, landmark.pixel_y),
                SKELETON_JOINT_RADIUS_PX, SKELETON_JOINT_COLOR, thickness=-1,
            )


def _hand_scale_px(hand: HandData) -> float:
    """Diagonal of the hand's bounding box, in pixels.

    Used as the single "how big is this hand right now" reference so every
    effect scales consistently with distance from the camera.
    """
    box = hand.bounding_box
    return math.hypot(box.width, box.height) or 1.0


# --------------------------------------------------------------------------- #
# `hand_tracker.py`'s own Gesture.POINTING heuristic (see its
# `_get_finger_states`) calls a finger "extended" by comparing the
# fingertip's screen *y*-coordinate to its PIP joint's -- i.e. "is the tip
# higher up on screen". That works well for a hand pointing roughly
# upward (☝️) but silently misses a hand pointing sideways or downward
# (👈): the tip may not be "higher" even though the finger is clearly
# extended. Since hand_fx.py must not modify hand_tracker.py, this adds a
# rotation-invariant fallback check here instead, used as an OR alongside
# `hand.gesture` -- never a replacement for it -- so it only adds
# detections, never removes any.
# --------------------------------------------------------------------------- #
_POINTING_EXTENSION_RATIO = 1.6  # tip-to-wrist vs mcp-to-wrist distance ratio


def _finger_extended_from_wrist(hand: HandData, tip_index: int, mcp_index: int) -> bool:
    """Rotation-invariant "is this finger extended", using radial distance
    from the wrist instead of a fixed screen axis.
    """
    wrist = hand.landmarks[_WRIST]
    tip = hand.landmarks[tip_index]
    mcp = hand.landmarks[mcp_index]
    tip_dist = math.hypot(tip.pixel_x - wrist.pixel_x, tip.pixel_y - wrist.pixel_y)
    mcp_dist = math.hypot(mcp.pixel_x - wrist.pixel_x, mcp.pixel_y - wrist.pixel_y) or 1.0
    return tip_dist / mcp_dist > _POINTING_EXTENSION_RATIO


def _looks_like_pointing(hand: HandData) -> bool:
    """"Index finger extended, other three curled", independent of which
    way the hand is turned on screen. See the note above `hand.gesture`
    for why this is needed alongside it.
    """
    index_out = _finger_extended_from_wrist(hand, INDEX_FINGER_TIP, _INDEX_MCP)
    others_curled = not any(
        _finger_extended_from_wrist(hand, tip, mcp)
        for tip, mcp in (
            (_MIDDLE_FINGER_TIP, _MIDDLE_MCP),
            (_RING_FINGER_TIP, _RING_MCP),
            (_PINKY_FINGER_TIP, _PINKY_MCP),
        )
    )
    return index_out and others_curled


def _effective_gesture(hand: HandData) -> Optional[Gesture]:
    """The gesture that should drive an hand_fx effect for this hand.

    Trusts `hand.gesture` for OPEN_PALM as-is (no orientation bias there
    worth working around). For POINTING, accepts either `hand.gesture`
    *or* the rotation-invariant `_looks_like_pointing` fallback, so
    pointing sideways or downward triggers the energy ball just as
    reliably as pointing straight up.
    """
    if hand.gesture == Gesture.OPEN_PALM:
        return Gesture.OPEN_PALM
    if hand.gesture == Gesture.POINTING or _looks_like_pointing(hand):
        return Gesture.POINTING
    return None


def _draw_energy_ball(glow_layer: np.ndarray, hand: HandData, t: float, charge: float) -> None:
    """Draws a pulsing energy orb that charges up just off the index fingertip.

    `charge` is 0..1 (see `_update_hand_charges`): 0 means the gesture was just
    recognized (no ball yet), 1 means fully charged (normal pulsing size).
    The ball is anchored a bit *beyond* the fingertip -- extended along the
    finger's own direction -- so it visually floats above the finger
    instead of sitting glued on top of it.
    """
    tip = hand.landmarks[INDEX_FINGER_TIP]
    dip = hand.landmarks[INDEX_FINGER_TIP - 1]  # index DIP joint, one below the tip
    scale = _hand_scale_px(hand)

    dir_x, dir_y = tip.pixel_x - dip.pixel_x, tip.pixel_y - dip.pixel_y
    dir_len = math.hypot(dir_x, dir_y) or 1.0
    offset = scale * ENERGY_BALL_OFFSET_RATIO
    center = (
        int(tip.pixel_x + dir_x / dir_len * offset),
        int(tip.pixel_y + dir_y / dir_len * offset),
    )

    pulse = 1.0 + ENERGY_BALL_PULSE_DEPTH * math.sin(t * ENERGY_BALL_PULSE_SPEED)
    radius = scale * ENERGY_BALL_RADIUS_RATIO * pulse * charge
    if radius < 1:
        return

    # Three nested circles (outer -> core) is cheap and, once blurred,
    # reads as a smooth radial glow rather than flat rings.
    cv2.circle(glow_layer, center, int(radius), COLOR_ENERGY_OUTER, thickness=-1)
    cv2.circle(glow_layer, center, int(radius * 0.6), COLOR_ENERGY_MID, thickness=-1)
    cv2.circle(glow_layer, center, int(radius * 0.28), COLOR_ENERGY_CORE, thickness=-1)

    # Small particles orbiting the ball on an elliptical path (foreshortened
    # circle) for a bit of motion/energy instead of a static sphere. They
    # fade in with the same charge factor so they don't pop in early.
    orbit_radius = radius * ENERGY_BALL_ORBIT_RADIUS_RATIO
    for i in range(ENERGY_BALL_ORBIT_PARTICLES):
        angle = t * ENERGY_BALL_ORBIT_SPEED + i * (2 * math.pi / ENERGY_BALL_ORBIT_PARTICLES)
        px = int(center[0] + orbit_radius * math.cos(angle))
        py = int(center[1] + orbit_radius * 0.45 * math.sin(angle))
        cv2.circle(glow_layer, (px, py), max(2, int(radius * 0.08)),
                   COLOR_ENERGY_CORE, thickness=-1)


def _midpoint_displace(
    start: Tuple[float, float], end: Tuple[float, float],
    jitter: float, depth: int, rng: random.Random,
) -> List[Tuple[int, int]]:
    """Recursively displaces the midpoint perpendicular to the segment.

    This is the standard "fractal lightning" technique: split a straight
    line in half, nudge the midpoint sideways by a random amount, then
    recurse on the two new halves with a smaller jitter each level. The
    result is a jagged bolt instead of a straight, obviously-fake line.
    """
    if depth <= 0:
        return [start, end]

    mid_x = (start[0] + end[0]) / 2.0
    mid_y = (start[1] + end[1]) / 2.0

    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    normal_x, normal_y = -dy / length, dx / length

    offset = rng.uniform(-jitter, jitter)
    mid = (mid_x + normal_x * offset, mid_y + normal_y * offset)

    left = _midpoint_displace(start, mid, jitter * 0.5, depth - 1, rng)
    right = _midpoint_displace(mid, end, jitter * 0.5, depth - 1, rng)
    return left[:-1] + right


def _draw_palm_orb(glow_layer: np.ndarray, center: Tuple[int, int], scale: float,
                    t: float, charge: float) -> None:
    """Draws a small pulsing charge-orb at the palm the bolts appear to radiate from."""
    pulse = 1.0 + PALM_ORB_PULSE_DEPTH * math.sin(t * PALM_ORB_PULSE_SPEED)
    radius = scale * PALM_ORB_RADIUS_RATIO * pulse * charge
    if radius < 1:
        return
    cv2.circle(glow_layer, center, int(radius), COLOR_ENERGY_OUTER, thickness=-1)
    cv2.circle(glow_layer, center, int(radius * 0.55), COLOR_ENERGY_MID, thickness=-1)
    cv2.circle(glow_layer, center, int(radius * 0.22), COLOR_ENERGY_CORE, thickness=-1)


def _draw_lightning(glow_layer: np.ndarray, hand: HandData, rng: random.Random,
                     t: float, charge: float) -> None:
    """Draws several jagged bolts radiating outward from a small palm orb.

    WHY regenerated every frame with no throttling: at ~30fps, redrawing
    fresh random bolts each frame IS the flicker -- it's what makes
    lightning read as electric rather than as a static drawn shape. The
    recursion is shallow (depth 4) so this stays cheap.

    `charge` (0..1, see `_update_hand_charges`) scales both the palm orb and
    the bolt length/jitter, so the whole effect grows in from nothing
    instead of appearing at full size the instant the gesture is seen.
    """
    center = hand.center
    scale = _hand_scale_px(hand)

    _draw_palm_orb(glow_layer, center, scale, t, charge)

    length = scale * LIGHTNING_LENGTH_RATIO * charge
    jitter = scale * LIGHTNING_JITTER_RATIO * charge
    if length < 1:
        return

    for i in range(LIGHTNING_BOLT_COUNT):
        angle = (2 * math.pi / LIGHTNING_BOLT_COUNT) * i + rng.uniform(-0.3, 0.3)
        bolt_length = length * rng.uniform(0.75, 1.25)
        end = (
            center[0] + bolt_length * math.cos(angle),
            center[1] + bolt_length * math.sin(angle),
        )
        points = _midpoint_displace(
            (float(center[0]), float(center[1])), end,
            jitter, LIGHTNING_RECURSION_DEPTH, rng,
        )
        int_points = [(int(x), int(y)) for x, y in points]

        # Wider, dimmer pass first (feeds the blur -> glow), thin bright
        # core on top (survives the blur as a crisp bright center line).
        for j in range(len(int_points) - 1):
            cv2.line(glow_layer, int_points[j], int_points[j + 1],
                     COLOR_LIGHTNING, LIGHTNING_GLOW_THICKNESS_PX)
        for j in range(len(int_points) - 1):
            cv2.line(glow_layer, int_points[j], int_points[j + 1],
                     COLOR_LIGHTNING_CORE, LIGHTNING_CORE_THICKNESS_PX)


_EFFECT_RATE_SECONDS = {
    Gesture.POINTING: (ENERGY_BALL_GROW_SECONDS, ENERGY_BALL_DECAY_SECONDS),
    Gesture.OPEN_PALM: (LIGHTNING_GROW_SECONDS, LIGHTNING_DECAY_SECONDS),
}


def _update_hand_charges(
    charge_state: dict, hands: List[HandData], dt: float,
) -> List[Tuple[Optional["Gesture"], float]]:
    """Advances each hand's charge level toward 0 (off) or 1 (fully formed).

    Returns a list, aligned with `hands`, of (effect_gesture, charge) pairs.
    `effect_gesture` is the gesture whose effect should currently be drawn
    for that hand (POINTING / OPEN_PALM), or None if nothing should be
    drawn -- it can lag one frame behind `hand.gesture` while a previous
    effect is still shrinking out.

    This is a simple per-frame lerp toward a target (1.0 while the
    triggering gesture is held, 0.0 otherwise) at a fixed rate, rather than
    a fixed-duration timer -- so opening/closing the hand quickly grows or
    shrinks the effect at the same *speed* every time, matching how a real
    energy charge/discharge would look, instead of always taking the same
    wall-clock time regardless of how briefly the gesture was held.
    """
    results: List[Tuple[Optional[Gesture], float]] = []
    seen_slots = set()

    for i, hand in enumerate(hands):
        seen_slots.add(i)
        slot = charge_state.setdefault(i, {"gesture": None, "charge": 0.0})
        raw = _effective_gesture(hand)

        if slot["gesture"] is None and raw is not None:
            slot["gesture"] = raw  # nothing was showing -- start charging the new gesture

        target = 1.0 if (slot["gesture"] is not None and raw == slot["gesture"]) else 0.0

        if slot["gesture"] is not None:
            grow_seconds, decay_seconds = _EFFECT_RATE_SECONDS[slot["gesture"]]
            rate_seconds = grow_seconds if target > slot["charge"] else decay_seconds
            step = dt / max(rate_seconds, 1e-3)
            if slot["charge"] < target:
                slot["charge"] = min(target, slot["charge"] + step)
            else:
                slot["charge"] = max(target, slot["charge"] - step)

            # Fully discharged and the gesture that made it isn't held
            # anymore -- free the slot so a *different* gesture can start
            # charging immediately instead of waiting behind this one.
            if slot["charge"] <= 0.0 and raw != slot["gesture"]:
                slot["gesture"] = raw
                if raw is not None:
                    target = 1.0  # picked up a new gesture the same frame

        results.append((slot["gesture"], slot["charge"]))

    # Drop slots for hands that left the frame entirely, so a hand that
    # re-enters later starts its charge-up from zero again.
    for slot in list(charge_state.keys()):
        if slot not in seen_slots:
            del charge_state[slot]

    return results


def _update_between_hands_charge(state: dict, hands: List[HandData], dt: float) -> float:
    """Grows/shrinks a "combined energy" charge shared by both hands.

    Mirrors `_update_hand_charges`'s lerp-toward-target approach, but the
    trigger is proximity, not a specific gesture: charge eases toward 1
    while exactly two hands are held close together (praying-hands style)
    and back toward 0 once they separate. Proximity-based on purpose --
    hand_tracker.py's Gesture enum may not have a dedicated value for
    "palms pressed together", but `HandData.center` is enough to detect it
    without needing one.
    """
    target = 0.0
    if len(hands) == 2:
        a, b = hands
        dist = math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
        avg_scale = (_hand_scale_px(a) + _hand_scale_px(b)) / 2.0
        if dist < avg_scale * BETWEEN_HANDS_PROXIMITY_RATIO:
            target = 1.0

    rate_seconds = (
        BETWEEN_HANDS_GROW_SECONDS if target > state["charge"] else BETWEEN_HANDS_DECAY_SECONDS
    )
    step = dt / max(rate_seconds, 1e-3)
    if state["charge"] < target:
        state["charge"] = min(target, state["charge"] + step)
    else:
        state["charge"] = max(target, state["charge"] - step)
    return state["charge"]


def _draw_between_hands_orb(glow_layer: np.ndarray, hands: List[HandData],
                             t: float, charge: float) -> None:
    """Draws a growing "combined power" orb at the midpoint between two hands
    brought close together, the way charging energy between cupped/pressed
    palms reads in action movies.
    """
    if len(hands) != 2 or charge <= 0.0:
        return
    a, b = hands
    center = (
        int((a.center[0] + b.center[0]) / 2),
        int((a.center[1] + b.center[1]) / 2),
    )
    scale = (_hand_scale_px(a) + _hand_scale_px(b)) / 2.0

    pulse = 1.0 + BETWEEN_HANDS_PULSE_DEPTH * math.sin(t * BETWEEN_HANDS_PULSE_SPEED)
    radius = scale * BETWEEN_HANDS_RADIUS_RATIO * pulse * charge
    if radius < 1:
        return

    cv2.circle(glow_layer, center, int(radius), COLOR_ENERGY_OUTER, thickness=-1)
    cv2.circle(glow_layer, center, int(radius * 0.6), COLOR_ENERGY_MID, thickness=-1)
    cv2.circle(glow_layer, center, int(radius * 0.28), COLOR_ENERGY_CORE, thickness=-1)

    orbit_radius = radius * BETWEEN_HANDS_ORBIT_RADIUS_RATIO
    for i in range(BETWEEN_HANDS_ORBIT_PARTICLES):
        angle = t * BETWEEN_HANDS_ORBIT_SPEED + i * (2 * math.pi / BETWEEN_HANDS_ORBIT_PARTICLES)
        px = int(center[0] + orbit_radius * math.cos(angle))
        py = int(center[1] + orbit_radius * 0.6 * math.sin(angle))
        cv2.circle(glow_layer, (px, py), max(2, int(radius * 0.07)),
                   COLOR_ENERGY_CORE, thickness=-1)


def _smoothstep(x: float) -> float:
    """Eases a 0..1 value with zero slope at both ends (S-curve).

    Applied on top of the linear grow/decay above so the motion still
    covers the full 0..1 range in the same short, snappy duration, but
    doesn't start/stop instantaneously -- it eases in from the same
    (fast) speed rather than moving strictly linearly.
    """
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _apply_glow(frame: np.ndarray, glow_layer: np.ndarray) -> np.ndarray:
    """Blurs the glow layer and adds it back onto the frame with saturation."""
    blurred = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=GLOW_BLUR_SIGMA)
    bloom = cv2.addWeighted(blurred, GLOW_INTENSITY, glow_layer, 1.0, 0)
    # cv2.add (not '+') saturates at 255 per channel instead of wrapping
    # around -- essential, or bright overlaps turn into dark noise.
    return cv2.add(frame, bloom)


def _run(args: argparse.Namespace) -> int:
    capture = _open_capture(args.camera_index, args.width, args.height)
    rng = random.Random()
    show_skeleton = not args.no_skeleton
    start_time = time.monotonic()
    charge_state: dict = {}
    between_hands_state = {"charge": 0.0}
    prev_t = 0.0

    try:
        with HandTracker(max_num_hands=2, draw_landmarks=False) as tracker:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame = cv2.flip(frame, 1)

                try:
                    hand_result = tracker.detect(frame)
                except InvalidFrameError:
                    logger.exception("Bad frame; skipping.")
                    continue

                t = time.monotonic() - start_time
                dt = t - prev_t
                prev_t = t
                glow_layer = np.zeros_like(frame)

                # (effect_gesture, charge) per hand -- charge eases toward 1
                # while the gesture is held and back toward 0 once it's
                # released, so the effect grows AND shrinks smoothly instead
                # of appearing/vanishing instantly.
                hand_charges = _update_hand_charges(charge_state, hand_result.hands, dt)

                for hand, (effect_gesture, charge) in zip(hand_result.hands, hand_charges):
                    if charge <= 0.0 or effect_gesture is None:
                        continue
                    eased_charge = _smoothstep(charge)
                    if effect_gesture == Gesture.POINTING:
                        _draw_energy_ball(glow_layer, hand, t, eased_charge)
                    elif effect_gesture == Gesture.OPEN_PALM:
                        _draw_lightning(glow_layer, hand, rng, t, eased_charge)

                # Two-hand "combined energy" orb -- independent of the
                # per-hand gestures above, so it can build up alongside or
                # instead of them when both hands are brought together.
                between_charge = _update_between_hands_charge(
                    between_hands_state, hand_result.hands, dt,
                )
                if between_charge > 0.0:
                    _draw_between_hands_orb(
                        glow_layer, hand_result.hands, t, _smoothstep(between_charge),
                    )

                output_frame = _apply_glow(frame, glow_layer)

                if show_skeleton:
                    _draw_hand_skeleton(output_frame, hand_result.hands)

                cv2.imshow(WINDOW_TITLE, output_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in QUIT_KEYS:
                    break
                if key == TOGGLE_SKELETON_KEY:
                    show_skeleton = not show_skeleton

    except (HandTrackerError, RuntimeError):
        logger.exception("Fatal error.")
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())