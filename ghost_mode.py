"""Ghost / Invisibility Mode demo.

Hold an "OK" pinch gesture (thumb tip close to index tip) with your hand
in front of the camera, and you gradually fade into the background —
exactly like the "GHOST 0% -> 100%" progress-bar effect. Let go of the
gesture and you fade back to normal.

This file is a thin orchestration layer, same spirit as `main.py`: it
does NOT modify `hand_tracker.py`, `background_remover.py`, or
`utils.py` at all. It just wires:

    HandTracker.detect()          -> is the OK/pinch gesture currently held?
    BackgroundRemover.set_alpha() -> how "invisible" should the person be?
    BackgroundRemover.replace_background() -> do the actual compositing

`set_alpha(value)` controls the per-pixel blend strength used inside
`blend()`: alpha=1.0 -> person fully composited over the background
(normal), alpha=0.0 -> the background completely replaces the person's
pixels (fully invisible). Animating that value from 1.0 down to 0.0
while the gesture is held is what produces the "ghost" effect.

Usage:
    python ghost_mode.py
    python ghost_mode.py --camera-index 1 --hold-seconds 2.5

Controls:
    q / Esc  - quit
    r        - re-capture the background plate (step out of frame first)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

import cv2
import numpy as np

from background_remover import (
    BackgroundRemovalConfig,
    BackgroundRemover,
    BackgroundRemoverError,
)
from hand_tracker import Gesture, HandTracker, HandTrackerError, InvalidFrameError

logger = logging.getLogger("ghost_mode")

WINDOW_TITLE = "Ghost / Invisibility Mode"
QUIT_KEYS = frozenset({ord("q"), ord("Q"), 27})
RECAPTURE_KEY = ord("r")

# --------------------------------------------------------------------------- #
# Tunables. Kept here (not in utils.py) because these are demo/UX choices,
# not reusable math -- same boundary the rest of the project already draws.
# --------------------------------------------------------------------------- #
DEFAULT_HOLD_SECONDS = 3.0   # Time holding the gesture to reach 100% ghost.
DEFAULT_RELEASE_SECONDS = 1.5  # Time to fade back to 0% after releasing.
COUNTDOWN_SECONDS = 3          # "step out of frame" countdown at startup.

# WHY this threshold exists: replace_background() always runs the mask
# through the segmentation model whenever a background is saved, even at
# alpha=1.0 (fully visible). The mask is a per-frame probabilistic output,
# not a perfectly stable one -- it wobbles slightly frame to frame,
# especially at edges (hair, clothing boundaries), which reads as a faint
# flicker/"jitter" over the WHOLE image even while resting fully visible.
# Below this threshold we skip compositing entirely and show the plain
# camera frame, so you only ever see mask-based flicker while an actual
# fade is in progress, never while at rest.
GHOST_ACTIVE_EPSILON = 0.01

BAR_WIDTH_PX = 260
BAR_HEIGHT_PX = 18
BAR_ORIGIN = (20, 40)
TEXT_COLOR = (255, 255, 255)
BAR_FILL_COLOR = (0, 140, 255)
BAR_BG_COLOR = (60, 60, 60)

# Standard 21-point MediaPipe Hands topology (fixed; see hand_tracker.py's
# NUM_HAND_LANDMARKS docstring). Defined here, not imported, because
# hand_tracker.py deliberately doesn't expose a public "draw" function --
# it only draws its own debug overlay internally. We draw our own here so
# we can render it AFTER background compositing (see _run), independent of
# the fade, instead of before it.
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
SKELETON_JOINT_RADIUS_PX = 4
SKELETON_BONE_THICKNESS_PX = 2


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture-driven ghost mode.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--hold-seconds", type=float, default=DEFAULT_HOLD_SECONDS,
        help="Seconds the gesture must be held to go from 0%% to 100%% ghost.",
    )
    parser.add_argument(
        "--release-seconds", type=float, default=DEFAULT_RELEASE_SECONDS,
        help="Seconds to fade back to fully visible once the gesture stops.",
    )
    return parser.parse_args(argv)


def _open_capture(index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {index}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def _capture_clean_background(
    capture: cv2.VideoCapture, remover: BackgroundRemover
) -> None:
    """Shows a countdown so the user can step out, then saves the plate."""
    start = time.monotonic()
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        frame = cv2.flip(frame, 1)

        remaining = COUNTDOWN_SECONDS - (time.monotonic() - start)
        if remaining <= 0:
            remover.save_background(frame)
            logger.info("Background plate captured.")
            return

        message = f"Step out of frame... capturing in {remaining:.1f}s"
        cv2.putText(frame, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW_TITLE, frame)
        if (cv2.waitKey(1) & 0xFF) in QUIT_KEYS:
            raise KeyboardInterrupt


def _is_ghost_gesture(hands) -> bool:
    """True if any detected hand is holding the OK/pinch gesture."""
    return any(hand.gesture == Gesture.PINCH for hand in hands)


def _draw_hand_skeleton(frame: np.ndarray, hands) -> None:
    """Draws the hand skeleton directly, always at full opacity.

    WHY drawn manually instead of via HandTracker's built-in overlay:
    HandTracker(draw_landmarks=True) bakes the skeleton into the pixels of
    the frame it's given. If that frame is then faded by the background
    remover, the skeleton fades along with it. Drawing it ourselves, after
    compositing, on top of the final output, means the hands/bones stay
    fully solid no matter how "ghosted" the rest of the body is.
    """
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


def _draw_hud(frame: np.ndarray, ghost_progress: float) -> None:
    percent = int(round(ghost_progress * 100))
    label = f"GHOST {percent}%"
    cv2.putText(frame, label, (BAR_ORIGIN[0], BAR_ORIGIN[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

    x, y = BAR_ORIGIN
    cv2.rectangle(frame, (x, y), (x + BAR_WIDTH_PX, y + BAR_HEIGHT_PX),
                  BAR_BG_COLOR, thickness=-1)
    fill_width = int(BAR_WIDTH_PX * ghost_progress)
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + BAR_HEIGHT_PX),
                      BAR_FILL_COLOR, thickness=-1)
    cv2.rectangle(frame, (x, y), (x + BAR_WIDTH_PX, y + BAR_HEIGHT_PX),
                  TEXT_COLOR, thickness=1)


def _run(args: argparse.Namespace) -> int:
    capture = _open_capture(args.camera_index, args.width, args.height)
    # Larger blur/feather kernels than the defaults (7/9) trade a slightly
    # softer edge for a noticeably more stable one -- this is the other
    # half of the jitter fix: it reduces how much the mask wobbles frame to
    # frame *while a fade is actually in progress* (the epsilon guard below
    # only stops flicker at rest, not during an active transition).
    config = BackgroundRemovalConfig(
        gaussian_blur_kernel_size=11,
        edge_feather_kernel_size=15,
    )
    ghost_progress = 0.0
    previous_time = time.monotonic()

    # WHY draw_landmarks=False here: we draw the skeleton ourselves, after
    # compositing (see _draw_hand_skeleton), so it never fades with the body.
    try:
        with HandTracker(max_num_hands=2, draw_landmarks=False) as tracker, \
             BackgroundRemover(config=config) as remover:

            # WHY needed: unlike HandTracker (which initializes its model in
            # __init__), BackgroundRemover.__enter__ does NOT call
            # initialize() for you -- segment()/replace_background() raise
            # NotInitializedError until you call it explicitly. Skipping
            # this line is exactly why the effect silently no-ops (caught
            # below as a BackgroundRemoverError and swallowed by falling
            # back to the raw frame).
            remover.initialize()

            _capture_clean_background(capture, remover)

            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame = cv2.flip(frame, 1)

                # WHY hand detection runs on the RAW frame: the composited
                # frame's opacity depends on the gesture itself. Detecting
                # on the faded output would make the hand harder to see for
                # MediaPipe right when ghost_progress is high, causing
                # detection to drop, which would be read as "gesture
                # released" and yank progress back down -- a fight-itself
                # flicker loop. Detecting on the untouched frame keeps
                # gesture tracking stable regardless of how transparent the
                # person currently is.
                try:
                    hand_result = tracker.detect(frame)
                except InvalidFrameError:
                    logger.exception("Bad frame; skipping.")
                    continue

                now = time.monotonic()
                dt = now - previous_time
                previous_time = now

                gesture_held = _is_ghost_gesture(hand_result.hands)
                if gesture_held:
                    ghost_progress += dt / max(args.hold_seconds, 1e-6)
                else:
                    ghost_progress -= dt / max(args.release_seconds, 1e-6)
                ghost_progress = min(1.0, max(0.0, ghost_progress))

                if ghost_progress > GHOST_ACTIVE_EPSILON:
                    try:
                        remover.set_alpha(1.0 - ghost_progress)
                        result = remover.replace_background(frame)
                        output_frame = result.processed_frame
                    except BackgroundRemoverError:
                        logger.exception("Background removal failed; using raw frame.")
                        output_frame = frame.copy()
                else:
                    # At rest (not ghosting): skip the segmentation
                    # pipeline entirely so you see the real, un-composited
                    # camera image -- no mask-driven flicker at all.
                    output_frame = frame.copy()

                # Skeleton drawn LAST, on the final output, so it's always
                # fully solid -- never faded, per your request.
                _draw_hand_skeleton(output_frame, hand_result.hands)
                _draw_hud(output_frame, ghost_progress)
                cv2.imshow(WINDOW_TITLE, output_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in QUIT_KEYS:
                    break
                if key == RECAPTURE_KEY:
                    _capture_clean_background(capture, remover)
                    previous_time = time.monotonic()

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