"""Combo Mode demo: Ghost Mode + Hand FX running together.

Both gesture systems are live at the same time, driven by ONE shared
HandTracker.detect() call per frame:

    Gesture.PINCH (OK sign)  -> gradual fade into the background, exactly
                                  like `ghost_mode.py` on its own.
    Gesture.POINTING          -> pulsing energy ball off the fingertip,
    Gesture.OPEN_PALM          -> flickering lightning from the palm,
    two hands held close       -> shared "combined energy" orb,
                                  exactly like `hand_fx.py` on its own.

WHY this file exists instead of just running both scripts side by side:
each of `ghost_mode.py` and `hand_fx.py` opens its own camera capture and
calls `HandTracker.detect()` independently. Two processes both grabbing
the same camera index either fails outright or fights over frames. This
file is the actual merge the other two scripts' docstrings pointed at:
"combining them later is just importing this module's draw functions
into ghost_mode.py's loop" -- so that's exactly what happens below, one
capture, one `detect()` call per frame, both effects drawn from the same
`hand_result.hands`.

WHY draw order matters here (background composite -> glow overlay ->
skeleton -> HUD): the energy ball / lightning glow is meant to look like
it's ON the ghosted person, so it's drawn on top of whatever
`replace_background()` produced (fully visible, ghosted, or in between).
The skeleton is drawn last, same as `ghost_mode.py` alone, so it stays
fully solid no matter how transparent the body is.

This file does NOT modify `hand_tracker.py`, `background_remover.py`,
`ghost_mode.py`, or `hand_fx.py` -- it only imports their public pieces
(and the handful of underscore-prefixed helpers those two modules'
docstrings already call out as the intended reuse surface) and wires
them into a single loop.

Usage:
    python combo_mode.py
    python combo_mode.py --camera-index 1 --no-skeleton

Controls:
    q / Esc  - quit
    r        - re-capture the background plate (step out of frame first)
    s        - toggle hand skeleton overlay
"""

from __future__ import annotations

import argparse
import logging
import random
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
from hand_tracker import HandTracker, HandTrackerError, InvalidFrameError

from ghost_mode import (
    COUNTDOWN_SECONDS,
    DEFAULT_HOLD_SECONDS,
    DEFAULT_RELEASE_SECONDS,
    GHOST_ACTIVE_EPSILON,
    _capture_clean_background,
    _draw_hand_skeleton,
    _draw_hud,
    _is_ghost_gesture,
)
from hand_fx import (
    TOGGLE_SKELETON_KEY,
    _apply_glow,
    _draw_between_hands_orb,
    _draw_energy_ball,
    _draw_lightning,
    _smoothstep,
    _update_between_hands_charge,
    _update_hand_charges,
)
from hand_tracker import Gesture

logger = logging.getLogger("combo_mode")

WINDOW_TITLE = "Combo Mode (Ghost + Hand FX)"
QUIT_KEYS = frozenset({ord("q"), ord("Q"), 27})
RECAPTURE_KEY = ord("r")


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ghost Mode + Hand FX running together.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--hold-seconds", type=float, default=DEFAULT_HOLD_SECONDS,
        help="Seconds the pinch gesture must be held to go from 0%% to 100%% ghost.",
    )
    parser.add_argument(
        "--release-seconds", type=float, default=DEFAULT_RELEASE_SECONDS,
        help="Seconds to fade back to fully visible once the pinch stops.",
    )
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


def _run(args: argparse.Namespace) -> int:
    capture = _open_capture(args.camera_index, args.width, args.height)
    # Same edge-stability tuning ghost_mode.py uses on its own -- larger
    # blur/feather kernels than the BackgroundRemover defaults, to keep the
    # segmentation mask from flickering while a fade is in progress.
    config = BackgroundRemovalConfig(
        gaussian_blur_kernel_size=11,
        edge_feather_kernel_size=15,
    )

    show_skeleton = not args.no_skeleton
    rng = random.Random()
    ghost_progress = 0.0
    start_time = time.monotonic()
    previous_time = start_time

    # hand_fx's per-hand and two-hand "combined energy" charge state --
    # identical shape/usage to what hand_fx.py keeps on its own.
    charge_state: dict = {}
    between_hands_state = {"charge": 0.0}

    # WHY draw_landmarks=False: same reason as both source files -- the
    # skeleton is drawn manually, after every other effect, so it never
    # fades with the body and never gets buried under the glow layer.
    try:
        with HandTracker(max_num_hands=2, draw_landmarks=False) as tracker, \
             BackgroundRemover(config=config) as remover:

            # WHY needed: BackgroundRemover.__enter__ does NOT call
            # initialize() for you -- see ghost_mode.py's note on this.
            remover.initialize()

            _capture_clean_background(capture, remover)

            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame = cv2.flip(frame, 1)

                # WHY detection runs on the RAW frame, once, shared by both
                # effect systems: ghost_mode.py's note applies here too
                # (detecting on a partially-faded frame would make gesture
                # tracking unstable right when it matters most), and a
                # single detect() call means the two effect systems never
                # see two different hand reads for the same instant.
                try:
                    hand_result = tracker.detect(frame)
                except InvalidFrameError:
                    logger.exception("Bad frame; skipping.")
                    continue

                now = time.monotonic()
                dt = now - previous_time
                previous_time = now
                t = now - start_time

                # --- Ghost fade (pinch) -------------------------------- #
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
                    # At rest: skip segmentation entirely, same
                    # flicker-avoidance as ghost_mode.py alone.
                    output_frame = frame.copy()

                # --- Hand FX (energy ball / lightning / combined orb) -- #
                # Drawn onto their own glow layer and composited on TOP of
                # the (possibly ghosted) output_frame above, so the energy
                # effects read as sitting on the person regardless of how
                # transparent they currently are.
                glow_layer = np.zeros_like(frame)

                hand_charges = _update_hand_charges(charge_state, hand_result.hands, dt)
                for hand, (effect_gesture, charge) in zip(hand_result.hands, hand_charges):
                    if charge <= 0.0 or effect_gesture is None:
                        continue
                    eased_charge = _smoothstep(charge)
                    if effect_gesture == Gesture.POINTING:
                        _draw_energy_ball(glow_layer, hand, t, eased_charge)
                    elif effect_gesture == Gesture.OPEN_PALM:
                        _draw_lightning(glow_layer, hand, rng, t, eased_charge)

                between_charge = _update_between_hands_charge(
                    between_hands_state, hand_result.hands, dt,
                )
                if between_charge > 0.0:
                    _draw_between_hands_orb(
                        glow_layer, hand_result.hands, t, _smoothstep(between_charge),
                    )

                output_frame = _apply_glow(output_frame, glow_layer)

                # --- Overlay UI (always full opacity, drawn last) ------ #
                if show_skeleton:
                    _draw_hand_skeleton(output_frame, hand_result.hands)
                _draw_hud(output_frame, ghost_progress)

                cv2.imshow(WINDOW_TITLE, output_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in QUIT_KEYS:
                    break
                if key == RECAPTURE_KEY:
                    _capture_clean_background(capture, remover)
                    previous_time = time.monotonic()
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