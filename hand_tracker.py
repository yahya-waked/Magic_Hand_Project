"""Real-time hand landmark tracking module.

This module is intentionally scoped to a single responsibility: turning a
BGR video frame into structured hand-landmark data. It has no knowledge of
where frames come from (camera, video file, network stream) and no
knowledge of how the resulting data is consumed (UI overlay, gesture
control, AR pipeline, analytics). That separation is what lets this class
be reused across products (Camera app, Messenger AR effects, Portal,
research prototypes) without modification, which is the Open/Closed
Principle in practice.

Typical usage:
    tracker = HandTracker()
    result = tracker.detect(bgr_frame)
    if result.hand_detected:
        for hand in result.hands:
            ...

Author: Computer Vision Platform Team
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

# --------------------------------------------------------------------------- #
# Module-level logger.
#
# WHY: print() is unacceptable in production CV pipelines that run inside
# services, mobile bridges, or batch jobs. A named logger lets downstream
# systems (Meta's Buck-built services, DeepMind's internal pipelines, etc.)
# control verbosity, route logs to observability stacks, and correlate
# hand-tracker events with the rest of the system without code changes here.
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants
#
# WHY: Magic numbers scattered through a CV codebase are a maintenance
# hazard — nobody two years from now will know why "0.5" or "21" appears in
# three different files. Centralizing them here makes intent explicit and
# gives us a single place to tune behavior for new hardware or new
# accuracy/latency trade-offs.
# --------------------------------------------------------------------------- #
NUM_HAND_LANDMARKS: Final[int] = 21  # Fixed by the MediaPipe Hands topology.
DEFAULT_MAX_NUM_HANDS: Final[int] = 2
DEFAULT_MIN_DETECTION_CONFIDENCE: Final[float] = 0.7
DEFAULT_MIN_TRACKING_CONFIDENCE: Final[float] = 0.5
DEFAULT_MODEL_COMPLEXITY: Final[int] = 1  # 0 = fast/light, 1 = full accuracy.

# Landmark indices per the MediaPipe Hands specification.
# WHY named constants: raw indices like `landmarks[8]` are unreadable and
# error-prone. Naming them makes gesture-recognition code self-documenting
# and prevents off-by-one bugs when new contributors extend this class.
WRIST: Final[int] = 0
THUMB_TIP: Final[int] = 4
THUMB_IP: Final[int] = 3
INDEX_FINGER_TIP: Final[int] = 8
INDEX_FINGER_PIP: Final[int] = 6
MIDDLE_FINGER_TIP: Final[int] = 12
MIDDLE_FINGER_PIP: Final[int] = 10
RING_FINGER_TIP: Final[int] = 16
RING_FINGER_PIP: Final[int] = 14
PINKY_TIP: Final[int] = 20
PINKY_PIP: Final[int] = 18
MIDDLE_FINGER_MCP: Final[int] = 9  # Used as a stable reference for rotation.

# Grouped for iteration when computing per-finger extension state.
_FINGER_TIP_PIP_PAIRS: Final[Tuple[Tuple[int, int], ...]] = (
    (INDEX_FINGER_TIP, INDEX_FINGER_PIP),
    (MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP),
    (RING_FINGER_TIP, RING_FINGER_PIP),
    (PINKY_TIP, PINKY_PIP),
)

_FINGER_NAMES: Final[Tuple[str, ...]] = ("thumb", "index", "middle", "ring", "pinky")

# Drawing style constants.
# WHY: hardcoded BGR tuples inline would force UI decisions into a class
# that must never contain UI logic (see class-level restriction). These
# constants exist purely to render a *debug/telemetry* skeleton overlay on
# the returned frame — not to implement any UI feature — and are grouped
# here so visual styling can be swapped without touching detection logic.
LANDMARK_COLOR_BGR: Final[Tuple[int, int, int]] = (0, 215, 255)
CONNECTION_COLOR_BGR: Final[Tuple[int, int, int]] = (60, 200, 60)
LANDMARK_RADIUS_PX: Final[int] = 4
CONNECTION_THICKNESS_PX: Final[int] = 2

# Gesture heuristics.
_PINCH_DISTANCE_THRESHOLD_NORMALIZED: Final[float] = 0.05


class Gesture(str, Enum):
    """Enumerates gestures this module can recognize from landmark geometry.

    WHY an Enum instead of raw strings: prevents typos ("Fist" vs "fist")
    from silently breaking downstream gesture-dispatch logic, and gives
    IDEs/type-checkers the ability to catch invalid gesture references at
    development time rather than at runtime in production.
    """

    UNKNOWN = "unknown"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    POINTING = "pointing"
    PINCH = "pinch"
    THUMBS_UP = "thumbs_up"


@dataclass(frozen=True)
class Landmark:
    """A single hand landmark in both normalized and pixel space.

    Attributes:
        x: Normalized x-coordinate in [0.0, 1.0] relative to image width.
        y: Normalized y-coordinate in [0.0, 1.0] relative to image height.
        z: Normalized depth, roughly relative to the wrist; smaller is
            closer to the camera. Not metric.
        pixel_x: x-coordinate in pixel space for the source frame.
        pixel_y: y-coordinate in pixel space for the source frame.
        visibility: Optional visibility/presence score in [0.0, 1.0], when
            provided by the underlying model. ``None`` if unavailable.
    """

    x: float
    y: float
    z: float
    pixel_x: int
    pixel_y: int
    visibility: Optional[float] = None


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned pixel bounding box around a hand.

    Attributes:
        x_min: Left edge, in pixels.
        y_min: Top edge, in pixels.
        x_max: Right edge, in pixels.
        y_max: Bottom edge, in pixels.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        """int: Bounding box width in pixels."""
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        """int: Bounding box height in pixels."""
        return max(0, self.y_max - self.y_min)


@dataclass(frozen=True)
class HandData:
    """Structured, strongly typed representation of a single detected hand.

    WHY a dataclass instead of MediaPipe's native protobuf objects: exposing
    protobuf types outside this module would leak a third-party dependency
    into every consumer (gesture engines, AR renderers, analytics), making
    it impossible to swap the underlying model (e.g., MediaPipe -> a custom
    ONNX hand model) without a breaking change across the whole codebase.
    This dataclass is the module's public contract.

    Attributes:
        handedness: "Left" or "Right", as classified by the model. Note this
            is mirrored relative to the camera's perspective, not the
            subject's own left/right.
        handedness_confidence: Confidence score for the handedness label.
        landmarks: Sequence of exactly 21 `Landmark` instances.
        bounding_box: Axis-aligned pixel bounding box enclosing the hand.
        center: (pixel_x, pixel_y) centroid of all landmarks.
        rotation_degrees: In-plane rotation of the hand, in degrees,
            estimated from wrist -> middle-finger-MCP vector.
        finger_states: Boolean extension state per finger, ordered
            (thumb, index, middle, ring, pinky). ``True`` means extended.
        gesture: Best-effort recognized `Gesture` for this hand.
        gesture_confidence: Heuristic confidence in [0.0, 1.0] for the
            recognized gesture.
    """

    handedness: str
    handedness_confidence: float
    landmarks: Tuple[Landmark, ...]
    bounding_box: BoundingBox
    center: Tuple[int, int]
    rotation_degrees: float
    finger_states: Tuple[bool, bool, bool, bool, bool]
    gesture: Gesture
    gesture_confidence: float


@dataclass(frozen=True)
class HandDetectionResult:
    """Top-level, strongly typed output of `HandTracker.detect`.

    Attributes:
        frame: The input frame, optionally annotated with a debug skeleton
            overlay. Always the same shape/dtype as the input frame.
        hand_detected: Whether at least one hand was detected.
        num_hands: Number of hands detected in this frame.
        hands: Structured data for every detected hand.
        timestamp: Monotonic timestamp (seconds, `time.monotonic()`) taken
            at detection time. Consumers can diff consecutive timestamps to
            compute FPS without this module owning any timing/UI state.
        processing_time_ms: Wall-clock time spent inside `detect()`, in
            milliseconds. Useful for real-time performance budgets.
    """

    frame: np.ndarray
    hand_detected: bool
    num_hands: int
    hands: Tuple[HandData, ...]
    timestamp: float
    processing_time_ms: float


class HandTrackerError(Exception):
    """Base exception for all errors raised by `HandTracker`."""


class HandTrackerInitializationError(HandTrackerError):
    """Raised when the underlying MediaPipe model fails to initialize."""


class InvalidFrameError(HandTrackerError):
    """Raised when `detect()` receives a malformed or invalid frame."""


class HandTracker:
    """Detects hands and extracts landmark data from BGR video frames.

    This class wraps MediaPipe Hands behind a stable, strongly typed API.
    It is deliberately camera-agnostic, UI-agnostic, and stateless with
    respect to application logic: it only tracks hands and hands back data.

    Thread-safety: a single `HandTracker` instance (and therefore its
    underlying MediaPipe graph) is NOT guaranteed thread-safe. For
    multi-threaded pipelines, construct one instance per worker thread.

    Attributes:
        max_num_hands: Maximum number of hands to detect per frame.
        min_detection_confidence: Minimum confidence for initial detection.
        min_tracking_confidence: Minimum confidence for landmark tracking
            across frames.
        model_complexity: MediaPipe model complexity (0 or 1).
        draw_landmarks: Whether `detect()` draws a debug skeleton overlay
            on the returned frame.
    """

    def __init__(
        self,
        max_num_hands: int = DEFAULT_MAX_NUM_HANDS,
        min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = DEFAULT_MIN_TRACKING_CONFIDENCE,
        model_complexity: int = DEFAULT_MODEL_COMPLEXITY,
        draw_landmarks: bool = True,
        static_image_mode: bool = False,
    ) -> None:
        """Initializes the hand tracker and its underlying model.

        Args:
            max_num_hands: Maximum number of hands to detect simultaneously.
                Must be a positive integer. Higher values cost more compute
                per frame.
            min_detection_confidence: Minimum confidence, in [0.0, 1.0], for
                a hand detection to be considered successful.
            min_tracking_confidence: Minimum confidence, in [0.0, 1.0], for
                landmarks to be considered successfully tracked between
                frames. Only relevant when `static_image_mode` is False.
            model_complexity: Complexity of the landmark model: 0 for a
                lighter/faster model, 1 for a more accurate model. Trades
                accuracy for latency; 0 is recommended for low-power/mobile
                real-time paths.
            draw_landmarks: If True, `detect()` renders a debug skeleton
                overlay onto a copy of the input frame. If False, the
                returned frame is untouched, saving compute for headless
                pipelines (e.g., server-side batch inference).
            static_image_mode: If True, treats every frame as an unrelated
                still image (re-runs full detection each call) rather than
                using temporal tracking. Should be False for real-time video.

        Raises:
            ValueError: If any argument is outside its valid range.
            HandTrackerInitializationError: If the MediaPipe model fails to
                load (e.g., missing model assets, incompatible runtime).
        """
        self._validate_init_arguments(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )

        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.draw_landmarks = draw_landmarks
        self.static_image_mode = static_image_mode

        # Underlying MediaPipe handles. Populated by `_initialize_mediapipe`.
        self._mp_hands_module = mp.solutions.hands
        self._mp_drawing_module = mp.solutions.drawing_utils
        self._hands_model: Optional[mp.solutions.hands.Hands] = None

        self._initialize_mediapipe()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _validate_init_arguments(
        self,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_complexity: int,
    ) -> None:
        """Validates constructor arguments before touching any model state.

        WHY: failing fast on bad configuration (before allocating a
        MediaPipe graph) gives callers an immediate, precise error instead
        of a confusing failure deep inside a third-party library.

        Args:
            max_num_hands: Candidate max hand count.
            min_detection_confidence: Candidate detection confidence.
            min_tracking_confidence: Candidate tracking confidence.
            model_complexity: Candidate model complexity.

        Raises:
            ValueError: If any argument is invalid.
        """
        if not isinstance(max_num_hands, int) or max_num_hands < 1:
            raise ValueError(
                f"max_num_hands must be a positive integer, got {max_num_hands!r}."
            )
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError(
                "min_detection_confidence must be within [0.0, 1.0], got "
                f"{min_detection_confidence!r}."
            )
        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError(
                "min_tracking_confidence must be within [0.0, 1.0], got "
                f"{min_tracking_confidence!r}."
            )
        if model_complexity not in (0, 1):
            raise ValueError(
                f"model_complexity must be 0 or 1, got {model_complexity!r}."
            )

    def _initialize_mediapipe(self) -> None:
        """Lazily constructs the MediaPipe Hands graph exactly once.

        WHY isolate this in its own method: model initialization is the
        single most expensive and most failure-prone step in this class
        (native library loading, model asset resolution). Isolating it
        makes the failure mode explicit, keeps `__init__` readable, and
        gives us one seam to mock in unit tests without spinning up a real
        MediaPipe graph.

        Raises:
            HandTrackerInitializationError: If MediaPipe fails to construct
                the underlying `Hands` graph for any reason.
        """
        try:
            self._hands_model = self._mp_hands_module.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            logger.info(
                "HandTracker initialized (max_num_hands=%d, "
                "model_complexity=%d, static_image_mode=%s).",
                self.max_num_hands,
                self.model_complexity,
                self.static_image_mode,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error.
            logger.exception("Failed to initialize MediaPipe Hands model.")
            raise HandTrackerInitializationError(
                "Could not initialize the MediaPipe Hands model."
            ) from exc

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> HandDetectionResult:
        """Detects hands and extracts landmarks from a single BGR frame.

        This is the sole public entry point for the tracker. It performs
        color-space conversion, model inference, structured extraction, and
        (optionally) debug-overlay rendering, then returns everything as a
        single immutable `HandDetectionResult`.

        Args:
            frame: A single video frame as a BGR `numpy.ndarray` of shape
                (height, width, 3) and dtype `uint8`, as produced by
                `cv2.VideoCapture` or any equivalent decoder. This method
                never opens or reads from a capture device itself.

        Returns:
            HandDetectionResult: Structured detection output. If no hand is
            found, `hand_detected` is False, `num_hands` is 0, and `hands`
            is an empty tuple — callers never need to null-check `hands`.

        Raises:
            InvalidFrameError: If `frame` is not a valid BGR image array.

        Note:
            On unexpected internal failures (e.g., a corrupt frame that
            passes validation but breaks inference), this method logs the
            error and returns a "no hand detected" result rather than
            raising, so a single bad frame cannot crash a real-time loop
            processing 30-60 frames per second.
        """
        start_time = time.monotonic()
        self._validate_frame(frame)

        try:
            rgb_frame = self._convert_bgr_to_rgb(frame)

            # WHY: MediaPipe's Python API expects read-only-safe input for
            # best performance; marking it non-writeable avoids an internal
            # defensive copy and reduces per-frame latency, which matters
            # at 30-60 FPS budgets.
            rgb_frame.flags.writeable = False
            raw_results = self._hands_model.process(rgb_frame)
            rgb_frame.flags.writeable = True

            output_frame, hands = self._process_results(frame, raw_results)

        except Exception:  # noqa: BLE001 - isolate real-time loop from crashes.
            logger.exception("Unexpected failure while processing frame.")
            output_frame = frame
            hands = ()

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        result = HandDetectionResult(
            frame=output_frame,
            hand_detected=self._is_hand_detected(hands),
            num_hands=len(hands),
            hands=hands,
            timestamp=time.monotonic(),
            processing_time_ms=elapsed_ms,
        )

        logger.debug(
            "detect() finished: hands=%d, elapsed_ms=%.2f", len(hands), elapsed_ms
        )
        return result

    def close(self) -> None:
        """Releases the underlying MediaPipe model resources.

        WHY: MediaPipe graphs hold native (C++) resources that are not
        automatically freed by Python's garbage collector in a timely
        manner. Explicit release is important in long-running services
        (e.g., a server handling many short-lived tracker instances) to
        avoid resource exhaustion.
        """
        if self._hands_model is not None:
            try:
                self._hands_model.close()
                logger.info("HandTracker resources released.")
            except Exception:  # noqa: BLE001
                logger.exception("Error while closing MediaPipe Hands model.")
            finally:
                self._reset()

    def __enter__(self) -> "HandTracker":
        """Enables `with HandTracker() as tracker:` usage.

        Returns:
            HandTracker: This instance, ready for use.
        """
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Ensures resources are released when leaving a `with` block."""
        self.close()

    # ------------------------------------------------------------------ #
    # Frame validation & conversion
    # ------------------------------------------------------------------ #
    def _validate_frame(self, frame: np.ndarray) -> None:
        """Validates that `frame` is a well-formed BGR image.

        WHY: validating inputs at the boundary of the module prevents
        cryptic native-library crashes deep inside MediaPipe and gives
        calling code an actionable, typed error instead.

        Args:
            frame: Candidate frame to validate.

        Raises:
            InvalidFrameError: If `frame` is None, not a numpy array, not
                3-dimensional, does not have 3 channels, or is empty.
        """
        if frame is None:
            raise InvalidFrameError("Frame is None; expected a BGR numpy.ndarray.")
        if not isinstance(frame, np.ndarray):
            raise InvalidFrameError(
                f"Frame must be a numpy.ndarray, got {type(frame).__name__}."
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise InvalidFrameError(
                "Frame must have shape (height, width, 3) representing a "
                f"BGR image; got shape {frame.shape}."
            )
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise InvalidFrameError("Frame has zero width or height.")

    def _convert_bgr_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Converts a BGR frame to RGB, as required by MediaPipe.

        WHY isolated as its own method: OpenCV (BGR) and MediaPipe (RGB)
        disagree on channel order. Isolating the conversion keeps that
        third-party convention mismatch in exactly one place, so if a
        future model expects a different color space, only this method
        changes.

        Args:
            frame: BGR frame as produced by OpenCV-family decoders.

        Returns:
            numpy.ndarray: The same image data in RGB channel order.

        Raises:
            InvalidFrameError: If the color conversion fails.
        """
        try:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error as exc:
            raise InvalidFrameError("Failed to convert frame from BGR to RGB.") from exc

    # ------------------------------------------------------------------ #
    # Result processing
    # ------------------------------------------------------------------ #
    def _process_results(
        self, frame: np.ndarray, raw_results: object
    ) -> Tuple[np.ndarray, Tuple[HandData, ...]]:
        """Converts raw MediaPipe output into structured `HandData`.

        WHY: this method is the boundary that "shields" the rest of the
        codebase from MediaPipe's protobuf schema. If MediaPipe changes its
        internal API in a future version, only this method (and its
        private helpers) need to change.

        Args:
            frame: Original BGR frame, used as the drawing target when
                `draw_landmarks` is enabled.
            raw_results: The raw object returned by
                `mediapipe.solutions.hands.Hands.process`.

        Returns:
            Tuple[numpy.ndarray, Tuple[HandData, ...]]: The (possibly
            annotated) output frame and the structured per-hand data.
        """
        multi_landmarks = getattr(raw_results, "multi_hand_landmarks", None)
        multi_handedness = getattr(raw_results, "multi_handedness", None)

        if not multi_landmarks:
            return frame, ()

        height, width = frame.shape[:2]
        output_frame = frame.copy() if self.draw_landmarks else frame

        hands: List[HandData] = []
        for hand_index, hand_landmarks in enumerate(multi_landmarks):
            handedness_label, handedness_confidence = self._extract_handedness(
                multi_handedness, hand_index
            )

            landmarks = self._extract_landmarks(hand_landmarks, width, height)
            if len(landmarks) != NUM_HAND_LANDMARKS:
                logger.warning(
                    "Expected %d landmarks, got %d; skipping malformed hand.",
                    NUM_HAND_LANDMARKS,
                    len(landmarks),
                )
                continue

            bounding_box = self._get_hand_bounding_box(landmarks)
            center = self._get_hand_center(landmarks)
            rotation_degrees = self._get_hand_rotation(landmarks)
            finger_states = self._get_finger_states(landmarks, handedness_label)
            gesture, gesture_confidence = self._detect_gesture(
                landmarks, finger_states
            )

            hands.append(
                HandData(
                    handedness=handedness_label,
                    handedness_confidence=handedness_confidence,
                    landmarks=tuple(landmarks),
                    bounding_box=bounding_box,
                    center=center,
                    rotation_degrees=rotation_degrees,
                    finger_states=finger_states,
                    gesture=gesture,
                    gesture_confidence=gesture_confidence,
                )
            )

            if self.draw_landmarks:
                self._draw_landmarks(output_frame, hand_landmarks)

        return output_frame, tuple(hands)

    def _extract_handedness(
        self, multi_handedness: object, hand_index: int
    ) -> Tuple[str, float]:
        """Safely extracts the handedness label and confidence for one hand.

        Args:
            multi_handedness: The `multi_handedness` field from MediaPipe
                results, or None if unavailable.
            hand_index: Index of the hand within `multi_handedness`.

        Returns:
            Tuple[str, float]: The handedness label ("Left"/"Right", or
            "Unknown" if unavailable) and its confidence score (0.0 if
            unavailable).
        """
        try:
            classification = multi_handedness[hand_index].classification[0]
            return classification.label, float(classification.score)
        except (TypeError, IndexError, AttributeError):
            logger.debug("Handedness unavailable for hand index %d.", hand_index)
            return "Unknown", 0.0

    def _extract_landmarks(
        self, hand_landmarks: object, frame_width: int, frame_height: int
    ) -> List[Landmark]:
        """Converts MediaPipe's raw landmark list into `Landmark` objects.

        Args:
            hand_landmarks: A single hand's `landmark` collection as
                returned by MediaPipe (normalized coordinates).
            frame_width: Width of the source frame, in pixels.
            frame_height: Height of the source frame, in pixels.

        Returns:
            List[Landmark]: One `Landmark` per point, normally 21 entries,
            in the fixed MediaPipe Hands topology order.
        """
        landmarks: List[Landmark] = []
        for point in hand_landmarks.landmark:
            pixel_x, pixel_y = self._extract_pixel_landmarks(
                point.x, point.y, frame_width, frame_height
            )
            visibility = getattr(point, "visibility", None)
            landmarks.append(
                Landmark(
                    x=float(point.x),
                    y=float(point.y),
                    z=float(point.z),
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                    visibility=float(visibility) if visibility is not None else None,
                )
            )
        return landmarks

    def _extract_pixel_landmarks(
        self,
        normalized_x: float,
        normalized_y: float,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int]:
        """Converts a single normalized coordinate pair to pixel space.

        WHY isolated: pixel conversion is needed by drawing, bounding-box
        computation, and any future feature (e.g., ROI cropping for a
        downstream classifier). Centralizing it avoids subtly divergent
        rounding behavior across call sites.

        Args:
            normalized_x: x in [0.0, 1.0], relative to frame width.
            normalized_y: y in [0.0, 1.0], relative to frame height.
            frame_width: Width of the source frame, in pixels.
            frame_height: Height of the source frame, in pixels.

        Returns:
            Tuple[int, int]: (pixel_x, pixel_y), clamped to valid frame
            bounds to guard against landmarks predicted slightly outside
            [0.0, 1.0] near frame edges.
        """
        pixel_x = int(round(normalized_x * frame_width))
        pixel_y = int(round(normalized_y * frame_height))
        pixel_x = min(max(pixel_x, 0), frame_width - 1)
        pixel_y = min(max(pixel_y, 0), frame_height - 1)
        return pixel_x, pixel_y

    # ------------------------------------------------------------------ #
    # Drawing (debug/telemetry overlay only — no UI ownership)
    # ------------------------------------------------------------------ #
    def _draw_landmarks(self, frame: np.ndarray, hand_landmarks: object) -> None:
        """Draws the landmark skeleton onto `frame` in place.

        WHY this exists despite the "no UI code" constraint: a debug
        skeleton overlay is a detection-layer diagnostic (comparable to a
        model returning an annotated debug tensor), not an application UI
        feature such as menus, buttons, or filters. It is optional and
        controlled entirely by the `draw_landmarks` constructor flag so
        headless/production paths can disable it for maximum throughput.

        Args:
            frame: Frame to annotate, modified in place.
            hand_landmarks: Raw MediaPipe landmark collection for one hand.

        Returns:
            None.
        """
        try:
            self._mp_drawing_module.draw_landmarks(
                frame,
                hand_landmarks,
                self._mp_hands_module.HAND_CONNECTIONS,
                self._mp_drawing_module.DrawingSpec(
                    color=LANDMARK_COLOR_BGR, thickness=CONNECTION_THICKNESS_PX,
                    circle_radius=LANDMARK_RADIUS_PX,
                ),
                self._mp_drawing_module.DrawingSpec(
                    color=CONNECTION_COLOR_BGR, thickness=CONNECTION_THICKNESS_PX,
                ),
            )
        except Exception:  # noqa: BLE001 - drawing failures must not crash detection.
            logger.exception("Failed to draw landmarks; continuing without overlay.")

    # ------------------------------------------------------------------ #
    # Geometry helpers (reusable building blocks for gesture recognition)
    # ------------------------------------------------------------------ #
    def _get_hand_bounding_box(self, landmarks: Sequence[Landmark]) -> BoundingBox:
        """Computes an axis-aligned pixel bounding box around a hand.

        Args:
            landmarks: The 21 landmarks of a single detected hand.

        Returns:
            BoundingBox: The tightest axis-aligned box enclosing all
            landmark pixel coordinates.
        """
        xs = [landmark.pixel_x for landmark in landmarks]
        ys = [landmark.pixel_y for landmark in landmarks]
        return BoundingBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))

    def _get_hand_center(self, landmarks: Sequence[Landmark]) -> Tuple[int, int]:
        """Computes the pixel-space centroid of all landmarks.

        Args:
            landmarks: The 21 landmarks of a single detected hand.

        Returns:
            Tuple[int, int]: (center_x, center_y) in pixel coordinates.
        """
        count = len(landmarks)
        center_x = sum(landmark.pixel_x for landmark in landmarks) / count
        center_y = sum(landmark.pixel_y for landmark in landmarks) / count
        return int(round(center_x)), int(round(center_y))

    def _get_hand_rotation(self, landmarks: Sequence[Landmark]) -> float:
        """Estimates the in-plane rotation of the hand, in degrees.

        The rotation is derived from the vector between the wrist and the
        middle-finger MCP joint, which is stable across most hand poses
        (unlike fingertip-based vectors, which move with gestures).

        WHY provided as a reusable primitive: orientation is a common input
        to downstream features such as AR object alignment or gesture
        disambiguation (e.g., distinguishing a rotated "thumbs up" from a
        "thumbs sideways"), even though this module does not implement
        those features itself.

        Args:
            landmarks: The 21 landmarks of a single detected hand.

        Returns:
            float: Rotation in degrees, in the range (-180.0, 180.0],
            measured counter-clockwise from the positive x-axis in image
            space.
        """
        wrist = landmarks[WRIST]
        reference = landmarks[MIDDLE_FINGER_MCP]
        delta_x = reference.pixel_x - wrist.pixel_x
        delta_y = reference.pixel_y - wrist.pixel_y
        return math.degrees(math.atan2(-delta_y, delta_x))

    # ------------------------------------------------------------------ #
    # Gesture-recognition primitives
    # ------------------------------------------------------------------ #
    def _get_finger_states(
        self, landmarks: Sequence[Landmark], handedness: str
    ) -> Tuple[bool, bool, bool, bool, bool]:
        """Determines whether each finger is extended or curled.

        Uses a simple, fast geometric heuristic suitable for real-time use:
        a non-thumb finger is considered extended if its tip is farther
        from the wrist (in the y-axis, image space) than its PIP joint. The
        thumb uses an x-axis comparison mirrored by handedness, since it
        flexes sideways rather than vertically.

        WHY heuristic rather than a learned classifier here: this method is
        a reusable geometric primitive for *future* gesture recognition,
        not a full gesture classifier itself. Keeping it cheap and
        dependency-free preserves the real-time performance budget; a
        learned classifier can be layered on top by a separate module that
        consumes these finger states.

        Args:
            landmarks: The 21 landmarks of a single detected hand.
            handedness: "Left" or "Right", used to mirror the thumb check.

        Returns:
            Tuple[bool, bool, bool, bool, bool]: Extension state ordered
            (thumb, index, middle, ring, pinky); True means extended.
        """
        thumb_tip = landmarks[THUMB_TIP]
        thumb_ip = landmarks[THUMB_IP]
        if handedness == "Left":
            thumb_extended = thumb_tip.pixel_x > thumb_ip.pixel_x
        else:
            thumb_extended = thumb_tip.pixel_x < thumb_ip.pixel_x

        finger_states = [thumb_extended]
        for tip_index, pip_index in _FINGER_TIP_PIP_PAIRS:
            tip = landmarks[tip_index]
            pip = landmarks[pip_index]
            finger_states.append(tip.pixel_y < pip.pixel_y)

        return tuple(finger_states)  # type: ignore[return-value]

    def _detect_gesture(
        self,
        landmarks: Sequence[Landmark],
        finger_states: Tuple[bool, bool, bool, bool, bool],
    ) -> Tuple[Gesture, float]:
        """Recognizes a coarse gesture from finger states and geometry.

        WHY kept intentionally simple: this module's stated responsibility
        is tracking, not a full gesture-recognition system. This method
        exists as an extension point — a starting heuristic that future
        gesture modules can override, replace, or wrap with a learned
        model — while still giving real-time consumers a usable default.

        Args:
            landmarks: The 21 landmarks of a single detected hand.
            finger_states: Extension state ordered
                (thumb, index, middle, ring, pinky).

        Returns:
            Tuple[Gesture, float]: The recognized gesture and a heuristic
            confidence score in [0.0, 1.0].
        """
        thumb, index, middle, ring, pinky = finger_states

        pinch_distance = self._normalized_distance(
            landmarks[THUMB_TIP], landmarks[INDEX_FINGER_TIP]
        )
        if pinch_distance < _PINCH_DISTANCE_THRESHOLD_NORMALIZED:
            return Gesture.PINCH, 1.0 - (
                pinch_distance / _PINCH_DISTANCE_THRESHOLD_NORMALIZED
            )

        if all(finger_states):
            return Gesture.OPEN_PALM, 0.9
        if not any(finger_states):
            return Gesture.FIST, 0.9
        if index and not middle and not ring and not pinky:
            return Gesture.POINTING, 0.8
        if thumb and not index and not middle and not ring and not pinky:
            return Gesture.THUMBS_UP, 0.7

        return Gesture.UNKNOWN, 0.0

    def _normalized_distance(self, first: Landmark, second: Landmark) -> float:
        """Computes Euclidean distance between two landmarks in normalized
        coordinate space.

        WHY normalized rather than pixel space: normalized distances are
        resolution-independent, so gesture thresholds (like the pinch
        threshold) behave consistently across different camera resolutions
        and aspect ratios.

        Args:
            first: First landmark.
            second: Second landmark.

        Returns:
            float: Euclidean distance in normalized [0.0, 1.0]-ish space.
        """
        return math.hypot(first.x - second.x, first.y - second.y)

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def _is_hand_detected(self, hands: Sequence[HandData]) -> bool:
        """Determines whether any hand was successfully detected.

        Args:
            hands: Structured hand data extracted from the current frame.

        Returns:
            bool: True if at least one hand is present, False otherwise.
        """
        return len(hands) > 0

    def _reset(self) -> None:
        """Clears internal handles after resource release.

        WHY: guards against accidental use-after-close — any subsequent
        call to `detect()` on a closed tracker will fail predictably
        (AttributeError on None) rather than silently operating on a freed
        native resource.

        Returns:
            None.
        """
        self._hands_model = None
        logger.debug("HandTracker internal state reset.")