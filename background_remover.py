"""Real-time person segmentation, background replacement, and alpha blending.

This module has exactly one responsibility: turning a BGR video frame plus
an optionally configured "virtual background" into a composited BGR frame.
It owns no camera, no window, no keyboard input, and no application state
machine — those concerns belong to orchestration layers above this module
(see the "THIS FILE MUST NEVER" boundary enforced throughout the class
docstrings below). That narrow scope is what lets this exact class be
reused unmodified across products with wildly different UI stacks (a
mobile AR camera, a desktop video-conferencing app, a server-side batch
renderer) — the Single Responsibility Principle applied at the module
level.

Typical usage::

    config = BackgroundRemovalConfig(confidence_threshold=0.6)
    remover = BackgroundRemover(config=config)
    remover.initialize()

    remover.save_background(clean_plate_frame)  # captured with no person
    remover.fade_in()

    result = remover.replace_background(frame)
    # result.processed_frame is ready for the caller's own display/encode
    # pipeline; this module never touches a window or a file.

    remover.close()

Author: Computer Vision Platform Team
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, Tuple

import cv2
import mediapipe as mp
import numpy as np

# --------------------------------------------------------------------------- #
# Module-level logger.
#
# WHY: a real-time AR pipeline runs inside services, mobile bridges, and
# render threads where stdout is either unavailable or a performance
# liability. A named logger lets host applications route, filter, and rate
# limit this module's diagnostics without touching this file.
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants
#
# WHY centralize: every one of these values is a tuning knob a graphics or
# ML engineer will eventually need to adjust for a new device tier or
# camera sensor. Naming them here — instead of scattering literals through
# the algorithm — makes every tunable parameter discoverable in one place
# and prevents silent inconsistency between call sites.
# --------------------------------------------------------------------------- #
NUM_BGR_CHANNELS = 3
BGR_MIN_VALUE = 0
BGR_MAX_VALUE = 255

DEFAULT_MODEL_SELECTION = 1  # 0 = landscape/fast model, 1 = general model.
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_ENABLE_THRESHOLD_BINARIZATION = False
DEFAULT_ENABLE_MORPHOLOGICAL_OPENING = True
DEFAULT_ENABLE_MORPHOLOGICAL_CLOSING = True
DEFAULT_MORPHOLOGY_KERNEL_SIZE = 5
DEFAULT_ENABLE_GAUSSIAN_BLUR = True
DEFAULT_GAUSSIAN_BLUR_KERNEL_SIZE = 7
DEFAULT_ENABLE_EDGE_SMOOTHING = True
DEFAULT_EDGE_FEATHER_KERNEL_SIZE = 9
DEFAULT_FADE_DURATION_SECONDS = 0.5
DEFAULT_ALPHA_ANIMATION_SPEED = None  # None -> derive from fade duration.
DEFAULT_BLEND_ALPHA = 1.0
DEFAULT_FADE_ALPHA = 1.0
MIN_ALPHA = 0.0
MAX_ALPHA = 1.0
DEFAULT_BACKGROUND_RESIZE_INTERPOLATION = cv2.INTER_LINEAR
_ALLOWED_INTERPOLATIONS = frozenset(
    {cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA}
)
_MIN_ODD_KERNEL_SIZE = 3


class BackgroundSourceType(str, Enum):
    """Enumerates the origin of the currently configured virtual background.

    WHY an Enum: downstream compositing logic branches on this value.
    String literals invite typos that only surface at runtime with a wrong
    (but silent) background; an Enum makes invalid states unrepresentable
    and gives static analyzers something to check.
    """

    NONE = "none"
    SAVED_FRAME = "saved_frame"
    COLOR = "color"
    IMAGE = "image"
    VIDEO = "video"  # Reserved for a future extension; see `_compute_background`.


BGRColor = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
# Exceptions
#
# WHY typed exceptions instead of bare `Exception`/`ValueError` everywhere:
# callers embedding this module in a larger AR engine need to distinguish
# "you gave me bad input" from "the model failed to load" from "you asked
# me to composite before I had a background" — each demands a different
# recovery strategy at the call site.
# --------------------------------------------------------------------------- #
class BackgroundRemoverError(Exception):
    """Base exception for all errors raised by `BackgroundRemover`."""


class BackgroundRemoverInitializationError(BackgroundRemoverError):
    """Raised when the underlying segmentation model fails to initialize."""


class InvalidFrameError(BackgroundRemoverError):
    """Raised when a method receives a malformed or invalid image array."""


class InvalidConfigurationError(BackgroundRemoverError):
    """Raised when `BackgroundRemovalConfig` contains invalid values."""


class BackgroundNotAvailableError(BackgroundRemoverError):
    """Raised when a background-dependent operation has no background set."""


class NotInitializedError(BackgroundRemoverError):
    """Raised when a model-dependent method is called before `initialize()`."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BackgroundRemovalConfig:
    """Immutable configuration for `BackgroundRemover`.

    Grouping every tunable parameter into a single frozen dataclass (rather
    than a long list of constructor keyword arguments) keeps `__init__`
    readable, makes configurations trivially serializable for A/B testing
    across device tiers, and prevents accidental in-place mutation of
    "live" settings from an unrelated part of the codebase.

    Attributes:
        model_selection: MediaPipe Selfie Segmentation model variant.
            0 selects the lighter "landscape" model (faster, tuned for
            wide shots); 1 selects the "general" model (more accurate,
            works for both landscape and portrait framing).
        confidence_threshold: Threshold in [0.0, 1.0] used when
            `enable_threshold_binarization` is True to convert the
            probabilistic segmentation mask into a hard foreground/
            background decision.
        enable_threshold_binarization: If True, the mask is hard-
            thresholded at `confidence_threshold` before refinement. If
            False, the soft probability mask is kept, which usually
            produces smoother edges at the cost of slightly "leaking"
            background in low-confidence regions.
        enable_morphological_opening: If True, applies an opening
            operation (erosion then dilation) to remove small false-
            positive noise specks from the mask.
        enable_morphological_closing: If True, applies a closing operation
            (dilation then erosion) to fill small holes inside the person
            silhouette.
        morphology_kernel_size: Odd, positive kernel size (pixels) used for
            both morphological operations.
        enable_gaussian_blur: If True, applies Gaussian blur to the mask
            to remove hard, aliased edges before compositing.
        gaussian_blur_kernel_size: Odd, positive kernel size (pixels) for
            the Gaussian blur pass.
        enable_edge_smoothing: If True, applies an additional lightweight
            feathering pass focused on producing anti-aliased edges,
            distinct from the general-purpose Gaussian blur above.
        edge_feather_kernel_size: Odd, positive kernel size (pixels) for
            the feathering/anti-aliasing pass.
        fade_duration_seconds: Duration, in seconds, of a full 0.0 -> 1.0
            (or 1.0 -> 0.0) alpha transition triggered by `fade_in` /
            `fade_out`, when `alpha_animation_speed` is not explicitly set.
        alpha_animation_speed: Optional explicit fade speed in alpha units
            per second. When None (the default), the speed is derived from
            `fade_duration_seconds` so a full transition always takes
            exactly that long regardless of frame rate.
        background_resize_interpolation: OpenCV interpolation flag used
            when resizing a saved/loaded background image to match the
            live frame resolution.
        enable_gpu_acceleration: Reserved extension flag signaling that a
            GPU-backed `SegmentationBackend` should be preferred where
            available. The bundled MediaPipe backend does not branch on
            this flag directly (MediaPipe manages its own execution
            delegate internally); it exists so a future backend
            implementation (e.g., a TensorRT or CoreML backend) can be
            selected purely through configuration, without changing any
            call site. See `SegmentationBackend`.

    Raises:
        InvalidConfigurationError: If any field is outside its valid range,
            raised eagerly from `__post_init__` so misconfiguration is
            caught at construction time, not mid-stream in a video loop.
    """

    model_selection: int = DEFAULT_MODEL_SELECTION
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    enable_threshold_binarization: bool = DEFAULT_ENABLE_THRESHOLD_BINARIZATION
    enable_morphological_opening: bool = DEFAULT_ENABLE_MORPHOLOGICAL_OPENING
    enable_morphological_closing: bool = DEFAULT_ENABLE_MORPHOLOGICAL_CLOSING
    morphology_kernel_size: int = DEFAULT_MORPHOLOGY_KERNEL_SIZE
    enable_gaussian_blur: bool = DEFAULT_ENABLE_GAUSSIAN_BLUR
    gaussian_blur_kernel_size: int = DEFAULT_GAUSSIAN_BLUR_KERNEL_SIZE
    enable_edge_smoothing: bool = DEFAULT_ENABLE_EDGE_SMOOTHING
    edge_feather_kernel_size: int = DEFAULT_EDGE_FEATHER_KERNEL_SIZE
    fade_duration_seconds: float = DEFAULT_FADE_DURATION_SECONDS
    alpha_animation_speed: Optional[float] = DEFAULT_ALPHA_ANIMATION_SPEED
    background_resize_interpolation: int = DEFAULT_BACKGROUND_RESIZE_INTERPOLATION
    enable_gpu_acceleration: bool = False

    def __post_init__(self) -> None:
        """Validates all configuration fields eagerly.

        Raises:
            InvalidConfigurationError: If any field is invalid.
        """
        if self.model_selection not in (0, 1):
            raise InvalidConfigurationError(
                f"model_selection must be 0 or 1, got {self.model_selection!r}."
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise InvalidConfigurationError(
                "confidence_threshold must be within [0.0, 1.0], got "
                f"{self.confidence_threshold!r}."
            )
        self._validate_odd_kernel_size(
            "morphology_kernel_size", self.morphology_kernel_size
        )
        self._validate_odd_kernel_size(
            "gaussian_blur_kernel_size", self.gaussian_blur_kernel_size
        )
        self._validate_odd_kernel_size(
            "edge_feather_kernel_size", self.edge_feather_kernel_size
        )
        if self.fade_duration_seconds <= 0.0:
            raise InvalidConfigurationError(
                "fade_duration_seconds must be positive, got "
                f"{self.fade_duration_seconds!r}."
            )
        if self.alpha_animation_speed is not None and self.alpha_animation_speed <= 0.0:
            raise InvalidConfigurationError(
                "alpha_animation_speed must be positive when provided, got "
                f"{self.alpha_animation_speed!r}."
            )
        if self.background_resize_interpolation not in _ALLOWED_INTERPOLATIONS:
            raise InvalidConfigurationError(
                "background_resize_interpolation must be one of "
                f"{sorted(_ALLOWED_INTERPOLATIONS)}, got "
                f"{self.background_resize_interpolation!r}."
            )

    @staticmethod
    def _validate_odd_kernel_size(field_name: str, value: int) -> None:
        """Validates that a kernel size is a positive odd integer.

        Args:
            field_name: Name of the field being validated, used in error
                messages.
            value: Candidate kernel size.

        Raises:
            InvalidConfigurationError: If `value` is not a positive odd
                integer of at least `_MIN_ODD_KERNEL_SIZE`.
        """
        if (
            not isinstance(value, int)
            or value < _MIN_ODD_KERNEL_SIZE
            or value % 2 == 0
        ):
            raise InvalidConfigurationError(
                f"{field_name} must be an odd integer >= {_MIN_ODD_KERNEL_SIZE}, "
                f"got {value!r}."
            )

    @property
    def effective_alpha_speed(self) -> float:
        """float: Alpha units per second used to drive fade animations.

        Returns the explicit `alpha_animation_speed` if set, otherwise
        derives a speed that completes a full 0.0 <-> 1.0 transition in
        exactly `fade_duration_seconds`.
        """
        if self.alpha_animation_speed is not None:
            return self.alpha_animation_speed
        return (MAX_ALPHA - MIN_ALPHA) / self.fade_duration_seconds


# --------------------------------------------------------------------------- #
# Result dataclasses
#
# WHY dataclasses instead of dicts: a dict-based return value has no fixed
# shape a type checker can verify, invites `KeyError` typos at every call
# site, and silently accepts extra/missing keys. These dataclasses are this
# module's public contract and are the only artifacts consumers should
# depend on — never the MediaPipe objects underneath.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SegmentationMask:
    """A person-segmentation mask at one or more stages of refinement.

    Attributes:
        raw_mask: Raw probability mask straight from the model, shape
            (height, width), dtype float32, values in [0.0, 1.0] where
            higher means "more likely foreground (person)".
        refined_mask: Mask after `refine_mask` post-processing (threshold,
            morphology, blur/feathering), same shape/dtype as `raw_mask`.
            `None` if refinement has not yet been applied.
        threshold: Confidence threshold associated with this mask.
        timestamp: Monotonic timestamp (`time.monotonic()`) when the raw
            mask was produced.
    """

    raw_mask: np.ndarray
    refined_mask: Optional[np.ndarray]
    threshold: float
    timestamp: float

    @property
    def is_refined(self) -> bool:
        """bool: Whether `refine_mask` has been applied to this instance."""
        return self.refined_mask is not None

    @property
    def active_mask(self) -> np.ndarray:
        """numpy.ndarray: The refined mask if available, else the raw mask.

        WHY: gives every downstream consumer (compositing, future gesture
        or pose modules) a single field to read without needing to know
        whether refinement happened, while still preserving both stages
        for debugging/telemetry.
        """
        return self.refined_mask if self.refined_mask is not None else self.raw_mask


@dataclass(frozen=True)
class Foreground:
    """The extracted foreground (person) layer, ready for compositing.

    Attributes:
        image: BGR image, shape (height, width, 3), dtype uint8. This is
            intentionally the *full* source frame rather than a pre-masked
            copy — see the WHY note on `_compute_foreground` for the
            performance rationale — with `alpha_mask` carrying the per-
            pixel foreground weight used at composite time.
        alpha_mask: Per-pixel foreground weight, shape (height, width),
            dtype float32, values in [0.0, 1.0].
        timestamp: Monotonic timestamp when this foreground was computed.
    """

    image: np.ndarray
    alpha_mask: np.ndarray
    timestamp: float


@dataclass(frozen=True)
class Background:
    """A background layer, resolved to the live frame's resolution.

    Attributes:
        image: BGR image, shape (height, width, 3), dtype uint8, resized/
            generated to exactly match the frame currently being composited.
        source: Which configured source produced this background.
        timestamp: Monotonic timestamp when this background was resolved.
    """

    image: np.ndarray
    source: BackgroundSourceType
    timestamp: float


@dataclass(frozen=True)
class BackgroundRemovalResult:
    """Top-level, strongly typed output of `BackgroundRemover.replace_background`.

    Attributes:
        processed_frame: Final composited BGR frame, same shape/dtype as
            the input frame. This is the only artifact meant for display,
            encoding, or further pipeline stages.
        mask: The `SegmentationMask` computed for this frame.
        foreground: The `Foreground` computed for this frame.
        background: The `Background` used for compositing, or `None` if no
            background was configured (in which case `processed_frame` is
            the original frame, unmodified except for the fade pass).
        alpha: The effective overall alpha applied to the composited
            effect for this frame (see `apply_fade`); this is the
            time-animated fade value, distinct from the static per-pixel
            blend strength set via `set_alpha`.
        background_available: Whether a background was configured for
            this call.
        timestamp: Monotonic timestamp taken at the end of processing.
        processing_time_ms: Wall-clock time spent inside
            `replace_background`, in milliseconds.
    """

    processed_frame: np.ndarray
    mask: SegmentationMask
    foreground: Foreground
    background: Optional[Background]
    alpha: float
    background_available: bool
    timestamp: float
    processing_time_ms: float


# --------------------------------------------------------------------------- #
# Segmentation backend abstraction
#
# WHY a Protocol here: this is the module's primary extension seam for
# "Support future GPU acceleration." `BackgroundRemover` depends only on
# this interface (Dependency Inversion Principle), so a future
# TensorRT/CoreML/ONNX-Runtime-GPU backend can be dropped in via
# constructor injection without touching a single line of compositing,
# masking, or blending logic below.
# --------------------------------------------------------------------------- #
class SegmentationBackend(Protocol):
    """Interface for any model that turns an RGB frame into a person mask."""

    def process(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Runs person segmentation on a single RGB frame.

        Args:
            rgb_frame: RGB image, shape (height, width, 3), dtype uint8.

        Returns:
            numpy.ndarray: Probability mask, shape (height, width), dtype
            float32, values in [0.0, 1.0].
        """
        ...

    def close(self) -> None:
        """Releases any native resources held by the backend."""
        ...


class _MediaPipeSelfieSegmentationBackend:
    """`SegmentationBackend` implementation backed by MediaPipe Selfie Segmentation.

    This is the default, CPU-friendly backend used when no custom backend
    is injected into `BackgroundRemover`.
    """

    def __init__(self, model_selection: int) -> None:
        """Initializes the MediaPipe Selfie Segmentation graph.

        Args:
            model_selection: 0 for the landscape model, 1 for the general
                model.

        Raises:
            BackgroundRemoverInitializationError: If the underlying
                MediaPipe graph fails to construct.
        """
        try:
            self._model = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=model_selection
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error.
            raise BackgroundRemoverInitializationError(
                "Could not initialize the MediaPipe Selfie Segmentation model."
            ) from exc

    def process(self, rgb_frame: np.ndarray) -> np.ndarray:
        """See `SegmentationBackend.process`."""
        results = self._model.process(rgb_frame)
        return results.segmentation_mask.astype(np.float32)

    def close(self) -> None:
        """See `SegmentationBackend.close`."""
        self._model.close()


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #
class BackgroundRemover:
    """Real-time person segmentation, background replacement, and alpha blending.

    Purpose:
        Turns a BGR frame plus an optionally configured virtual background
        into a single composited BGR frame, with smooth fade transitions
        and configurable mask refinement. This is the segmentation/
        compositing layer of an AR pipeline — nothing more.

    Responsibilities:
        - Own exactly one segmentation model instance (via
          `SegmentationBackend`), initialized once.
        - Convert BGR -> RGB, run segmentation, refine the resulting mask.
        - Store at most one configured virtual background (saved frame,
          solid color, or static image).
        - Composite foreground and background with per-pixel feathering
          and a globally adjustable, optionally animated alpha.
        - Return plain data (`BackgroundRemovalResult`); never display,
          write, or persist anything itself.

    Explicitly out of scope (enforced by omission, not by stub methods):
        Opening a camera, calling `cv2.imshow`, hand/face/pose detection,
        gesture recognition, keyboard handling, UI/menus, and any
        application-level state machine. Those belong in layers that
        *use* this class.

    Thread safety:
        A single instance is safe for concurrent use from two roles: (a)
        one dedicated "frame-processing" thread calling `segment` /
        `replace_background`, and (b) any number of "control" threads
        (e.g., a UI thread) calling the configuration setters (`set_alpha`,
        `fade_in`, `fade_out`, `save_background`, `set_background_*`,
        `clear_background`, `reset`). All state-mutating methods and the
        segmentation backend call are guarded by an internal `RLock`. It is
        NOT safe to call `segment` or `replace_background` concurrently
        from multiple frame-processing threads on the same instance —
        construct one `BackgroundRemover` per processing thread in that
        case, mirroring MediaPipe's own single-graph-per-thread model.

    Performance notes:
        - `extract_foreground` deliberately avoids copying or zeroing
          pixel data; it defers all masking to a single blend pass to
          minimize per-frame allocations (see its docstring).
        - Background resize/generation results are cached and only
          recomputed when the source background or the live frame
          resolution changes, avoiding a resize on every single frame.
        - All mask refinement operates on a single-channel float32 array
          the size of the frame, not the 3-channel frame itself.

    Complexity:
        Per-frame cost is O(H x W) in the frame's pixel count for mask
        refinement and compositing, plus the cost of one forward pass of
        the underlying segmentation model. No per-frame allocation scales
        with hand/face count or history length; this class holds no
        temporal buffers beyond the single last input frame (for fade)
        and the single most recent mask/foreground/background.
    """

    def __init__(
        self,
        config: Optional[BackgroundRemovalConfig] = None,
        segmentation_backend: Optional[SegmentationBackend] = None,
    ) -> None:
        """Constructs the remover without allocating the segmentation model.

        Model allocation is deferred to `initialize()` by design (staged
        resource acquisition), so a host application can construct many
        `BackgroundRemover` instances cheaply (e.g., one per user session
        in a pooled server) and only pay model-initialization cost for the
        ones actually put to work.

        Args:
            config: Immutable tuning configuration. Defaults to
                `BackgroundRemovalConfig()` (see its field defaults) if not
                provided.
            segmentation_backend: Optional pre-built `SegmentationBackend`.
                Providing one skips MediaPipe entirely, which is the
                intended integration point for a future GPU-accelerated or
                custom backend (Dependency Inversion Principle). If
                omitted, `initialize()` constructs the default MediaPipe
                backend.

        Raises:
            InvalidConfigurationError: If `config` contains invalid values
                (propagated from `BackgroundRemovalConfig.__post_init__`).
        """
        self._config = config if config is not None else BackgroundRemovalConfig()
        self._injected_backend = segmentation_backend
        self._segmentation_backend: Optional[SegmentationBackend] = None
        self._initialized = False

        # WHY RLock (re-entrant) rather than Lock: `replace_background`
        # internally calls other locked public methods (e.g., `blend`,
        # `apply_fade`); a plain Lock would deadlock on the same thread.
        self._lock = threading.RLock()

        # Background configuration state.
        self._background_source = BackgroundSourceType.NONE
        self._background_asset: Optional[np.ndarray] = None  # SAVED_FRAME / IMAGE.
        self._background_color: Optional[BGRColor] = None  # COLOR.

        # Cached, resolution-matched background to avoid per-frame resizing.
        self._background_cache: Optional[np.ndarray] = None
        self._background_cache_shape: Optional[Tuple[int, int]] = None
        self._background_cache_source: Optional[BackgroundSourceType] = None

        # Alpha state. See the class docstring's distinction between the
        # static per-pixel "blend" alpha and the time-animated "fade" alpha.
        self._blend_alpha = DEFAULT_BLEND_ALPHA
        self._fade_alpha = DEFAULT_FADE_ALPHA
        self._fade_start_alpha = DEFAULT_FADE_ALPHA
        self._fade_target_alpha = DEFAULT_FADE_ALPHA
        self._fade_start_time = time.monotonic()

        # Per-frame transient state, used by `apply_fade` and `get_mask`.
        self._current_mask: Optional[SegmentationMask] = None
        self._current_foreground: Optional[Foreground] = None
        self._last_input_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def initialize(self) -> None:
        """Allocates the segmentation model exactly once.

        Idempotent: calling this more than once logs a debug message and
        returns without reallocating the model, so callers do not need to
        track initialization state themselves.

        Raises:
            BackgroundRemoverInitializationError: If model construction
                fails.
        """
        with self._lock:
            if self._initialized:
                logger.debug("initialize() called on an already-initialized instance.")
                return
            self._initialize_mediapipe()
            self._initialized = True

    def _initialize_mediapipe(self) -> None:
        """Constructs the segmentation backend (MediaPipe, unless injected).

        WHY isolated as its own method: mirrors this module's other heavy,
        failure-prone initialization steps, keeping `initialize()` readable
        and giving unit tests a single seam to mock.

        Raises:
            BackgroundRemoverInitializationError: If the backend fails to
                construct.
        """
        if self._injected_backend is not None:
            self._segmentation_backend = self._injected_backend
            logger.info("BackgroundRemover initialized with an injected backend.")
            return

        try:
            self._segmentation_backend = _MediaPipeSelfieSegmentationBackend(
                model_selection=self._config.model_selection
            )
            logger.info(
                "BackgroundRemover initialized with MediaPipe Selfie Segmentation "
                "(model_selection=%d, gpu_acceleration_requested=%s).",
                self._config.model_selection,
                self._config.enable_gpu_acceleration,
            )
        except BackgroundRemoverInitializationError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to initialize the segmentation backend.")
            raise BackgroundRemoverInitializationError(
                "Could not initialize the segmentation backend."
            ) from exc

    def _ensure_initialized(self) -> None:
        """Guards model-dependent methods against use-before-`initialize`.

        Raises:
            NotInitializedError: If `initialize()` has not been called (or
                the instance has since been closed).
        """
        if not self._initialized or self._segmentation_backend is None:
            raise NotInitializedError(
                "BackgroundRemover.initialize() must be called before this "
                "operation."
            )

    def close(self) -> None:
        """Releases the underlying segmentation model resources.

        WHY: native (C++) resources held by the segmentation graph are not
        promptly reclaimed by Python's garbage collector. Explicit release
        matters in long-running services that create and discard many
        `BackgroundRemover` instances (e.g., one per video call).
        """
        with self._lock:
            if self._segmentation_backend is not None:
                try:
                    self._segmentation_backend.close()
                    logger.info("BackgroundRemover resources released.")
                except Exception:  # noqa: BLE001
                    logger.exception("Error while closing the segmentation backend.")
                finally:
                    self._segmentation_backend = None
                    self._initialized = False

    def __enter__(self) -> "BackgroundRemover":
        """Enables `with BackgroundRemover(...) as remover:` usage."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Ensures resources are released when leaving a `with` block."""
        self.close()

    # ------------------------------------------------------------------ #
    # Frame validation & conversion
    # ------------------------------------------------------------------ #
    def _validate_frame(self, frame: np.ndarray) -> None:
        """Validates that `frame` is a well-formed BGR image.

        WHY: validating at the module boundary converts opaque native
        crashes deep inside OpenCV/MediaPipe into precise, typed,
        actionable errors for the caller.

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
        if frame.ndim != 3 or frame.shape[2] != NUM_BGR_CHANNELS:
            raise InvalidFrameError(
                "Frame must have shape (height, width, 3) representing a "
                f"BGR image; got shape {frame.shape}."
            )
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise InvalidFrameError("Frame has zero width or height.")

    def _convert_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Converts a BGR frame to RGB, as required by the segmentation model.

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
    # Segmentation
    # ------------------------------------------------------------------ #
    def generate_mask(self, frame: np.ndarray) -> SegmentationMask:
        """Runs the segmentation model and wraps its raw output.

        Args:
            frame: BGR frame, shape (height, width, 3), dtype uint8.

        Returns:
            SegmentationMask: Contains only `raw_mask`; `refined_mask` is
            `None`. Call `refine_mask` to populate it.

        Raises:
            InvalidFrameError: If `frame` fails validation or conversion.
            NotInitializedError: If `initialize()` has not been called.

        Notes:
            This method is a pure function of its input beyond reading the
            model: it does not mutate `self._current_mask`. Use `segment`
            for the stateful, orchestrated pipeline.
        """
        self._ensure_initialized()
        self._validate_frame(frame)

        rgb_frame = self._convert_to_rgb(frame)
        rgb_frame.flags.writeable = False
        with self._lock:
            raw_mask = self._segmentation_backend.process(rgb_frame)
        rgb_frame.flags.writeable = True

        return SegmentationMask(
            raw_mask=raw_mask,
            refined_mask=None,
            threshold=self._config.confidence_threshold,
            timestamp=time.monotonic(),
        )

    def refine_mask(self, mask: SegmentationMask) -> SegmentationMask:
        """Applies configured post-processing to a raw segmentation mask.

        Args:
            mask: A `SegmentationMask` as returned by `generate_mask`.

        Returns:
            SegmentationMask: A new instance with `refined_mask` populated.
            `SegmentationMask` is immutable, so the input is never mutated.

        Raises:
            InvalidFrameError: If `mask.raw_mask` is not a valid 2D array.

        Notes:
            Refinement stages (threshold, morphology, blur) are each
            individually toggleable via `BackgroundRemovalConfig`, so this
            method's cost scales with how much refinement is actually
            enabled — a headless low-power path can disable everything but
            thresholding.
        """
        if not isinstance(mask.raw_mask, np.ndarray) or mask.raw_mask.ndim != 2:
            raise InvalidFrameError(
                "SegmentationMask.raw_mask must be a 2D numpy.ndarray, got "
                f"{getattr(mask.raw_mask, 'shape', None)!r}."
            )

        refined = self._postprocess_mask(mask.raw_mask)
        return SegmentationMask(
            raw_mask=mask.raw_mask,
            refined_mask=refined,
            threshold=mask.threshold,
            timestamp=mask.timestamp,
        )

    def segment(self, frame: np.ndarray) -> SegmentationMask:
        """Orchestrates the full segmentation pipeline for one frame.

        Combines `generate_mask` and `refine_mask`, and additionally
        caches the result on `self` so `get_mask()` reflects the latest
        call. This is the method most callers should use directly;
        `generate_mask`/`refine_mask` remain available individually for
        callers building a custom pipeline (e.g., reusing a raw mask for
        multiple different refinement configs).

        Args:
            frame: BGR frame, shape (height, width, 3), dtype uint8.

        Returns:
            SegmentationMask: The fully refined mask for this frame.

        Raises:
            InvalidFrameError: If `frame` is invalid.
            NotInitializedError: If `initialize()` has not been called.
        """
        raw = self.generate_mask(frame)
        refined = self.refine_mask(raw)
        with self._lock:
            self._current_mask = refined
        return refined

    def _postprocess_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        """Runs the configured refinement pipeline on a raw mask.

        Args:
            raw_mask: Probability mask, shape (height, width), dtype
                float32, values in [0.0, 1.0].

        Returns:
            numpy.ndarray: Refined mask, same shape/dtype, values clipped
            to [0.0, 1.0].
        """
        working_mask = raw_mask

        if self._config.enable_threshold_binarization:
            working_mask = (
                working_mask >= self._config.confidence_threshold
            ).astype(np.float32)

        if self._config.enable_morphological_opening or (
            self._config.enable_morphological_closing
        ):
            working_mask = self._apply_morphology(working_mask)

        if self._config.enable_gaussian_blur or self._config.enable_edge_smoothing:
            working_mask = self._apply_blur(working_mask)

        return np.clip(working_mask, MIN_ALPHA, MAX_ALPHA).astype(np.float32)

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Applies configured opening/closing morphology to a mask.

        WHY opening before closing: opening first removes small isolated
        false-positive specks (which closing would otherwise preserve or
        even merge into the silhouette), then closing fills small holes
        inside the now-cleaner silhouette. Reversing the order tends to
        "heal" noise specks into the mask instead of removing them.

        Args:
            mask: Mask array, values in [0.0, 1.0], any float dtype.

        Returns:
            numpy.ndarray: Mask after morphology, dtype float32, values in
            [0.0, 1.0].
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._config.morphology_kernel_size, self._config.morphology_kernel_size),
        )
        mask_uint8 = (np.clip(mask, MIN_ALPHA, MAX_ALPHA) * BGR_MAX_VALUE).astype(
            np.uint8
        )

        if self._config.enable_morphological_opening:
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        if self._config.enable_morphological_closing:
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        return (mask_uint8.astype(np.float32)) / BGR_MAX_VALUE

    def _apply_blur(self, mask: np.ndarray) -> np.ndarray:
        """Applies configured Gaussian blur and/or edge-feathering to a mask.

        Args:
            mask: Mask array, values in [0.0, 1.0], dtype float32.

        Returns:
            numpy.ndarray: Blurred/feathered mask, dtype float32, values
            in [0.0, 1.0].
        """
        result = mask
        if self._config.enable_gaussian_blur:
            kernel_size = self._config.gaussian_blur_kernel_size
            result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
        if self._config.enable_edge_smoothing:
            # WHY a second, typically smaller pass: the Gaussian blur above
            # softens the mask broadly; this pass specifically targets
            # residual staircase/aliasing artifacts along the silhouette
            # edge for a cleaner anti-aliased composite.
            feather_size = self._config.edge_feather_kernel_size
            result = cv2.GaussianBlur(result, (feather_size, feather_size), 0)
        return result

    def get_mask(self) -> Optional[SegmentationMask]:
        """Returns the most recently computed segmentation mask.

        Returns:
            Optional[SegmentationMask]: The last mask produced by
            `segment`, or `None` if `segment`/`replace_background` has not
            been called yet (or `reset` was called since).
        """
        with self._lock:
            return self._current_mask

    # ------------------------------------------------------------------ #
    # Foreground / background computation
    # ------------------------------------------------------------------ #
    def extract_foreground(
        self, frame: np.ndarray, mask: SegmentationMask
    ) -> Foreground:
        """Builds the `Foreground` layer used for compositing.

        Args:
            frame: BGR frame the mask was computed from, shape
                (height, width, 3).
            mask: A `SegmentationMask`, ideally refined (see
                `SegmentationMask.active_mask`).

        Returns:
            Foreground: Wraps the frame and the active mask.

        Raises:
            InvalidFrameError: If `frame` and `mask.active_mask` have
                mismatched spatial dimensions.

        Notes:
            This method deliberately does NOT zero out background pixels
            or allocate a new masked copy of `frame`. Actual masking is
            applied once, per-pixel, inside `blend`. Producing a
            pre-masked copy here would mean every frame pays for two
            full-frame array operations (mask here, then blend later)
            instead of one — a meaningful cost at 30-60 FPS. `image` on
            the returned `Foreground` is the original frame array itself,
            not a copy; callers must not mutate it in place.
        """
        active_mask = mask.active_mask
        if active_mask.shape[:2] != frame.shape[:2]:
            raise InvalidFrameError(
                "Mask spatial dimensions "
                f"{active_mask.shape[:2]} do not match frame dimensions "
                f"{frame.shape[:2]}."
            )
        return self._compute_foreground(frame, active_mask)

    def _compute_foreground(
        self, frame: np.ndarray, active_mask: np.ndarray
    ) -> Foreground:
        """Builds a `Foreground` from a validated frame and mask.

        Args:
            frame: BGR frame, already validated against `active_mask`.
            active_mask: Mask array, values in [0.0, 1.0], dtype float32.

        Returns:
            Foreground: See `extract_foreground` for the "no-copy" design
            rationale.
        """
        return Foreground(image=frame, alpha_mask=active_mask, timestamp=time.monotonic())

    def _compute_background(self, frame_shape: Tuple[int, int, int]) -> Background:
        """Resolves the configured background source to the frame's resolution.

        Uses a shape/source-keyed cache so a saved frame, loaded image, or
        solid color is only resized/regenerated when the live frame
        resolution or the background source actually changes — not on
        every single call, which matters at real-time frame rates.

        Args:
            frame_shape: Shape tuple `(height, width, channels)` of the
                live frame currently being composited.

        Returns:
            Background: Resolved background image matching `frame_shape`.

        Raises:
            BackgroundNotAvailableError: If no background source is
                configured.
            NotImplementedError: If the configured source is `VIDEO`,
                which is reserved for a future extension.
        """
        target_shape = (frame_shape[0], frame_shape[1])

        if self._background_source == BackgroundSourceType.NONE:
            raise BackgroundNotAvailableError(
                "No background is configured; call save_background, "
                "set_background_image, or set_background_color first."
            )

        cache_hit = (
            self._background_cache is not None
            and self._background_cache_shape == target_shape
            and self._background_cache_source == self._background_source
        )
        if cache_hit:
            return Background(
                image=self._background_cache,
                source=self._background_source,
                timestamp=time.monotonic(),
            )

        if self._background_source == BackgroundSourceType.VIDEO:
            raise NotImplementedError(
                "Video backgrounds are not yet supported; reserved for a "
                "future extension of BackgroundRemover."
            )

        if self._background_source == BackgroundSourceType.COLOR:
            resolved = self._build_solid_color_background(target_shape)
        else:  # SAVED_FRAME or IMAGE both resolve from `_background_asset`.
            resolved = self._resize_background_asset(target_shape)

        self._background_cache = resolved
        self._background_cache_shape = target_shape
        self._background_cache_source = self._background_source

        return Background(
            image=resolved, source=self._background_source, timestamp=time.monotonic()
        )

    def _build_solid_color_background(
        self, target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Builds a solid-color BGR image at the given resolution.

        Args:
            target_shape: `(height, width)` of the desired image.

        Returns:
            numpy.ndarray: Shape `(height, width, 3)`, dtype uint8, filled
            with `self._background_color`.

        Raises:
            BackgroundNotAvailableError: If no color has been configured
                (defensive; should be unreachable given `set_background_color`
                is the only way to set source=COLOR).
        """
        if self._background_color is None:
            raise BackgroundNotAvailableError(
                "Background source is COLOR but no color is configured."
            )
        height, width = target_shape
        return np.full(
            (height, width, NUM_BGR_CHANNELS), self._background_color, dtype=np.uint8
        )

    def _resize_background_asset(self, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resizes the stored background asset (saved frame or image) to fit.

        Args:
            target_shape: `(height, width)` of the desired image.

        Returns:
            numpy.ndarray: Shape `(height, width, 3)`, dtype uint8.

        Raises:
            BackgroundNotAvailableError: If no asset is configured
                (defensive; mirrors `_build_solid_color_background`).
        """
        if self._background_asset is None:
            raise BackgroundNotAvailableError(
                "Background source requires a stored frame/image but none "
                "is configured."
            )
        height, width = target_shape
        if self._background_asset.shape[:2] == (height, width):
            return self._background_asset
        return cv2.resize(
            self._background_asset,
            (width, height),
            interpolation=self._config.background_resize_interpolation,
        )

    def _invalidate_background_cache(self) -> None:
        """Clears the resolution-matched background cache.

        WHY a dedicated method: every setter that changes the background
        source must invalidate the cache identically; centralizing this
        avoids the DRY violation of repeating three attribute resets in
        four different setter methods.
        """
        self._background_cache = None
        self._background_cache_shape = None
        self._background_cache_source = None

    # ------------------------------------------------------------------ #
    # Background configuration (public setters)
    # ------------------------------------------------------------------ #
    def save_background(self, frame: np.ndarray) -> None:
        """Captures a clean background plate from the live feed.

        Intended to be called with a frame containing no person (e.g.,
        during an onboarding "please step out of frame" moment).

        Args:
            frame: BGR frame, shape (height, width, 3), dtype uint8.

        Raises:
            InvalidFrameError: If `frame` fails validation.

        Notes:
            Semantically distinct from `set_background_image` (a static
            asset) even though both are stored the same way internally:
            this method documents the "captured from the live camera"
            provenance, which matters for features like re-capture
            prompts or background staleness detection.
        """
        self.set_background_frame(frame)

    def set_background_frame(self, frame: np.ndarray) -> None:
        """Sets the virtual background to an explicit BGR frame.

        Args:
            frame: BGR frame, shape (height, width, 3), dtype uint8.

        Raises:
            InvalidFrameError: If `frame` fails validation.

        Notes:
            The frame is copied before storage. Callers (particularly
            those reading from `cv2.VideoCapture`, which frequently reuses
            its internal buffer across reads) must not rely on their own
            reference remaining valid; this method guarantees the stored
            copy is independent of the caller's buffer.
        """
        self._validate_frame(frame)
        with self._lock:
            self._background_asset = frame.copy()
            self._background_color = None
            self._background_source = BackgroundSourceType.SAVED_FRAME
            self._invalidate_background_cache()
        logger.info("Background set from a captured frame.")

    def set_background_image(self, image: np.ndarray) -> None:
        """Sets the virtual background to a static loaded image.

        Args:
            image: BGR image, shape (height, width, 3), dtype uint8.

        Raises:
            InvalidFrameError: If `image` fails validation.
        """
        self._validate_frame(image)
        with self._lock:
            self._background_asset = image.copy()
            self._background_color = None
            self._background_source = BackgroundSourceType.IMAGE
            self._invalidate_background_cache()
        logger.info("Background set from a static image (shape=%s).", image.shape)

    def set_background_color(self, color: BGRColor) -> None:
        """Sets the virtual background to a solid BGR color.

        Args:
            color: A 3-tuple of integers `(blue, green, red)`, each in
                [0, 255].

        Raises:
            InvalidConfigurationError: If `color` is not a valid 3-tuple of
                integers within range.
        """
        self._validate_color(color)
        with self._lock:
            self._background_color = tuple(color)  # type: ignore[assignment]
            self._background_asset = None
            self._background_source = BackgroundSourceType.COLOR
            self._invalidate_background_cache()
        logger.info("Background set to solid color BGR=%s.", color)

    def _validate_color(self, color: BGRColor) -> None:
        """Validates a BGR color tuple.

        Args:
            color: Candidate color.

        Raises:
            InvalidConfigurationError: If `color` is not a 3-tuple of
                integers within [0, 255].
        """
        if (
            not isinstance(color, (tuple, list))
            or len(color) != NUM_BGR_CHANNELS
            or not all(isinstance(channel, int) for channel in color)
            or not all(BGR_MIN_VALUE <= channel <= BGR_MAX_VALUE for channel in color)
        ):
            raise InvalidConfigurationError(
                f"color must be a 3-tuple of ints in [{BGR_MIN_VALUE}, "
                f"{BGR_MAX_VALUE}], got {color!r}."
            )

    def clear_background(self) -> None:
        """Removes the configured virtual background entirely.

        After this call, `has_saved_background()` returns False and
        `replace_background` falls back to pass-through compositing.
        """
        with self._lock:
            self._background_asset = None
            self._background_color = None
            self._background_source = BackgroundSourceType.NONE
            self._invalidate_background_cache()
        logger.info("Background cleared.")

    def has_saved_background(self) -> bool:
        """Reports whether a virtual background is currently configured.

        Returns:
            bool: True if any background source other than `NONE` is set.
        """
        with self._lock:
            return self._background_source != BackgroundSourceType.NONE

    # ------------------------------------------------------------------ #
    # Alpha blending & fading
    # ------------------------------------------------------------------ #
    def blend(self, foreground: Foreground, background: Background, alpha: float) -> np.ndarray:
        """Composites a foreground and background using per-pixel alpha.

        Args:
            foreground: A `Foreground` as returned by `extract_foreground`.
            background: A `Background` as returned by `_compute_background`
                (typically obtained indirectly through `replace_background`).
            alpha: Global blend strength in [0.0, 1.0], multiplied into the
                per-pixel foreground mask. 1.0 uses the mask as-is (full
                cutout strength); 0.0 produces the background image alone.

        Returns:
            numpy.ndarray: Composited BGR frame, same shape/dtype as
            `foreground.image`.

        Raises:
            InvalidConfigurationError: If `alpha` is outside [0.0, 1.0].
            InvalidFrameError: If `foreground.image` and `background.image`
                have mismatched shapes.
        """
        if not MIN_ALPHA <= alpha <= MAX_ALPHA:
            raise InvalidConfigurationError(
                f"alpha must be within [{MIN_ALPHA}, {MAX_ALPHA}], got {alpha!r}."
            )
        if foreground.image.shape != background.image.shape:
            raise InvalidFrameError(
                "Foreground and background shapes must match; got "
                f"{foreground.image.shape} vs {background.image.shape}."
            )

        combined_alpha = self._create_alpha_mask(foreground.alpha_mask, alpha)
        foreground_f = foreground.image.astype(np.float32)
        background_f = background.image.astype(np.float32)

        composited = (
            foreground_f * combined_alpha + background_f * (1.0 - combined_alpha)
        )
        return np.clip(composited, BGR_MIN_VALUE, BGR_MAX_VALUE).astype(np.uint8)

    def _create_alpha_mask(self, mask: np.ndarray, alpha: float) -> np.ndarray:
        """Builds a 3-channel, broadcastable alpha map for compositing.

        Args:
            mask: Per-pixel foreground weight, shape (height, width),
                values in [0.0, 1.0].
            alpha: Global scalar blend strength in [0.0, 1.0].

        Returns:
            numpy.ndarray: Shape (height, width, 1), dtype float32, values
            in [0.0, 1.0], equal to `mask * alpha` clipped to range. The
            trailing singleton dimension lets it broadcast directly against
            a (height, width, 3) image without an explicit channel-stack.
        """
        combined = np.clip(mask * alpha, MIN_ALPHA, MAX_ALPHA).astype(np.float32)
        return combined[..., np.newaxis]

    def set_alpha(self, value: float) -> None:
        """Immediately sets the static per-pixel blend strength.

        Unlike `fade_in`/`fade_out`, this is a hard, instantaneous change
        with no animation — use it for direct transparency control (e.g.,
        a UI slider), and use the fade methods for smooth enable/disable
        transitions.

        Args:
            value: Blend strength in [0.0, 1.0].

        Raises:
            InvalidConfigurationError: If `value` is outside [0.0, 1.0].
        """
        if not MIN_ALPHA <= value <= MAX_ALPHA:
            raise InvalidConfigurationError(
                f"alpha value must be within [{MIN_ALPHA}, {MAX_ALPHA}], got {value!r}."
            )
        with self._lock:
            self._blend_alpha = float(value)
        logger.debug("Blend alpha set to %.3f.", value)

    def fade_in(self) -> None:
        """Starts an animated transition of the overall effect toward fully visible.

        The transition is advanced over time by `apply_fade`, which must
        be called once per frame (it is called automatically as the final
        step of `replace_background`).

        Returns:
            None.
        """
        with self._lock:
            self._fade_start_alpha = self._fade_alpha
            self._fade_target_alpha = MAX_ALPHA
            self._fade_start_time = time.monotonic()
        logger.debug("Fade-in started from alpha=%.3f.", self._fade_start_alpha)

    def fade_out(self) -> None:
        """Starts an animated transition of the overall effect toward fully hidden.

        When the fade completes (`apply_fade` reaches alpha 0.0), the
        composited output converges to the original, unmodified input
        frame — i.e., the background-replacement effect fully disengages.

        Returns:
            None.
        """
        with self._lock:
            self._fade_start_alpha = self._fade_alpha
            self._fade_target_alpha = MIN_ALPHA
            self._fade_start_time = time.monotonic()
        logger.debug("Fade-out started from alpha=%.3f.", self._fade_start_alpha)

    def apply_fade(self, frame: np.ndarray) -> np.ndarray:
        """Blends a processed frame with the original input using the fade alpha.

        Advances the internal fade animation based on elapsed wall-clock
        time (not frame count), so transitions take a consistent amount of
        real time regardless of the host application's current frame
        rate, then linearly interpolates between the last raw input frame
        (`self._last_input_frame`) and `frame` using the resulting alpha.

        Args:
            frame: The already background-replaced (or otherwise
                processed) BGR frame to fade toward/away from.

        Returns:
            numpy.ndarray: The faded BGR frame. If no prior input frame is
            known (i.e., `segment`/`replace_background` has never been
            called), `frame` is returned unchanged and a warning is logged,
            since there is nothing to fade against.

        Raises:
            InvalidFrameError: If `frame` fails validation.

        Notes:
            When no fade is in progress (current alpha already equals the
            target), this still applies the current *static* alpha value,
            which is what enables `set_alpha`-style held partial
            transparency in addition to animated transitions.
        """
        self._validate_frame(frame)

        with self._lock:
            if self._last_input_frame is None:
                logger.warning(
                    "apply_fade called with no prior input frame; returning "
                    "frame unchanged."
                )
                return frame

            self._advance_fade_alpha()
            current_alpha = self._fade_alpha
            reference_frame = self._last_input_frame

        if reference_frame.shape != frame.shape:
            raise InvalidFrameError(
                "apply_fade frame shape does not match the last processed "
                f"input frame shape; got {frame.shape} vs {reference_frame.shape}."
            )

        if current_alpha >= MAX_ALPHA:
            return frame
        if current_alpha <= MIN_ALPHA:
            return reference_frame

        return cv2.addWeighted(
            frame, current_alpha, reference_frame, 1.0 - current_alpha, 0.0
        )

    def _advance_fade_alpha(self) -> None:
        """Advances `self._fade_alpha` toward its target based on elapsed time.

        WHY time-based rather than a fixed per-call increment: a per-call
        increment would make fade duration depend on the host's current
        frame rate (a fade would visibly take longer on a slower device),
        which is inconsistent product behavior across hardware tiers.

        Returns:
            None.
        """
        if self._fade_alpha == self._fade_target_alpha:
            return

        elapsed_seconds = time.monotonic() - self._fade_start_time
        speed = self._config.effective_alpha_speed
        max_delta = speed * elapsed_seconds

        if self._fade_target_alpha > self._fade_start_alpha:
            self._fade_alpha = min(
                self._fade_start_alpha + max_delta, self._fade_target_alpha
            )
        else:
            self._fade_alpha = max(
                self._fade_start_alpha - max_delta, self._fade_target_alpha
            )

    # ------------------------------------------------------------------ #
    # High-level orchestration
    # ------------------------------------------------------------------ #
    def replace_background(self, frame: np.ndarray) -> BackgroundRemovalResult:
        """Runs the full segmentation-to-composite pipeline for one frame.

        Orchestrates `segment`, `extract_foreground`, background
        resolution, `blend`, and `apply_fade` into a single call. This is
        the primary entry point most real-time callers should use; the
        lower-level methods remain public for callers assembling a custom
        pipeline (e.g., reusing one mask across multiple candidate
        backgrounds).

        Args:
            frame: BGR frame, shape (height, width, 3), dtype uint8, as
                produced by the caller's own capture pipeline. This method
                never opens or reads from a capture device.

        Returns:
            BackgroundRemovalResult: Structured result. If no background is
            configured, `processed_frame` is the (fade-adjusted) original
            frame and `background`/`background_available` reflect that.

        Raises:
            InvalidFrameError: If `frame` fails validation.
            NotInitializedError: If `initialize()` has not been called.

        Notes:
            On an unexpected internal failure during compositing (but
            after successful segmentation), this method logs the error and
            falls back to returning the original frame rather than
            raising, so a single bad frame cannot crash a real-time loop
            processing 30-60 frames per second. Segmentation/validation
            failures are still raised, since those indicate a genuine
            misuse of the API (bad frame, uninitialized model) rather than
            a transient compositing issue.
        """
        start_time = time.monotonic()
        self._validate_frame(frame)
        self._ensure_initialized()

        with self._lock:
            self._last_input_frame = frame

        mask = self.segment(frame)
        foreground = self.extract_foreground(frame, mask)
        with self._lock:
            self._current_foreground = foreground

        background: Optional[Background] = None
        background_available = self.has_saved_background()

        try:
            if background_available:
                background = self._compute_background(frame.shape)
                composited = self.blend(foreground, background, self._blend_alpha)
            else:
                composited = frame
            processed_frame = self.apply_fade(composited)
        except Exception:  # noqa: BLE001 - isolate the real-time loop from crashes.
            logger.exception(
                "Unexpected failure while compositing; returning original frame."
            )
            processed_frame = frame

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        result = BackgroundRemovalResult(
            processed_frame=processed_frame,
            mask=mask,
            foreground=foreground,
            background=background,
            alpha=self._fade_alpha,
            background_available=background_available,
            timestamp=time.monotonic(),
            processing_time_ms=elapsed_ms,
        )

        logger.debug(
            "replace_background() finished: background_available=%s, "
            "elapsed_ms=%.2f",
            background_available,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # State management
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clears per-frame transient state and alpha animation state.

        Resets the cached mask/foreground, the last input frame used by
        `apply_fade`, and both alpha values back to their defaults. Does
        NOT affect model initialization (`close`/`initialize` own that) or
        the configured virtual background (`clear_background` owns that) —
        those are independent lifecycles by design, so a caller can reset
        transient per-session state without having to re-initialize the
        model or re-configure their background.

        Returns:
            None.
        """
        with self._lock:
            self._current_mask = None
            self._current_foreground = None
            self._last_input_frame = None
            self._blend_alpha = DEFAULT_BLEND_ALPHA
            self._fade_alpha = DEFAULT_FADE_ALPHA
            self._fade_start_alpha = DEFAULT_FADE_ALPHA
            self._fade_target_alpha = DEFAULT_FADE_ALPHA
            self._fade_start_time = time.monotonic()
        logger.debug("BackgroundRemover transient state reset.")