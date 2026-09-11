"""Reusable mathematical, geometric, interpolation, normalization, and
gesture-heuristic primitives for real-time computer vision pipelines.

This module is the shared numerical foundation beneath tracking and
gesture-recognition subsystems (hand tracking, face tracking, body
tracking, AR object placement, and analytics). It is deliberately kept
free of any dependency on a specific perception framework or rendering
stack so that the same math can be reused unmodified across products,
platforms, and even languages if ported later (Open/Closed Principle:
open for reuse, closed for framework-specific modification).

WHY this module exists as a hard boundary:
    Mixing "how do I compute a distance" with "how do I open a camera"
    or "which model detects a hand" is a classic Clean Architecture
    violation. Once math and perception are tangled, the math cannot be
    unit tested without a camera, a GPU, or a native SIMD build of a
    detection model, which makes CI slow, flaky, and expensive at scale.
    Every function in this module is a pure function (or a pure
    function operating on an explicit, versionable dataclass) with no
    hidden state, no I/O, and no side effects, which makes it trivially
    testable, trivially parallelizable, and trivially reusable.

Hard constraints (enforced by review, not just convention):
    * MUST NOT import cv2.
    * MUST NOT import mediapipe.
    * MUST NOT open, read, or reference a camera or video stream.
    * MUST NOT call any drawing/rendering/UI function.
    * MUST NOT perform hand, face, or body detection.
    * MUST NOT perform image segmentation.
    * MUST NOT hold application or session state.
    * MUST NOT contain business logic specific to any one product.

Typical usage:
    from utils import calculate_distance, Point2D, clamp

    a = Point2D(x=0.1, y=0.2)
    b = Point2D(x=0.4, y=0.6)
    distance = calculate_distance(a, b)
    bounded = clamp(distance, minimum=0.0, maximum=1.0)

Author: Computer Vision Platform Team
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, List, Optional, Sequence, Tuple, Union

import numpy as np

# --------------------------------------------------------------------------- #
# Module-level logger.
#
# WHY: this module runs inside real-time loops (30-60 FPS) across many
# host applications. print() cannot be silenced, routed, leveled, or
# correlated with the rest of a service's telemetry. A named logger lets
# every downstream consumer (mobile bridges, backend batch jobs, desktop
# apps) control verbosity independently without touching this file.
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)


# =========================================================================== #
# Constants
#
# WHY centralized here: magic numbers scattered across a real-time CV
# codebase are a maintenance and correctness hazard. A single source of
# truth for tunable thresholds means a product team can retune behavior
# (e.g., pinch sensitivity for a new sensor) in one place, and a reviewer
# can see every default at a glance instead of hunting through call sites.
# =========================================================================== #
DEFAULT_PINCH_THRESHOLD: Final[float] = 0.05
"""Default normalized distance below which thumb and index tips are
considered pinched together. Tuned empirically against MediaPipe Hands
normalized landmark space; re-tune if the upstream landmark model changes."""

DEFAULT_FADE_SPEED: Final[float] = 2.5
"""Default alpha units-per-second for fade_in/fade_out helpers."""

DEFAULT_ALPHA_MIN: Final[float] = 0.0
"""Lower bound of the normalized alpha/opacity range."""

DEFAULT_ALPHA_MAX: Final[float] = 1.0
"""Upper bound of the normalized alpha/opacity range."""

EPSILON: Final[float] = 1e-9
"""Small value used to guard against division-by-zero and to compare
floating-point numbers for near-equality without exact-equality bugs."""

MAX_FLOAT: Final[float] = sys.float_info.max
"""Largest representable finite float on this platform. Used as a safe
"positive infinity" sentinel for min-reduction algorithms (e.g., seeding
a running minimum) without importing math.inf semantics everywhere."""

MIN_FLOAT: Final[float] = -sys.float_info.max
"""Smallest representable finite float on this platform. Used as a safe
"negative infinity" sentinel for max-reduction algorithms."""

DEFAULT_SMOOTHING_FACTOR: Final[float] = 0.35
"""Default weight applied to the new sample in exponential smoothing and
low-pass filtering. Lower values favor stability; higher values favor
responsiveness. 0.35 is a balanced default for 30 FPS hand tracking."""

DEFAULT_INTERPOLATION_SPEED: Final[float] = 8.0
"""Default per-second interpolation speed used by frame-rate-independent
lerp-based smoothing (see `update_alpha`)."""

DEFAULT_MOVING_AVERAGE_WINDOW: Final[int] = 5
"""Default window size, in samples, for the moving-average helper."""

DEFAULT_STABILITY_THRESHOLD: Final[float] = 0.01
"""Default normalized-distance threshold below which a landmark set is
considered stable (not actively moving) between two frames."""


# =========================================================================== #
# Enums
# =========================================================================== #
class GestureType(str, Enum):
    """Coarse, framework-agnostic gesture classification.

    WHY an Enum instead of raw strings: prevents typos (e.g., "Pinch" vs
    "pinch") from silently breaking downstream gesture-dispatch logic,
    and gives static type-checkers the ability to catch invalid gesture
    references at development time instead of at runtime in production.

    WHY it lives in the pure-math layer rather than a detection module:
    gesture *classification from geometry* (this file) is a mathematical
    concern; gesture *detection from pixels* (which model, which
    landmarks) is a perception concern that belongs to a higher layer.
    This enum is the shared vocabulary between those layers.
    """

    UNKNOWN = "unknown"
    PINCH = "pinch"
    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    POINTING = "pointing"
    CUSTOM = "custom"


# =========================================================================== #
# Dataclasses
#
# WHY dataclasses instead of bare tuples/dicts everywhere: tuples and
# dicts carry no semantic meaning at the type level ("was that (x, y) or
# (y, x)?"), cannot be validated on construction, and cannot evolve
# without breaking every call site via positional-argument confusion.
# Frozen dataclasses give us immutability (safe to share across threads
# and across frames without defensive copying), explicit field names,
# and a stable public contract that can gain new *optional* fields
# without breaking existing consumers.
# =========================================================================== #
@dataclass(frozen=True)
class Point2D:
    """An immutable point in 2D space.

    Attributes:
        x: X-coordinate. Unit is caller-defined (normalized [0, 1] or
            pixel space); this dataclass is unit-agnostic by design so
            it can represent either without duplication.
        y: Y-coordinate, same unit convention as `x`.
    """

    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]:
        """Returns the point as a plain ``(x, y)`` tuple.

        Returns:
            Tuple[float, float]: The point's coordinates.
        """
        return (self.x, self.y)


@dataclass(frozen=True)
class Point3D:
    """An immutable point in 3D space.

    Attributes:
        x: X-coordinate.
        y: Y-coordinate.
        z: Z-coordinate (depth). Unit is caller-defined; for landmark
            models such as MediaPipe Hands this is a roughly wrist-
            relative, non-metric depth value.
    """

    x: float
    y: float
    z: float

    def as_tuple(self) -> Tuple[float, float, float]:
        """Returns the point as a plain ``(x, y, z)`` tuple.

        Returns:
            Tuple[float, float, float]: The point's coordinates.
        """
        return (self.x, self.y, self.z)

    def to_2d(self) -> Point2D:
        """Projects this point onto the XY plane, discarding depth.

        Returns:
            Point2D: The (x, y) projection of this point.
        """
        return Point2D(x=self.x, y=self.y)


@dataclass(frozen=True)
class BoundingBox:
    """An immutable axis-aligned bounding box.

    Attributes:
        x_min: Left edge.
        y_min: Top edge.
        x_max: Right edge.
        y_max: Bottom edge.

    Raises:
        ValueError: If constructed with `x_max < x_min` or
            `y_max < y_min`, since such a box is geometrically invalid
            and would silently corrupt any downstream area/overlap math.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError(
                "Invalid BoundingBox: max coordinates must be >= min "
                f"coordinates (got x=[{self.x_min}, {self.x_max}], "
                f"y=[{self.y_min}, {self.y_max}])."
            )

    @property
    def width(self) -> float:
        """float: Bounding box width."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """float: Bounding box height."""
        return self.y_max - self.y_min

    @property
    def center(self) -> Point2D:
        """Point2D: Centroid of the bounding box."""
        return Point2D(
            x=(self.x_min + self.x_max) / 2.0,
            y=(self.y_min + self.y_max) / 2.0,
        )

    @property
    def area(self) -> float:
        """float: Bounding box area (`width * height`)."""
        return self.width * self.height


@dataclass(frozen=True)
class Vector2D:
    """An immutable 2D vector.

    Attributes:
        dx: Component along the x-axis.
        dy: Component along the y-axis.
    """

    dx: float
    dy: float

    def as_tuple(self) -> Tuple[float, float]:
        """Returns the vector as a plain ``(dx, dy)`` tuple.

        Returns:
            Tuple[float, float]: The vector's components.
        """
        return (self.dx, self.dy)


@dataclass(frozen=True)
class Vector3D:
    """An immutable 3D vector.

    Attributes:
        dx: Component along the x-axis.
        dy: Component along the y-axis.
        dz: Component along the z-axis.
    """

    dx: float
    dy: float
    dz: float

    def as_tuple(self) -> Tuple[float, float, float]:
        """Returns the vector as a plain ``(dx, dy, dz)`` tuple.

        Returns:
            Tuple[float, float, float]: The vector's components.
        """
        return (self.dx, self.dy, self.dz)


@dataclass(frozen=True)
class Velocity:
    """Instantaneous velocity of a tracked point, in units per second.

    WHY a dedicated type instead of a raw float or tuple: velocity is
    directional and time-scaled, and conflating it with a plain
    `Vector2D` would let callers accidentally treat a per-second rate as
    a per-frame displacement (a common, hard-to-spot real-time bug).

    Attributes:
        vx: Velocity component along the x-axis, units per second.
        vy: Velocity component along the y-axis, units per second.
    """

    vx: float
    vy: float

    @property
    def magnitude(self) -> float:
        """float: Speed (vector magnitude of the velocity), units/second."""
        return math.hypot(self.vx, self.vy)


@dataclass(frozen=True)
class Acceleration:
    """Instantaneous acceleration of a tracked point, in units per
    second squared.

    Attributes:
        ax: Acceleration component along the x-axis, units per second^2.
        ay: Acceleration component along the y-axis, units per second^2.
    """

    ax: float
    ay: float

    @property
    def magnitude(self) -> float:
        """float: Magnitude of the acceleration vector, units/second^2."""
        return math.hypot(self.ax, self.ay)


@dataclass(frozen=True)
class GestureResult:
    """Outcome of a geometry-based gesture heuristic.

    WHY bundling gesture + confidence + metadata instead of returning a
    bare enum: gesture heuristics are inherently approximate, and a
    downstream consumer (e.g., a gesture-triggered UI action) typically
    needs a confidence score to decide whether to act, debounce, or
    ignore a borderline classification.

    Attributes:
        gesture: The classified `GestureType`.
        confidence: Heuristic confidence in ``[0.0, 1.0]``.
        metadata: Optional free-form auxiliary data (e.g., the raw
            pinch distance that triggered a `PINCH` classification),
            kept as a plain dict so this module never needs to grow new
            dataclass fields for every future gesture's debug payload.
    """

    gesture: GestureType
    confidence: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}."
            )


@dataclass(frozen=True)
class InterpolationResult:
    """Outcome of a single frame-rate-independent interpolation step.

    WHY: alpha/value fades are computed every frame across many UI and
    AR overlays. Returning both the new value and whether it has
    converged lets callers stop scheduling further updates (saving CPU)
    without re-deriving convergence from raw floats at every call site.

    Attributes:
        value: The interpolated value after this step.
        has_converged: True if `value` is within `EPSILON` of its
            target and no further interpolation is necessary.
    """

    value: float
    has_converged: bool


# Type aliases for functions that accept either a typed point/vector or
# a plain tuple, so this module is ergonomic to call from both fully
# typed code and lightweight scripts/tests without forcing allocation
# of dataclass instances everywhere.
PointLike2D = Union[Point2D, Tuple[float, float]]
PointLike3D = Union[Point3D, Tuple[float, float, float]]
VectorLike2D = Union[Vector2D, Tuple[float, float]]


# =========================================================================== #
# Internal coercion helpers
#
# WHY: every public function below accepts flexible "point-like" inputs
# for ergonomics, but flexible inputs multiply the number of code paths
# that must validate correctly. Centralizing coercion here means the
# validation logic exists exactly once and every public function stays
# short and auditable.
# =========================================================================== #
def _coerce_xy(point: PointLike2D, *, param_name: str) -> Tuple[float, float]:
    """Coerces a `Point2D` or a 2-tuple into a plain `(x, y)` tuple.

    Args:
        point: A `Point2D` instance, or a 2-tuple/2-list of numbers.
        param_name: Name of the offending parameter, used to produce an
            actionable error message.

    Returns:
        Tuple[float, float]: The coerced `(x, y)` pair.

    Raises:
        TypeError: If `point` is neither a `Point2D` nor a 2-length
            sequence of numbers.
    """
    if isinstance(point, Point2D):
        return point.x, point.y
    if isinstance(point, (tuple, list)) and len(point) == 2:
        try:
            return float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{param_name} must contain numeric values, got {point!r}."
            ) from exc
    raise TypeError(
        f"{param_name} must be a Point2D or a 2-length (x, y) sequence, "
        f"got {type(point).__name__}."
    )


def _coerce_xyz(point: PointLike3D, *, param_name: str) -> Tuple[float, float, float]:
    """Coerces a `Point3D` or a 3-tuple into a plain `(x, y, z)` tuple.

    Args:
        point: A `Point3D` instance, or a 3-tuple/3-list of numbers.
        param_name: Name of the offending parameter, used to produce an
            actionable error message.

    Returns:
        Tuple[float, float, float]: The coerced `(x, y, z)` triple.

    Raises:
        TypeError: If `point` is neither a `Point3D` nor a 3-length
            sequence of numbers.
    """
    if isinstance(point, Point3D):
        return point.x, point.y, point.z
    if isinstance(point, (tuple, list)) and len(point) == 3:
        try:
            return float(point[0]), float(point[1]), float(point[2])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{param_name} must contain numeric values, got {point!r}."
            ) from exc
    raise TypeError(
        f"{param_name} must be a Point3D or a 3-length (x, y, z) sequence, "
        f"got {type(point).__name__}."
    )


def _require_non_empty(values: Sequence, *, param_name: str) -> None:
    """Validates that a sequence is non-empty.

    Args:
        values: The sequence to validate.
        param_name: Name of the offending parameter, for the error
            message.

    Raises:
        ValueError: If `values` is empty.
    """
    if len(values) == 0:
        raise ValueError(f"{param_name} must not be empty.")


# =========================================================================== #
# Distance & vector mathematics
# =========================================================================== #
def calculate_distance(point1: PointLike2D, point2: PointLike2D) -> float:
    """Computes the Euclidean distance between two 2D points.

    WHY it exists: Euclidean distance is the single most common
    primitive in gesture and proximity heuristics (pinch detection,
    hand-to-hand distance, cursor hit-testing). Centralizing it avoids
    N slightly-different reimplementations across consumers.

    Args:
        point1: First point, as `Point2D` or `(x, y)`.
        point2: Second point, as `Point2D` or `(x, y)`.

    Returns:
        float: The non-negative Euclidean distance between the points.

    Raises:
        TypeError: If either point is not a valid point-like input.

    Complexity:
        O(1) time, O(1) space.
    """
    x1, y1 = _coerce_xy(point1, param_name="point1")
    x2, y2 = _coerce_xy(point2, param_name="point2")
    return math.hypot(x2 - x1, y2 - y1)


def calculate_squared_distance(point1: PointLike2D, point2: PointLike2D) -> float:
    """Computes the squared Euclidean distance between two 2D points.

    WHY it exists: many hot-path comparisons (e.g., "is this the
    nearest of N candidates?") only need relative ordering, not the
    true distance. Skipping `sqrt` avoids a measurable per-call cost
    when this runs thousands of times per frame across many landmarks.

    Args:
        point1: First point, as `Point2D` or `(x, y)`.
        point2: Second point, as `Point2D` or `(x, y)`.

    Returns:
        float: The non-negative squared Euclidean distance.

    Raises:
        TypeError: If either point is not a valid point-like input.

    Complexity:
        O(1) time, O(1) space.
    """
    x1, y1 = _coerce_xy(point1, param_name="point1")
    x2, y2 = _coerce_xy(point2, param_name="point2")
    dx = x2 - x1
    dy = y2 - y1
    return dx * dx + dy * dy


def calculate_angle(
    point_a: PointLike2D, point_b: PointLike2D, point_c: PointLike2D
) -> float:
    """Computes the interior angle at vertex `point_b`, in degrees.

    The angle is measured between ray `point_b -> point_a` and ray
    `point_b -> point_c`. This is the standard formulation for joint
    flexion angles (e.g., a finger's PIP joint angle between its
    adjacent bone segments).

    Args:
        point_a: First outer point, as `Point2D` or `(x, y)`.
        point_b: Vertex point (where the angle is measured), as
            `Point2D` or `(x, y)`.
        point_c: Second outer point, as `Point2D` or `(x, y)`.

    Returns:
        float: The interior angle in degrees, in the range
        ``[0.0, 180.0]``.

    Raises:
        TypeError: If any point is not a valid point-like input.
        ValueError: If `point_a` or `point_c` coincides with `point_b`,
            making the angle undefined (zero-length ray).

    Complexity:
        O(1) time, O(1) space.
    """
    ax, ay = _coerce_xy(point_a, param_name="point_a")
    bx, by = _coerce_xy(point_b, param_name="point_b")
    cx, cy = _coerce_xy(point_c, param_name="point_c")

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    ba_length = math.hypot(*ba)
    bc_length = math.hypot(*bc)
    if ba_length < EPSILON or bc_length < EPSILON:
        raise ValueError(
            "calculate_angle received a degenerate ray (point_a or "
            "point_c coincides with point_b); the angle is undefined."
        )

    cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / (ba_length * bc_length)
    # Numerical noise can push cosine slightly outside [-1, 1] near 0
    # and 180 degrees; clamp defensively before calling acos to avoid a
    # ValueError from math domain errors in production.
    cosine = clamp(cosine, minimum=-1.0, maximum=1.0)
    return math.degrees(math.acos(cosine))


def calculate_vector(point1: PointLike2D, point2: PointLike2D) -> Vector2D:
    """Computes the vector pointing from `point1` to `point2`.

    Args:
        point1: Origin point, as `Point2D` or `(x, y)`.
        point2: Destination point, as `Point2D` or `(x, y)`.

    Returns:
        Vector2D: The displacement vector `point2 - point1`.

    Raises:
        TypeError: If either point is not a valid point-like input.

    Complexity:
        O(1) time, O(1) space.
    """
    x1, y1 = _coerce_xy(point1, param_name="point1")
    x2, y2 = _coerce_xy(point2, param_name="point2")
    return Vector2D(dx=x2 - x1, dy=y2 - y1)


def vector_length(vector: VectorLike2D) -> float:
    """Computes the magnitude (length) of a 2D vector.

    Args:
        vector: A `Vector2D` or `(dx, dy)` tuple.

    Returns:
        float: The non-negative magnitude of the vector.

    Raises:
        TypeError: If `vector` is not a valid vector-like input.

    Complexity:
        O(1) time, O(1) space.
    """
    dx, dy = _coerce_xy(vector, param_name="vector")
    return math.hypot(dx, dy)


def normalize_vector(vector: VectorLike2D) -> Vector2D:
    """Normalizes a 2D vector to unit length.

    WHY it exists: direction-only comparisons (e.g., "is the hand
    moving roughly rightward?") should be invariant to speed. Producing
    a stable unit vector for a near-zero-length input would amplify
    noise, so this raises instead of silently returning a garbage
    direction.

    Args:
        vector: A `Vector2D` or `(dx, dy)` tuple.

    Returns:
        Vector2D: A unit-length vector pointing in the same direction
        as `vector`.

    Raises:
        TypeError: If `vector` is not a valid vector-like input.
        ValueError: If `vector` has near-zero length (below `EPSILON`),
            since its direction is undefined.

    Complexity:
        O(1) time, O(1) space.
    """
    dx, dy = _coerce_xy(vector, param_name="vector")
    length = math.hypot(dx, dy)
    if length < EPSILON:
        raise ValueError(
            "normalize_vector received a near-zero-length vector; "
            "direction is undefined."
        )
    return Vector2D(dx=dx / length, dy=dy / length)


def dot_product(vector1: VectorLike2D, vector2: VectorLike2D) -> float:
    """Computes the dot product of two 2D vectors.

    Args:
        vector1: First vector, as `Vector2D` or `(dx, dy)`.
        vector2: Second vector, as `Vector2D` or `(dx, dy)`.

    Returns:
        float: The scalar dot product.

    Raises:
        TypeError: If either vector is not a valid vector-like input.

    Complexity:
        O(1) time, O(1) space.
    """
    dx1, dy1 = _coerce_xy(vector1, param_name="vector1")
    dx2, dy2 = _coerce_xy(vector2, param_name="vector2")
    return dx1 * dx2 + dy1 * dy2


def cross_product(vector1: VectorLike2D, vector2: VectorLike2D) -> float:
    """Computes the scalar (z-component) cross product of two 2D vectors.

    WHY a scalar rather than a 3D vector: for 2D vectors, only the
    z-component of the full 3D cross product is non-zero. Its sign
    indicates rotational direction (clockwise vs. counter-clockwise),
    which is the property most gesture/orientation heuristics need
    (e.g., detecting a swipe's turning direction).

    Args:
        vector1: First vector, as `Vector2D` or `(dx, dy)`.
        vector2: Second vector, as `Vector2D` or `(dx, dy)`.

    Returns:
        float: The scalar cross product; positive indicates `vector2`
        is counter-clockwise from `vector1`, negative indicates
        clockwise, and zero indicates the vectors are collinear.

    Raises:
        TypeError: If either vector is not a valid vector-like input.

    Complexity:
        O(1) time, O(1) space.
    """
    dx1, dy1 = _coerce_xy(vector1, param_name="vector1")
    dx2, dy2 = _coerce_xy(vector2, param_name="vector2")
    return dx1 * dy2 - dy1 * dx2


def midpoint(point1: PointLike2D, point2: PointLike2D) -> Point2D:
    """Computes the midpoint between two 2D points.

    Args:
        point1: First point, as `Point2D` or `(x, y)`.
        point2: Second point, as `Point2D` or `(x, y)`.

    Returns:
        Point2D: The midpoint of `point1` and `point2`.

    Raises:
        TypeError: If either point is not a valid point-like input.

    Complexity:
        O(1) time, O(1) space.
    """
    x1, y1 = _coerce_xy(point1, param_name="point1")
    x2, y2 = _coerce_xy(point2, param_name="point2")
    return Point2D(x=(x1 + x2) / 2.0, y=(y1 + y2) / 2.0)


def calculate_direction(point1: PointLike2D, point2: PointLike2D) -> float:
    """Computes the direction from `point1` to `point2`, in degrees.

    WHY degrees rather than radians as the public unit: this function's
    typical consumers are UI/gesture-dispatch code and logging, where
    degrees are far more human-readable and easier to reason about in
    debug overlays than radians.

    Args:
        point1: Origin point, as `Point2D` or `(x, y)`.
        point2: Destination point, as `Point2D` or `(x, y)`.

    Returns:
        float: The direction in degrees, in the range
        ``(-180.0, 180.0]``, measured counter-clockwise from the
        positive x-axis assuming a standard mathematical (not
        image-flipped) y-axis. Callers working in image space with a
        top-left origin should negate `dy` before interpreting the
        result as "up" vs. "down".

    Raises:
        TypeError: If either point is not a valid point-like input.
        ValueError: If `point1` and `point2` coincide, making the
            direction undefined.

    Complexity:
        O(1) time, O(1) space.
    """
    x1, y1 = _coerce_xy(point1, param_name="point1")
    x2, y2 = _coerce_xy(point2, param_name="point2")
    dx = x2 - x1
    dy = y2 - y1
    if math.hypot(dx, dy) < EPSILON:
        raise ValueError(
            "calculate_direction received coincident points; direction "
            "is undefined."
        )
    return math.degrees(math.atan2(dy, dx))


# =========================================================================== #
# Aggregate geometry
# =========================================================================== #
def calculate_centroid(points: Sequence[PointLike2D]) -> Point2D:
    """Computes the centroid (arithmetic mean) of a set of 2D points.

    Args:
        points: A non-empty sequence of `Point2D` or `(x, y)` values.

    Returns:
        Point2D: The centroid of `points`.

    Raises:
        ValueError: If `points` is empty.
        TypeError: If any element of `points` is not a valid point-like
            input.

    Complexity:
        O(n) time, O(1) additional space, where n is `len(points)`.
    """
    _require_non_empty(points, param_name="points")
    coords = [_coerce_xy(point, param_name="points[i]") for point in points]
    sum_x = sum(x for x, _ in coords)
    sum_y = sum(y for _, y in coords)
    count = len(coords)
    return Point2D(x=sum_x / count, y=sum_y / count)


def calculate_hand_center(points: Sequence[PointLike2D]) -> Point2D:
    """Computes the centroid of a set of hand landmark points.

    WHY a thin, separately named wrapper around `calculate_centroid`:
    the mathematical operation is identical, but naming it explicitly
    for hands gives call sites self-documenting intent ("the hand's
    center") without forcing every caller to know that a hand center is
    "just" a centroid. Future divergence (e.g., weighting the palm
    landmarks more heavily than fingertips) can be introduced here
    without touching the generic `calculate_centroid` used elsewhere.

    Args:
        points: A non-empty sequence of hand landmark points, as
            `Point2D` or `(x, y)` values.

    Returns:
        Point2D: The centroid of the provided hand landmarks.

    Raises:
        ValueError: If `points` is empty.
        TypeError: If any element of `points` is not a valid point-like
            input.

    Complexity:
        O(n) time, O(1) additional space, where n is `len(points)`.
    """
    return calculate_centroid(points)


def calculate_bounding_box(points: Sequence[PointLike2D]) -> BoundingBox:
    """Computes the tightest axis-aligned bounding box around a set of
    2D points.

    Args:
        points: A non-empty sequence of `Point2D` or `(x, y)` values.

    Returns:
        BoundingBox: The tightest axis-aligned box enclosing all
        `points`.

    Raises:
        ValueError: If `points` is empty.
        TypeError: If any element of `points` is not a valid point-like
            input.

    Complexity:
        O(n) time, O(1) additional space, where n is `len(points)`.
    """
    _require_non_empty(points, param_name="points")
    coords = [_coerce_xy(point, param_name="points[i]") for point in points]
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return BoundingBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))


def is_point_inside_rectangle(point: PointLike2D, rectangle: BoundingBox) -> bool:
    """Determines whether a point lies within (or on the edge of) a
    bounding box.

    Args:
        point: The point to test, as `Point2D` or `(x, y)`.
        rectangle: The `BoundingBox` to test against.

    Returns:
        bool: True if `point` lies within `rectangle`'s bounds
        (inclusive of edges), False otherwise.

    Raises:
        TypeError: If `point` is not a valid point-like input, or if
            `rectangle` is not a `BoundingBox`.

    Complexity:
        O(1) time, O(1) space.
    """
    if not isinstance(rectangle, BoundingBox):
        raise TypeError(
            f"rectangle must be a BoundingBox, got {type(rectangle).__name__}."
        )
    x, y = _coerce_xy(point, param_name="point")
    return (
        rectangle.x_min <= x <= rectangle.x_max
        and rectangle.y_min <= y <= rectangle.y_max
    )


def is_point_inside_circle(
    point: PointLike2D, center: PointLike2D, radius: float
) -> bool:
    """Determines whether a point lies within (or on the edge of) a
    circle.

    Args:
        point: The point to test, as `Point2D` or `(x, y)`.
        center: The circle's center, as `Point2D` or `(x, y)`.
        radius: The circle's radius. Must be non-negative.

    Returns:
        bool: True if `point` lies within `radius` of `center`
        (inclusive), False otherwise.

    Raises:
        TypeError: If `point` or `center` is not a valid point-like
            input.
        ValueError: If `radius` is negative.

    Complexity:
        O(1) time, O(1) space.
    """
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}.")
    return calculate_distance(point, center) <= radius


# =========================================================================== #
# Kinematics: velocity, acceleration, direction
# =========================================================================== #
def calculate_velocity(
    previous_point: PointLike2D, current_point: PointLike2D, delta_time: float
) -> Velocity:
    """Computes instantaneous velocity between two sampled points.

    Args:
        previous_point: The point's position at the previous sample, as
            `Point2D` or `(x, y)`.
        current_point: The point's position at the current sample, as
            `Point2D` or `(x, y)`.
        delta_time: Elapsed time, in seconds, between the two samples.
            Must be strictly positive.

    Returns:
        Velocity: The estimated velocity, in units per second (same
        spatial unit as the input points, e.g., normalized coordinates
        per second).

    Raises:
        TypeError: If either point is not a valid point-like input.
        ValueError: If `delta_time` is not strictly positive.

    Complexity:
        O(1) time, O(1) space.

    Notes:
        Real-time frame timings are rarely perfectly uniform; callers
        should measure `delta_time` from actual frame timestamps (see
        `calculate_frame_delta`) rather than assuming a fixed FPS, or
        velocity estimates will drift under frame-rate jitter.
    """
    if delta_time <= 0.0:
        raise ValueError(f"delta_time must be > 0, got {delta_time}.")
    x1, y1 = _coerce_xy(previous_point, param_name="previous_point")
    x2, y2 = _coerce_xy(current_point, param_name="current_point")
    return Velocity(vx=(x2 - x1) / delta_time, vy=(y2 - y1) / delta_time)


def calculate_acceleration(
    previous_velocity: Velocity, current_velocity: Velocity, delta_time: float
) -> Acceleration:
    """Computes instantaneous acceleration between two sampled velocities.

    Args:
        previous_velocity: Velocity at the previous sample.
        current_velocity: Velocity at the current sample.
        delta_time: Elapsed time, in seconds, between the two velocity
            samples. Must be strictly positive.

    Returns:
        Acceleration: The estimated acceleration, in units per second
        squared.

    Raises:
        TypeError: If either velocity is not a `Velocity` instance.
        ValueError: If `delta_time` is not strictly positive.

    Complexity:
        O(1) time, O(1) space.
    """
    if not isinstance(previous_velocity, Velocity):
        raise TypeError(
            "previous_velocity must be a Velocity, got "
            f"{type(previous_velocity).__name__}."
        )
    if not isinstance(current_velocity, Velocity):
        raise TypeError(
            "current_velocity must be a Velocity, got "
            f"{type(current_velocity).__name__}."
        )
    if delta_time <= 0.0:
        raise ValueError(f"delta_time must be > 0, got {delta_time}.")
    return Acceleration(
        ax=(current_velocity.vx - previous_velocity.vx) / delta_time,
        ay=(current_velocity.vy - previous_velocity.vy) / delta_time,
    )


def calculate_frame_delta(previous_time: float, current_time: float) -> float:
    """Computes the elapsed time between two frame timestamps.

    Args:
        previous_time: Timestamp of the previous frame, in seconds
            (e.g., from `time.perf_counter()`).
        current_time: Timestamp of the current frame, in seconds.

    Returns:
        float: Elapsed time in seconds. Guaranteed non-negative; clock
        sources that can jump backward (rare, but possible with some
        system clocks) are clamped to zero rather than returning a
        negative delta that would corrupt velocity/acceleration math.

    Raises:
        TypeError: If either timestamp is not numeric.

    Complexity:
        O(1) time, O(1) space.
    """
    if not isinstance(previous_time, (int, float)) or not isinstance(
        current_time, (int, float)
    ):
        raise TypeError("previous_time and current_time must be numeric.")
    return max(0.0, float(current_time) - float(previous_time))


def fps_from_delta(delta_time: float) -> float:
    """Converts a frame delta time into an instantaneous frames-per-second
    value.

    Args:
        delta_time: Elapsed time for the last frame, in seconds. Must
            be non-negative.

    Returns:
        float: Instantaneous FPS. Returns `MAX_FLOAT` if `delta_time`
        is closer to zero than `EPSILON`, representing an effectively
        unbounded frame rate rather than raising or returning
        `math.inf` (which downstream numeric code may not expect).

    Raises:
        ValueError: If `delta_time` is negative.

    Complexity:
        O(1) time, O(1) space.
    """
    if delta_time < 0.0:
        raise ValueError(f"delta_time must be >= 0, got {delta_time}.")
    if delta_time < EPSILON:
        return MAX_FLOAT
    return 1.0 / delta_time


# =========================================================================== #
# Normalization, clamping, and interpolation
# =========================================================================== #
def normalize(
    value: float, old_min: float, old_max: float, new_min: float, new_max: float
) -> float:
    """Linearly remaps `value` from one range to another.

    Args:
        value: The value to remap.
        old_min: Lower bound of the source range.
        old_max: Upper bound of the source range.
        new_min: Lower bound of the target range.
        new_max: Upper bound of the target range.

    Returns:
        float: `value` remapped into `[new_min, new_max]`. Note the
        result is not clamped; values outside `[old_min, old_max]`
        extrapolate linearly. Use `clamp` on the result if a strict
        bound is required.

    Raises:
        ValueError: If `old_max` equals `old_min` (zero-width source
            range makes the mapping undefined).

    Complexity:
        O(1) time, O(1) space.
    """
    old_range = old_max - old_min
    if abs(old_range) < EPSILON:
        raise ValueError(
            "normalize received a zero-width source range "
            f"(old_min == old_max == {old_min}); the mapping is undefined."
        )
    ratio = (value - old_min) / old_range
    return new_min + ratio * (new_max - new_min)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Restricts `value` to the closed interval `[minimum, maximum]`.

    Args:
        value: The value to clamp.
        minimum: Lower bound, inclusive.
        maximum: Upper bound, inclusive.

    Returns:
        float: `value` if within bounds, otherwise the nearest bound.

    Raises:
        ValueError: If `minimum` is greater than `maximum`.

    Complexity:
        O(1) time, O(1) space.
    """
    if minimum > maximum:
        raise ValueError(
            f"minimum ({minimum}) must be <= maximum ({maximum})."
        )
    return max(minimum, min(maximum, value))


def lerp(start: float, end: float, t: float) -> float:
    """Linearly interpolates between `start` and `end`.

    Args:
        start: Value at `t == 0.0`.
        end: Value at `t == 1.0`.
        t: Interpolation factor. Not required to lie in `[0, 1]`;
            values outside that range extrapolate linearly, which is
            intentionally permitted for overshoot/spring-style effects.

    Returns:
        float: The interpolated value `start + t * (end - start)`.

    Complexity:
        O(1) time, O(1) space.
    """
    return start + t * (end - start)


def inverse_lerp(start: float, end: float, value: float) -> float:
    """Computes the interpolation factor `t` such that
    ``lerp(start, end, t) == value``.

    Args:
        start: Value corresponding to `t == 0.0`.
        end: Value corresponding to `t == 1.0`.
        value: The value to solve for.

    Returns:
        float: The interpolation factor `t`. Not clamped to `[0, 1]`;
        `value` outside `[start, end]` yields `t` outside `[0, 1]`.

    Raises:
        ValueError: If `start` equals `end` (undefined inverse).

    Complexity:
        O(1) time, O(1) space.
    """
    span = end - start
    if abs(span) < EPSILON:
        raise ValueError(
            f"inverse_lerp received start == end == {start}; the "
            "inverse mapping is undefined."
        )
    return (value - start) / span


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Computes the classic Hermite smoothstep interpolation.

    WHY it exists: linear fades (`lerp`) have a visually abrupt
    velocity discontinuity at their endpoints, which reads as
    mechanical in UI/AR animation. Smoothstep produces zero first-
    derivative at both endpoints, giving a natural ease-in/ease-out
    feel with a single cheap polynomial evaluation.

    Args:
        edge0: Lower edge of the transition.
        edge1: Upper edge of the transition.
        x: Input value to evaluate.

    Returns:
        float: 0.0 for `x <= edge0`, 1.0 for `x >= edge1`, and a smooth
        cubic Hermite interpolation in between.

    Raises:
        ValueError: If `edge0` equals `edge1` (zero-width transition).

    Complexity:
        O(1) time, O(1) space.
    """
    if abs(edge1 - edge0) < EPSILON:
        raise ValueError(
            f"smoothstep received edge0 == edge1 == {edge0}; the "
            "transition width is undefined."
        )
    t = clamp((x - edge0) / (edge1 - edge0), minimum=0.0, maximum=1.0)
    return t * t * (3.0 - 2.0 * t)


def update_alpha(
    current_alpha: float, target_alpha: float, fade_speed: float, delta_time: float
) -> InterpolationResult:
    """Advances `current_alpha` toward `target_alpha` at a fixed,
    frame-rate-independent speed.

    WHY frame-rate independence matters: naively doing
    `alpha += fade_speed` every frame produces a fade duration that
    depends on FPS, so the same UI element fades at a visibly different
    speed on a 30 FPS phone versus a 60 FPS phone. Scaling the step by
    `delta_time` keeps wall-clock fade duration constant regardless of
    frame rate.

    Args:
        current_alpha: The current alpha/opacity value.
        target_alpha: The alpha/opacity value to move toward.
        fade_speed: Maximum rate of change, in alpha units per second.
            Must be non-negative.
        delta_time: Elapsed time since the last update, in seconds.
            Must be non-negative.

    Returns:
        InterpolationResult: The new alpha value (clamped to
        `[DEFAULT_ALPHA_MIN, DEFAULT_ALPHA_MAX]`) and whether it has
        converged to `target_alpha`.

    Raises:
        ValueError: If `fade_speed` or `delta_time` is negative.

    Complexity:
        O(1) time, O(1) space.
    """
    if fade_speed < 0.0:
        raise ValueError(f"fade_speed must be >= 0, got {fade_speed}.")
    if delta_time < 0.0:
        raise ValueError(f"delta_time must be >= 0, got {delta_time}.")

    max_step = fade_speed * delta_time
    difference = target_alpha - current_alpha

    if abs(difference) <= max_step:
        new_alpha = target_alpha
    else:
        new_alpha = current_alpha + math.copysign(max_step, difference)

    new_alpha = clamp(new_alpha, minimum=DEFAULT_ALPHA_MIN, maximum=DEFAULT_ALPHA_MAX)
    has_converged = abs(new_alpha - target_alpha) < EPSILON
    return InterpolationResult(value=new_alpha, has_converged=has_converged)


def fade_in(
    alpha: float, speed: float = DEFAULT_FADE_SPEED, delta_time: float = 0.0
) -> float:
    """Advances `alpha` toward full opacity at a fixed rate.

    Args:
        alpha: The current alpha/opacity value.
        speed: Rate of change, in alpha units per second. Defaults to
            `DEFAULT_FADE_SPEED`.
        delta_time: Elapsed time since the last update, in seconds.

    Returns:
        float: The new alpha value, clamped to
        `[DEFAULT_ALPHA_MIN, DEFAULT_ALPHA_MAX]`.

    Raises:
        ValueError: If `speed` or `delta_time` is negative.

    Complexity:
        O(1) time, O(1) space.
    """
    return update_alpha(
        current_alpha=alpha,
        target_alpha=DEFAULT_ALPHA_MAX,
        fade_speed=speed,
        delta_time=delta_time,
    ).value


def fade_out(
    alpha: float, speed: float = DEFAULT_FADE_SPEED, delta_time: float = 0.0
) -> float:
    """Advances `alpha` toward full transparency at a fixed rate.

    Args:
        alpha: The current alpha/opacity value.
        speed: Rate of change, in alpha units per second. Defaults to
            `DEFAULT_FADE_SPEED`.
        delta_time: Elapsed time since the last update, in seconds.

    Returns:
        float: The new alpha value, clamped to
        `[DEFAULT_ALPHA_MIN, DEFAULT_ALPHA_MAX]`.

    Raises:
        ValueError: If `speed` or `delta_time` is negative.

    Complexity:
        O(1) time, O(1) space.
    """
    return update_alpha(
        current_alpha=alpha,
        target_alpha=DEFAULT_ALPHA_MIN,
        fade_speed=speed,
        delta_time=delta_time,
    ).value


def ease_in(t: float) -> float:
    """Applies a quadratic ease-in curve to a normalized time value.

    WHY quadratic rather than a heavier curve family: a `t**2` curve is
    the cheapest ease-in that still reads as "starting slow" to a
    human eye, which matters when this may be evaluated per-frame for
    many simultaneously animating overlay elements.

    Args:
        t: Normalized time, expected in `[0.0, 1.0]` but not enforced,
            to allow deliberate overshoot animation.

    Returns:
        float: The eased value, `t * t`.

    Complexity:
        O(1) time, O(1) space.
    """
    return t * t


def ease_out(t: float) -> float:
    """Applies a quadratic ease-out curve to a normalized time value.

    Args:
        t: Normalized time, expected in `[0.0, 1.0]` but not enforced.

    Returns:
        float: The eased value, `1 - (1 - t) ** 2`.

    Complexity:
        O(1) time, O(1) space.
    """
    inverse = 1.0 - t
    return 1.0 - inverse * inverse


def ease_in_out(t: float) -> float:
    """Applies a quadratic ease-in-out curve to a normalized time value.

    Args:
        t: Normalized time, expected in `[0.0, 1.0]` but not enforced.

    Returns:
        float: The eased value; symmetric quadratic acceleration in the
        first half and deceleration in the second half.

    Complexity:
        O(1) time, O(1) space.
    """
    if t < 0.5:
        return 2.0 * t * t
    inverse = -2.0 * t + 2.0
    return 1.0 - (inverse * inverse) / 2.0


# =========================================================================== #
# Filtering & smoothing
# =========================================================================== #
def average(values: Sequence[float]) -> float:
    """Computes the arithmetic mean of a sequence of numbers.

    Args:
        values: A non-empty sequence of numbers.

    Returns:
        float: The arithmetic mean of `values`.

    Raises:
        ValueError: If `values` is empty.

    Complexity:
        O(n) time, O(1) additional space.
    """
    _require_non_empty(values, param_name="values")
    return sum(values) / len(values)


def moving_average(
    values: Sequence[float], window_size: int = DEFAULT_MOVING_AVERAGE_WINDOW
) -> float:
    """Computes the average of the most recent `window_size` samples.

    WHY it exists: a simple moving average is the cheapest possible
    jitter suppressor for noisy landmark streams and is often layered
    beneath a heavier filter (e.g., a one-euro filter) rather than
    replacing it, since it introduces predictable, bounded lag.

    Args:
        values: A non-empty sequence of numbers, ordered oldest-first.
        window_size: Number of most-recent samples to average. Must be
            a positive integer. If `window_size` exceeds `len(values)`,
            all available samples are used.

    Returns:
        float: The average of the last `min(window_size, len(values))`
        elements of `values`.

    Raises:
        ValueError: If `values` is empty or `window_size` is not a
            positive integer.

    Complexity:
        O(w) time, O(1) additional space, where w is the effective
        window size.
    """
    _require_non_empty(values, param_name="values")
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}.")
    effective_window = values[-window_size:]
    return sum(effective_window) / len(effective_window)


def low_pass_filter(
    previous: float, current: float, alpha: float = DEFAULT_SMOOTHING_FACTOR
) -> float:
    """Applies a first-order low-pass filter to a noisy signal.

    Args:
        previous: The previous (already filtered) value.
        current: The new raw sample.
        alpha: Weight applied to `current`, in `[0.0, 1.0]`. Higher
            values track `current` more closely (less smoothing);
            lower values favor `previous` (more smoothing). Defaults to
            `DEFAULT_SMOOTHING_FACTOR`.

    Returns:
        float: The filtered value, `alpha * current + (1 - alpha) *
        previous`.

    Raises:
        ValueError: If `alpha` is not in `[0.0, 1.0]`.

    Complexity:
        O(1) time, O(1) space.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}.")
    return alpha * current + (1.0 - alpha) * previous


def exponential_smoothing(
    previous: float, current: float, factor: float = DEFAULT_SMOOTHING_FACTOR
) -> float:
    """Applies exponential smoothing to a noisy signal.

    WHY a separate function from `low_pass_filter` despite an identical
    formula: the two are mathematically equivalent single-pole IIR
    filters, but they are reached for in different contexts (signal
    processing vs. time-series smoothing) by different engineers.
    Providing both names avoids forcing a "which one do I mean"
    lookup and matches the vocabulary used in the CV/tracking domain
    versus the analytics domain respectively.

    Args:
        previous: The previous (already smoothed) value.
        current: The new raw sample.
        factor: Smoothing factor in `[0.0, 1.0]`; higher values track
            `current` more closely. Defaults to
            `DEFAULT_SMOOTHING_FACTOR`.

    Returns:
        float: The smoothed value.

    Raises:
        ValueError: If `factor` is not in `[0.0, 1.0]`.

    Complexity:
        O(1) time, O(1) space.
    """
    return low_pass_filter(previous=previous, current=current, alpha=factor)


def is_hand_stable(
    previous_landmarks: Sequence[PointLike2D],
    current_landmarks: Sequence[PointLike2D],
    threshold: float = DEFAULT_STABILITY_THRESHOLD,
) -> bool:
    """Determines whether a hand's landmarks are effectively stationary
    between two frames.

    WHY it exists: many downstream features (e.g., "confirm" gestures
    that require the hand to hold still, or throttling expensive
    re-computation while a hand is not moving) need a single boolean
    stability signal derived from noisy per-landmark motion, without
    every consumer re-deriving its own averaging/thresholding logic.

    Args:
        previous_landmarks: Landmark positions from the previous frame,
            as `Point2D` or `(x, y)` values, ordered to align
            index-for-index with `current_landmarks`.
        current_landmarks: Landmark positions from the current frame,
            same length and ordering as `previous_landmarks`.
        threshold: Maximum mean per-landmark displacement, in the same
            unit as the input coordinates, below which the hand is
            considered stable. Defaults to
            `DEFAULT_STABILITY_THRESHOLD`.

    Returns:
        bool: True if the mean landmark displacement between the two
        frames is at or below `threshold`, False otherwise.

    Raises:
        ValueError: If either landmark sequence is empty, if their
            lengths differ, or if `threshold` is negative.
        TypeError: If any landmark is not a valid point-like input.

    Complexity:
        O(n) time, O(1) additional space, where n is the number of
        landmarks.
    """
    _require_non_empty(previous_landmarks, param_name="previous_landmarks")
    _require_non_empty(current_landmarks, param_name="current_landmarks")
    if len(previous_landmarks) != len(current_landmarks):
        raise ValueError(
            "previous_landmarks and current_landmarks must have the same "
            f"length (got {len(previous_landmarks)} and "
            f"{len(current_landmarks)})."
        )
    if threshold < 0.0:
        raise ValueError(f"threshold must be >= 0, got {threshold}.")

    displacements = [
        calculate_distance(previous, current)
        for previous, current in zip(previous_landmarks, current_landmarks)
    ]
    mean_displacement = sum(displacements) / len(displacements)
    return mean_displacement <= threshold


# =========================================================================== #
# Gesture heuristics
#
# WHY these live here rather than in a detection module: classifying a
# gesture from *already-extracted* landmark geometry is pure math with
# no dependency on which model produced the landmarks. Keeping the
# classification here means a future face-tracking or full-body-pose
# module can reuse the same finger-state/angle primitives without
# depending on a hand-specific detection module.
# =========================================================================== #
def detect_pinch(
    thumb_tip: PointLike2D,
    index_tip: PointLike2D,
    threshold: float = DEFAULT_PINCH_THRESHOLD,
) -> GestureResult:
    """Detects a pinch gesture from thumb-tip and index-tip proximity.

    Args:
        thumb_tip: Thumb fingertip position, as `Point2D` or `(x, y)`,
            in normalized coordinate space.
        index_tip: Index fingertip position, as `Point2D` or `(x, y)`,
            in normalized coordinate space.
        threshold: Normalized distance below which the fingers are
            considered pinched. Defaults to `DEFAULT_PINCH_THRESHOLD`.

    Returns:
        GestureResult: `GestureType.PINCH` with confidence scaling from
        0.0 (at `threshold`) to 1.0 (fingers touching) if pinched;
        otherwise `GestureType.UNKNOWN` with confidence 0.0. The raw
        distance is included in `metadata["distance"]` for callers that
        need it for debugging or fine-grained thresholds of their own.

    Raises:
        TypeError: If either point is not a valid point-like input.
        ValueError: If `threshold` is negative.

    Complexity:
        O(1) time, O(1) space.
    """
    if threshold < 0.0:
        raise ValueError(f"threshold must be >= 0, got {threshold}.")
    distance = calculate_distance(thumb_tip, index_tip)
    if distance < threshold:
        confidence = 1.0 - (distance / threshold if threshold > EPSILON else 0.0)
        return GestureResult(
            gesture=GestureType.PINCH,
            confidence=clamp(confidence, minimum=0.0, maximum=1.0),
            metadata={"distance": distance},
        )
    return GestureResult(
        gesture=GestureType.UNKNOWN,
        confidence=0.0,
        metadata={"distance": distance},
    )


def count_extended_fingers(finger_states: Sequence[bool]) -> int:
    """Counts how many fingers are extended.

    Args:
        finger_states: Per-finger extension flags, conventionally
            ordered (thumb, index, middle, ring, pinky), where True
            means extended.

    Returns:
        int: The number of True values in `finger_states`.

    Raises:
        ValueError: If `finger_states` is empty.

    Complexity:
        O(n) time, O(1) additional space.
    """
    _require_non_empty(finger_states, param_name="finger_states")
    return sum(1 for is_extended in finger_states if is_extended)


def detect_open_palm(finger_states: Sequence[bool]) -> GestureResult:
    """Detects an open-palm gesture (all fingers extended).

    Args:
        finger_states: Per-finger extension flags, ordered
            (thumb, index, middle, ring, pinky).

    Returns:
        GestureResult: `GestureType.OPEN_PALM` with confidence 0.9 if
        every finger is extended, otherwise `GestureType.UNKNOWN` with
        confidence 0.0.

        WHY 0.9 rather than 1.0: this is a discrete geometric heuristic,
        not a learned probability; reserving 1.0 avoids implying a
        stronger statistical guarantee than a boolean rule can provide,
        and leaves headroom for a future learned classifier to report
        genuinely higher confidence than the heuristic baseline.

    Raises:
        ValueError: If `finger_states` is empty.

    Complexity:
        O(n) time, O(1) additional space.
    """
    _require_non_empty(finger_states, param_name="finger_states")
    if all(finger_states):
        return GestureResult(gesture=GestureType.OPEN_PALM, confidence=0.9)
    return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)


def detect_closed_fist(finger_states: Sequence[bool]) -> GestureResult:
    """Detects a closed-fist gesture (no fingers extended).

    Args:
        finger_states: Per-finger extension flags, ordered
            (thumb, index, middle, ring, pinky).

    Returns:
        GestureResult: `GestureType.CLOSED_FIST` with confidence 0.9 if
        no finger is extended, otherwise `GestureType.UNKNOWN` with
        confidence 0.0.

    Raises:
        ValueError: If `finger_states` is empty.

    Complexity:
        O(n) time, O(1) additional space.
    """
    _require_non_empty(finger_states, param_name="finger_states")
    if not any(finger_states):
        return GestureResult(gesture=GestureType.CLOSED_FIST, confidence=0.9)
    return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)


def detect_peace_sign(finger_states: Sequence[bool]) -> GestureResult:
    """Detects a peace/victory sign (index and middle extended only).

    Args:
        finger_states: Per-finger extension flags, ordered exactly
            (thumb, index, middle, ring, pinky); length must be 5.

    Returns:
        GestureResult: `GestureType.PEACE` with confidence 0.85 if
        index and middle are extended and thumb, ring, and pinky are
        not, otherwise `GestureType.UNKNOWN` with confidence 0.0.

    Raises:
        ValueError: If `finger_states` does not have exactly 5 elements.

    Complexity:
        O(1) time, O(1) space.
    """
    if len(finger_states) != 5:
        raise ValueError(
            "finger_states must have exactly 5 elements ordered "
            f"(thumb, index, middle, ring, pinky), got {len(finger_states)}."
        )
    thumb, index, middle, ring, pinky = finger_states
    if index and middle and not thumb and not ring and not pinky:
        return GestureResult(gesture=GestureType.PEACE, confidence=0.85)
    return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)


def detect_thumbs_up(finger_states: Sequence[bool]) -> GestureResult:
    """Detects a thumbs-up gesture (only the thumb extended).

    Args:
        finger_states: Per-finger extension flags, ordered exactly
            (thumb, index, middle, ring, pinky); length must be 5.

    Returns:
        GestureResult: `GestureType.THUMBS_UP` with confidence 0.8 if
        only the thumb is extended, otherwise `GestureType.UNKNOWN`
        with confidence 0.0.

        Notes:
            This heuristic evaluates finger extension only; it does
            not verify wrist/hand orientation, so a sideways or
            downward-pointing thumb with all other fingers curled will
            still classify as `THUMBS_UP`. Callers needing directional
            confirmation should additionally check hand rotation (see
            `calculate_direction` / `calculate_angle`).

    Raises:
        ValueError: If `finger_states` does not have exactly 5 elements.

    Complexity:
        O(1) time, O(1) space.
    """
    if len(finger_states) != 5:
        raise ValueError(
            "finger_states must have exactly 5 elements ordered "
            f"(thumb, index, middle, ring, pinky), got {len(finger_states)}."
        )
    thumb, index, middle, ring, pinky = finger_states
    if thumb and not index and not middle and not ring and not pinky:
        return GestureResult(gesture=GestureType.THUMBS_UP, confidence=0.8)
    return GestureResult(gesture=GestureType.UNKNOWN, confidence=0.0)