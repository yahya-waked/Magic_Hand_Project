"""Aether Hands -- a standalone desktop GUI around the CV pipeline.

This is the "outside Visual Studio" professional shell around the three
CLI demos (`ghost_mode.py`, `hand_fx.py`, `combo_mode.py`): a real
embedded video window with a mode picker and buttons, not a terminal
command you have to remember flags for.

Nothing about the underlying vision pipeline is reimplemented here.
This file imports the same public classes (`HandTracker`,
`BackgroundRemover`) and the same underscore-prefixed helper functions
`combo_mode.py` already reuses from `hand_fx.py` and `ghost_mode.py`,
and drives them from a QThread instead of a blocking `while True` +
`cv2.imshow` loop -- so the camera never blocks the UI, and the mode can
be switched live while the camera is running.

Three modes, picked from a dropdown instead of CLI flags:
    Combo  -- ghost fade (pinch) + energy ball / lightning, all together
    Ghost  -- fade-into-background only
    Hand FX -- energy ball / lightning only

Setup:
    pip install PySide6 opencv-python numpy qrcode[pil]
    (qrcode[pil] is optional -- only needed for the "scan to download your
    clip" QR code after recording; everything else works without it)
    (run from the same folder as hand_tracker.py, background_remover.py,
     hand_fx.py, and ghost_mode.py -- this file imports from them)

    For recordings to actually PLAY when scanned on a phone, install
    ffmpeg system-wide and make sure it's on PATH (https://ffmpeg.org):
    recordings are captured with the 'mp4v' codec (the one that works
    out of the box with plain opencv-python), which desktop players open
    fine but most mobile browsers refuse to play. If ffmpeg is found,
    each clip is auto re-encoded to H.264 right after recording stops,
    right before the QR code is shown. Without ffmpeg, the QR code still
    appears but the linked clip may not play in a phone's browser.

Run:
    python gui_app.py

Packaging as a standalone .exe/.app (no visible Python/terminal at all):
    pip install pyinstaller
    pyinstaller --onefile --windowed --name "HandFXStudio" gui_app.py
    (the built binary will be in dist/)
"""

from __future__ import annotations

import functools
import http.server
import logging
import math
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PySide6.QtCore import QPointF, QRectF, QTimer, QUrl, Qt, QThread, Signal, Slot
from PySide6.QtGui import (
    QBrush,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from background_remover import (
    BackgroundRemovalConfig,
    BackgroundRemover,
    BackgroundRemoverError,
)
from hand_tracker import Gesture, HandTracker, HandTrackerError, InvalidFrameError

from ghost_mode import (
    COUNTDOWN_SECONDS,
    DEFAULT_HOLD_SECONDS,
    DEFAULT_RELEASE_SECONDS,
    GHOST_ACTIVE_EPSILON,
    _draw_hand_skeleton,
    _draw_hud,
    _is_ghost_gesture,
)
from hand_fx import (
    _apply_glow,
    _draw_between_hands_orb,
    _draw_energy_ball,
    _draw_lightning,
    _smoothstep,
    _update_between_hands_charge,
    _update_hand_charges,
)

try:
    import qrcode
    _HAS_QRCODE = True
except ImportError:
    _HAS_QRCODE = False

logger = logging.getLogger("gui_app")

APP_TITLE = "Aether Hands"

# WHY a fixed target FPS instead of measuring actual throughput: the
# camera/model pipeline's real frame rate varies with machine load, but
# cv2.VideoWriter needs one fixed FPS baked into the file up front. 20 is
# a safe middle ground for a webcam + MediaPipe + segmentation pipeline on
# typical hardware -- if the real loop runs slower, the saved clip will
# just play a little fast rather than needing frame-duplication logic.
DEFAULT_RECORDING_FPS = 20.0
RECORDINGS_DIR_NAME = "recordings"

# WHY a fixed local port for serving recordings: the QR code has to point
# somewhere stable. This just needs to not collide with common local dev
# ports; if it's already taken (rare), the server simply fails to start
# and the app falls back to showing the file path instead of a QR code
# (see MainWindow._start_file_server).
LOCAL_SERVER_PORT = 8642


class Mode(Enum):
    COMBO = "combo"      # ghost fade + energy/lightning, together
    GHOST = "ghost"       # fade-into-background only
    HAND_FX = "hand_fx"   # energy ball / lightning only


# --------------------------------------------------------------------------- #
# "Scan to download your clip" support: a tiny local HTTP server that serves
# the recordings folder on the LAN, plus a QR code pointing at whichever
# file just finished recording. A phone on the same Wi-Fi can scan it and
# grab the clip immediately -- this is the "give the trade-show visitor
# their video on the spot" feature, not a cloud upload of any kind. Nothing
# leaves the local network.
# --------------------------------------------------------------------------- #
def _get_lan_ip() -> str:
    """Best-effort guess at this machine's LAN IP (not 127.0.0.1).

    WHY the UDP "connect": no packets are actually sent -- UDP connect()
    just asks the OS to pick which network interface *would* be used to
    reach that address, without sending anything. That's enough to get a
    real LAN-facing IP instead of localhost, which is what a phone on the
    same Wi-Fi actually needs to reach this machine. Falls back to
    loopback if there's no network at all (e.g. offline demo).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


class _QuietRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Same as SimpleHTTPRequestHandler, minus a log line per HTTP request.

    WHY overridden: the default handler prints every request to stderr,
    which would spam the console with noise every time a phone loads the
    video (browsers issue several range requests for one video file).
    """

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class RecordingsFileServer:
    """Serves the recordings folder over plain HTTP on the local network."""

    def __init__(self, directory: Path, port: int = LOCAL_SERVER_PORT) -> None:
        self._directory = directory
        self._port = port
        self._httpd: Optional[http.server.ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Returns True if the server came up, False if the port was busy."""
        if self._httpd is not None:
            return True
        handler = functools.partial(_QuietRequestHandler, directory=str(self._directory))
        try:
            self._httpd = http.server.ThreadingHTTPServer(("0.0.0.0", self._port), handler)
        except OSError:
            logger.warning("Could not bind local file server on port %s.", self._port)
            self._httpd = None
            return False
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


def _make_qr_pixmap(data: str, box_size: int = 6) -> Optional[QPixmap]:
    """Renders `data` as a QR code image. Returns None if qrcode isn't installed."""
    if not _HAS_QRCODE:
        return None
    qr = qrcode.QRCode(border=1, box_size=box_size)
    qr.add_data(data)
    qr.make(fit=True)
    pil_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    w, h = pil_image.size
    raw = pil_image.tobytes("raw", "RGB")
    # WHY .copy(): same reason as VideoWorker._emit_frame -- QImage doesn't
    # own `raw`'s memory by default, and it would otherwise be garbage
    # collected as soon as this function returns.
    qimage = QImage(raw, w, h, w * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


# --------------------------------------------------------------------------- #
# Idle-state artwork for the video panel, shown before the camera starts.
# WHY drawn in code instead of loading an image file: no external asset to
# ship, lose, or mismatch with the dark theme -- and it echoes the same
# glow-orb + lightning look the app itself produces, so the very first
# thing you see already hints at what "Aether Hands" does.
# --------------------------------------------------------------------------- #
def _build_placeholder_pixmap(width: int = 960, height: int = 540) -> QPixmap:
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#0a0b0d"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    background_gradient = QRadialGradient(width / 2, height / 2, max(width, height) * 0.75)
    background_gradient.setColorAt(0.0, QColor(26, 30, 38))
    background_gradient.setColorAt(1.0, QColor(9, 10, 13))
    painter.fillRect(0, 0, width, height, QBrush(background_gradient))

    orb_x, orb_y = width / 2, height / 2 - height * 0.06

    # Soft layered glow, cool cyan-blue "aether" tone -- same idea as the
    # energy ball glow in hand_fx.py, just static instead of pulsing.
    for radius, alpha in ((150, 35), (105, 60), (65, 110), (32, 190)):
        glow = QRadialGradient(orb_x, orb_y, radius)
        inner = QColor(110, 190, 255)
        inner.setAlpha(alpha)
        outer = QColor(110, 190, 255)
        outer.setAlpha(0)
        glow.setColorAt(0.0, inner)
        glow.setColorAt(1.0, outer)
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(orb_x, orb_y), radius, radius)

    # A handful of jagged lightning-style accents radiating outward.
    # WHY a fixed seed: this is drawn once at startup and reused for the
    # whole session -- a fixed seed keeps it looking intentional/designed
    # rather than jittering to a different random shape every launch.
    pen = QPen(QColor(190, 225, 255, 170))
    pen.setWidthF(2.2)
    painter.setPen(pen)
    rng = random.Random(1337)
    for angle_deg in range(0, 360, 40):
        angle = math.radians(angle_deg + rng.uniform(-12, 12))
        inner_r = 55
        outer_r = inner_r + rng.uniform(85, 150)
        x1 = orb_x + math.cos(angle) * inner_r
        y1 = orb_y + math.sin(angle) * inner_r
        x2 = orb_x + math.cos(angle) * outer_r
        y2 = orb_y + math.sin(angle) * outer_r
        mid_x = (x1 + x2) / 2 + rng.uniform(-18, 18)
        mid_y = (y1 + y2) / 2 + rng.uniform(-18, 18)
        path = QPainterPath()
        path.moveTo(x1, y1)
        path.lineTo(mid_x, mid_y)
        path.lineTo(x2, y2)
        painter.drawPath(path)

    text_top = orb_y + 145
    painter.setPen(QColor(240, 240, 240))
    title_font = QFont("Segoe UI", 26, QFont.DemiBold)
    painter.setFont(title_font)
    painter.drawText(QRectF(0, text_top, width, 48), Qt.AlignHCenter | Qt.AlignVCenter, APP_TITLE)

    painter.setPen(QColor(140, 146, 154))
    subtitle_font = QFont("Segoe UI", 12)
    painter.setFont(subtitle_font)
    painter.drawText(
        QRectF(0, text_top + 46, width, 30), Qt.AlignHCenter | Qt.AlignVCenter,
        "Press Start Camera to begin",
    )

    painter.end()
    return pixmap


class TranscodeThread(QThread):
    """Re-encodes a saved clip to H.264 so phone browsers can actually play it.

    WHY this exists: OpenCV's VideoWriter here uses the 'mp4v' fourcc
    (MPEG-4 Part 2) because it's the codec that ships reliably in the
    plain `opencv-python` wheel with no extra system libraries. Desktop
    players (VLC, etc.) open that fine, but mobile browsers -- which is
    exactly what opens the link behind the QR code -- only support H.264
    in an MP4 container. Without this step the QR code opens a page that
    just fails to play.

    Runs `ffmpeg` (if it's on PATH) in the background, writes to a temp
    file, then swaps it in under the *original* filename -- so the QR
    code (built from the filename before encoding finishes) still points
    at a valid link once this completes. If ffmpeg isn't installed, the
    original mp4v file is left as-is and `failed` is emitted so the UI
    can say so.
    Also corrects playback SPEED: the raw file's header claims a fixed
    nominal frame rate (DEFAULT_RECORDING_FPS) regardless of how fast the
    pipeline actually ran while capturing, so on typical hardware the
    real capture rate is slower than that nominal number -- which makes
    the saved clip visibly play faster than it was actually filmed.
    `real_fps` (measured by VideoWorker from actual frames-written /
    wall-clock time) is passed to ffmpeg as an INPUT framerate override
    (`-r` before `-i`), which re-times each frame to match how long the
    recording really took, without touching a single pixel.
    """

    finished_ok = Signal(str)
    failed = Signal(str, str)  # (message, original_source_path)

    def __init__(self, source_path: str, real_fps: float, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._source_path = source_path
        self._real_fps = real_fps

    def run(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.failed.emit("ffmpeg not found on PATH -- keeping original file.", self._source_path)
            return

        source = Path(self._source_path)
        temp_output = source.with_name(source.stem + "_h264_tmp.mp4")
        # No audio track in the source (recording only ever captured
        # video frames), so there's nothing to map for -c:a -- video-only
        # re-encode. `-r` BEFORE `-i` overrides the input's timing (speed
        # fix); the encode settings after `-i` handle format/compatibility.
        cmd = [
            "ffmpeg", "-y",
            "-r", f"{self._real_fps:.3f}",
            "-i", str(source),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(temp_output),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=180)
        except (OSError, subprocess.SubprocessError) as exc:
            self.failed.emit(f"ffmpeg failed to run: {exc}", self._source_path)
            return

        if result.returncode != 0 or not temp_output.exists():
            stderr_tail = result.stderr.decode("utf-8", "ignore")[-300:]
            self.failed.emit(f"ffmpeg encoding failed: {stderr_tail}", self._source_path)
            return

        try:
            source.unlink(missing_ok=True)
            temp_output.rename(source)
        except OSError as exc:
            self.failed.emit(f"Could not replace original file: {exc}", self._source_path)
            return

        self.finished_ok.emit(str(source))


# --------------------------------------------------------------------------- #
# Worker thread: owns the camera, the models, and the per-frame pipeline.
# Runs continuously once started; mode/skeleton/recapture are toggled live
# via plain attributes set from the GUI thread (Qt's queued cross-thread
# signal delivery makes each individual attribute write atomic enough for
# this -- worst case a single frame reads a half-updated flag, which just
# means that one frame renders in the old mode for a moment).
# --------------------------------------------------------------------------- #
class VideoWorker(QThread):
    frame_ready = Signal(QImage)
    status_message = Signal(str)
    background_captured = Signal()
    error_occurred = Signal(str)
    recording_started = Signal(str)
    recording_stopped = Signal(str, float)
    recording_failed = Signal(str)

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        hold_seconds: float = DEFAULT_HOLD_SECONDS,
        release_seconds: float = DEFAULT_RELEASE_SECONDS,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._hold_seconds = hold_seconds
        self._release_seconds = release_seconds

        self.mode: Mode = Mode.COMBO
        self.show_skeleton: bool = True
        self._running = False
        self._recapture_requested = False
        self._countdown_deadline: Optional[float] = None

        self._record_start_requested = False
        self._record_stop_requested = False
        self._recording = False
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._current_recording_path: Optional[str] = None
        self._record_frame_count = 0
        self._record_start_wall: Optional[float] = None

    def request_recapture(self) -> None:
        """Called from the GUI thread; the run loop picks this up next frame."""
        self._recapture_requested = True

    def start_recording(self) -> None:
        """Called from the GUI thread; the run loop picks this up next frame."""
        self._record_start_requested = True

    def stop_recording(self) -> None:
        """Called from the GUI thread; the run loop picks this up next frame."""
        self._record_stop_requested = True

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _make_recording_path() -> str:
        # Saved next to this script (not the current working directory),
        # so recordings land in the same place regardless of where the app
        # was launched from -- important once this is packaged as a
        # standalone .exe that could be double-clicked from anywhere.
        recordings_dir = Path(__file__).resolve().parent / RECORDINGS_DIR_NAME
        recordings_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(recordings_dir / f"aether_hands_{timestamp}.mp4")

    # -- internals ---------------------------------------------------- #

    def _emit_frame(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # WHY .copy(): QImage wraps the numpy buffer's memory by reference.
        # That buffer gets reused/overwritten by the next frame's cvtColor
        # call before the GUI thread has necessarily painted it, which
        # would show torn/garbled frames. Copying makes each QImage own
        # independent memory that survives the hop across threads.
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.frame_ready.emit(qimg)

    def _draw_countdown(self, frame: np.ndarray, remaining: float) -> None:
        message = f"Step out of frame... capturing in {remaining:.1f}s"
        cv2.putText(frame, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

    def run(self) -> None:  # noqa: C901 -- single linear pipeline, kept together on purpose
        self._running = True
        capture = cv2.VideoCapture(self._camera_index)
        if not capture.isOpened():
            self.error_occurred.emit(f"Could not open camera index {self._camera_index}.")
            self._running = False
            return
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        config = BackgroundRemovalConfig(
            gaussian_blur_kernel_size=11,
            edge_feather_kernel_size=15,
        )

        rng = random.Random()
        charge_state: dict = {}
        between_hands_state = {"charge": 0.0}
        ghost_progress = 0.0
        start_time = time.monotonic()
        previous_time = start_time

        try:
            with HandTracker(max_num_hands=2, draw_landmarks=False) as tracker, \
                 BackgroundRemover(config=config) as remover:

                remover.initialize()

                # Always grab an initial background plate at startup, same
                # as the CLI tools -- ghost/combo mode need one on hand even
                # if you start out in Hand FX only mode and switch later.
                self.status_message.emit("Capturing background plate -- step out of frame...")
                self._countdown_deadline = time.monotonic() + COUNTDOWN_SECONDS

                while self._running:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        continue
                    frame = cv2.flip(frame, 1)

                    if self._recapture_requested:
                        self._recapture_requested = False
                        self._countdown_deadline = time.monotonic() + COUNTDOWN_SECONDS
                        self.status_message.emit("Re-capturing background -- step out of frame...")

                    if self._countdown_deadline is not None:
                        remaining = self._countdown_deadline - time.monotonic()
                        if remaining > 0:
                            preview = frame.copy()
                            self._draw_countdown(preview, remaining)
                            self._emit_frame(preview)
                            continue
                        remover.save_background(frame)
                        self._countdown_deadline = None
                        self.status_message.emit("Background captured. Ready.")
                        self.background_captured.emit()
                        previous_time = time.monotonic()
                        continue

                    try:
                        hand_result = tracker.detect(frame)
                    except InvalidFrameError:
                        logger.exception("Bad frame; skipping.")
                        continue

                    now = time.monotonic()
                    dt = now - previous_time
                    previous_time = now
                    t = now - start_time
                    mode = self.mode

                    output_frame = frame.copy()

                    if mode in (Mode.GHOST, Mode.COMBO):
                        gesture_held = _is_ghost_gesture(hand_result.hands)
                        if gesture_held:
                            ghost_progress += dt / max(self._hold_seconds, 1e-6)
                        else:
                            ghost_progress -= dt / max(self._release_seconds, 1e-6)
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
                        # Not fading in this mode -- keep the bar at 0 so it
                        # doesn't show a stale percentage if the user
                        # switches back to Ghost/Combo later.
                        ghost_progress = 0.0

                    if mode in (Mode.HAND_FX, Mode.COMBO):
                        glow_layer = np.zeros_like(frame)

                        hand_charges = _update_hand_charges(charge_state, hand_result.hands, dt)
                        for hand, (effect_gesture, charge) in zip(hand_result.hands, hand_charges):
                            if charge <= 0.0 or effect_gesture is None:
                                continue
                            eased = _smoothstep(charge)
                            if effect_gesture == Gesture.POINTING:
                                _draw_energy_ball(glow_layer, hand, t, eased)
                            elif effect_gesture == Gesture.OPEN_PALM:
                                _draw_lightning(glow_layer, hand, rng, t, eased)

                        between_charge = _update_between_hands_charge(
                            between_hands_state, hand_result.hands, dt,
                        )
                        if between_charge > 0.0:
                            _draw_between_hands_orb(
                                glow_layer, hand_result.hands, t, _smoothstep(between_charge),
                            )

                        output_frame = _apply_glow(output_frame, glow_layer)
                    else:
                        # Reset FX charge state so switching back to an FX
                        # mode later starts the charge-up from zero again,
                        # instead of resuming mid-charge from a stale state.
                        charge_state.clear()
                        between_hands_state["charge"] = 0.0

                    if self.show_skeleton:
                        _draw_hand_skeleton(output_frame, hand_result.hands)

                    if mode in (Mode.GHOST, Mode.COMBO):
                        _draw_hud(output_frame, ghost_progress)

                    if self._record_start_requested:
                        self._record_start_requested = False
                        if not self._recording:
                            h, w = output_frame.shape[:2]
                            path = self._make_recording_path()
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(path, fourcc, DEFAULT_RECORDING_FPS, (w, h))
                            if writer.isOpened():
                                self._video_writer = writer
                                self._current_recording_path = path
                                self._recording = True
                                self._record_frame_count = 0
                                self._record_start_wall = time.monotonic()
                                self.recording_started.emit(path)
                            else:
                                self.recording_failed.emit(
                                    "Could not start recording (video encoder failed to open).",
                                )

                    if self._record_stop_requested:
                        self._record_stop_requested = False
                        if self._recording:
                            self._recording = False
                            if self._video_writer is not None:
                                self._video_writer.release()
                                self._video_writer = None
                            # WHY this matters: the file was written with a
                            # fixed nominal DEFAULT_RECORDING_FPS baked into
                            # its header, but the pipeline (camera + hand
                            # tracking + segmentation) doesn't actually run
                            # at exactly that rate -- it's usually slower
                            # under real load. If the true capture rate was,
                            # say, 12 fps but the file claims 20 fps, a
                            # player shows the same number of frames in less
                            # time than they were actually captured over --
                            # i.e. the clip visibly plays faster than real
                            # life. Measuring frames-written / actual
                            # wall-clock time here gives the *true* average
                            # rate, which gets handed off for the ffmpeg
                            # step to correct (see TranscodeThread).
                            elapsed = (
                                time.monotonic() - self._record_start_wall
                                if self._record_start_wall is not None else 0.0
                            )
                            if elapsed > 0.5 and self._record_frame_count > 0:
                                measured_fps = self._record_frame_count / elapsed
                                measured_fps = max(1.0, min(60.0, measured_fps))
                            else:
                                measured_fps = DEFAULT_RECORDING_FPS
                            self.recording_stopped.emit(self._current_recording_path or "", measured_fps)

                    # WHY written here, after every effect/skeleton/HUD draw
                    # call above: this saves exactly what's on screen, not
                    # a raw feed -- the whole point of recording is to
                    # capture the ghost/energy/lightning effects themselves.
                    if self._recording and self._video_writer is not None:
                        self._video_writer.write(output_frame)
                        self._record_frame_count += 1

                    self._emit_frame(output_frame)

        except (HandTrackerError, RuntimeError) as exc:
            logger.exception("Fatal error in video worker.")
            self.error_occurred.emit(str(exc))
        finally:
            capture.release()
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
            self._recording = False
            self._running = False


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
_STYLE_SHEET = """
QMainWindow { background-color: #14161a; }
QLabel#header {
    color: #f5f5f5;
    font-size: 20px;
    font-weight: 600;
    padding: 6px 2px;
}
QLabel#status { color: #9aa0a6; font-size: 12px; padding: 4px 2px; }
QLabel#video {
    background-color: #0a0b0d;
    border: 1px solid #2a2d33;
    border-radius: 10px;
}
QComboBox, QPushButton {
    background-color: #1f2228;
    color: #f0f0f0;
    border: 1px solid #33363d;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
}
QPushButton:hover { background-color: #2a2d33; }
QPushButton:disabled { color: #5a5d63; }
QPushButton#startButton { background-color: #0f7a3d; border: none; }
QPushButton#startButton:hover { background-color: #128f47; }
QPushButton#startButton[running="true"] { background-color: #a3312b; }
QPushButton#startButton[running="true"]:hover { background-color: #b93831; }
QPushButton#recordButton { background-color: #2a2d33; }
QPushButton#recordButton[recording="true"] { background-color: #c62828; border: none; }
QPushButton#recordButton[recording="true"]:hover { background-color: #d63131; }
QLabel#recIndicator { color: #ff5555; font-size: 12px; font-weight: 600; }
QLabel#qrCode { background-color: #ffffff; border-radius: 8px; color: #333; font-size: 11px; }
QLabel#qrTitle { color: #f0f0f0; font-size: 14px; font-weight: 600; }
QLabel#qrUrl { color: #6db4ff; font-size: 12px; }
QCheckBox { color: #d0d0d0; font-size: 13px; }
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1180, 820)
        self.setStyleSheet(_STYLE_SHEET)

        self._worker: Optional[VideoWorker] = None
        self._recording_active = False
        self._record_start_time: Optional[float] = None
        self._record_elapsed_timer = QTimer(self)
        self._record_elapsed_timer.setInterval(1000)
        self._record_elapsed_timer.timeout.connect(self._update_rec_indicator)

        self._recordings_dir = Path(__file__).resolve().parent / RECORDINGS_DIR_NAME
        self._recordings_dir.mkdir(parents=True, exist_ok=True)
        self._lan_ip = _get_lan_ip()
        self._file_server = RecordingsFileServer(self._recordings_dir)
        self._file_server_ok = self._file_server.start()
        self._transcode_thread: Optional[TranscodeThread] = None

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header = QLabel(APP_TITLE)
        header.setObjectName("header")
        layout.addWidget(header)

        self.video_label = QLabel()
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        self._placeholder_pixmap = _build_placeholder_pixmap()
        self.video_label.setPixmap(self._placeholder_pixmap)
        layout.addWidget(self.video_label, stretch=1)

        controls = QHBoxLayout()
        controls.setSpacing(10)

        controls.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Combo -- Ghost + Energy + Lightning", Mode.COMBO)
        self.mode_combo.addItem("Ghost Mode only", Mode.GHOST)
        self.mode_combo.addItem("Hand FX only (Energy + Lightning)", Mode.HAND_FX)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self.mode_combo)

        controls.addStretch(1)

        self.skeleton_checkbox = QCheckBox("Show hand skeleton")
        self.skeleton_checkbox.setChecked(True)
        self.skeleton_checkbox.stateChanged.connect(self._on_skeleton_toggled)
        controls.addWidget(self.skeleton_checkbox)

        self.recapture_button = QPushButton("Recapture Background")
        self.recapture_button.clicked.connect(self._on_recapture_clicked)
        self.recapture_button.setEnabled(False)
        controls.addWidget(self.recapture_button)

        self.rec_indicator = QLabel("")
        self.rec_indicator.setObjectName("recIndicator")
        controls.addWidget(self.rec_indicator)

        self.record_button = QPushButton("\u25cf Start Recording")
        self.record_button.setObjectName("recordButton")
        self.record_button.clicked.connect(self._on_record_clicked)
        self.record_button.setEnabled(False)
        controls.addWidget(self.record_button)

        self.open_folder_button = QPushButton("Open Recordings Folder")
        self.open_folder_button.clicked.connect(self._on_open_folder_clicked)
        controls.addWidget(self.open_folder_button)

        self.start_button = QPushButton("Start Camera")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self._on_start_stop_clicked)
        controls.addWidget(self.start_button)

        layout.addLayout(controls)

        qr_row = QHBoxLayout()
        qr_row.setSpacing(14)

        self.qr_label = QLabel()
        self.qr_label.setObjectName("qrCode")
        self.qr_label.setFixedSize(140, 140)
        self.qr_label.setAlignment(Qt.AlignCenter)
        self.qr_label.setWordWrap(True)
        qr_row.addWidget(self.qr_label)

        qr_info_layout = QVBoxLayout()
        qr_title = QLabel("Scan to download your clip")
        qr_title.setObjectName("qrTitle")
        qr_info_layout.addWidget(qr_title)

        self.qr_url_label = QLabel("")
        self.qr_url_label.setObjectName("qrUrl")
        self.qr_url_label.setWordWrap(True)
        self.qr_url_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        qr_info_layout.addWidget(self.qr_url_label)

        qr_hint = QLabel("Phone must be on the same Wi-Fi network as this PC.")
        qr_hint.setObjectName("status")
        qr_info_layout.addWidget(qr_hint)
        qr_info_layout.addStretch(1)
        qr_row.addLayout(qr_info_layout, stretch=1)

        self.qr_container = QWidget()
        self.qr_container.setLayout(qr_row)
        self.qr_container.setVisible(False)
        layout.addWidget(self.qr_container)

        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("status")
        layout.addWidget(self.status_label)

        self.setCentralWidget(central)

    # -- actions -------------------------------------------------------- #

    def _on_start_stop_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._stop_worker()
        else:
            self._start_worker()

    def _start_worker(self) -> None:
        self._worker = VideoWorker()
        self._worker.mode = self.mode_combo.currentData()
        self._worker.show_skeleton = self.skeleton_checkbox.isChecked()
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.status_message.connect(self._on_status_message)
        self._worker.background_captured.connect(self._on_background_captured)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.recording_started.connect(self._on_recording_started)
        self._worker.recording_stopped.connect(self._on_recording_stopped)
        self._worker.recording_failed.connect(self._on_recording_failed)
        self._worker.start()

        self.start_button.setText("Stop Camera")
        self.start_button.setProperty("running", "true")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self.recapture_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.status_label.setText("Starting camera...")

    def _stop_worker(self) -> None:
        if self._worker is None:
            return
        self._worker.stop()
        self._worker.wait(2000)
        self._worker = None

        if self._recording_active:
            # The worker's own finally-block already released the video
            # writer even mid-recording (see VideoWorker.run), so the file
            # on disk is safe -- this just brings the UI back in sync.
            self._recording_active = False
            self._record_elapsed_timer.stop()
            self.rec_indicator.setText("")
            self.record_button.setText("\u25cf Start Recording")
            self.record_button.setProperty("recording", "false")
            self.record_button.style().unpolish(self.record_button)
            self.record_button.style().polish(self.record_button)

        self.start_button.setText("Start Camera")
        self.start_button.setProperty("running", "false")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self.recapture_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.video_label.setPixmap(self._placeholder_pixmap)
        self.status_label.setText("Stopped.")

    def _on_mode_changed(self, _index: int) -> None:
        if self._worker is not None:
            self._worker.mode = self.mode_combo.currentData()

    def _on_skeleton_toggled(self, _state: int) -> None:
        if self._worker is not None:
            self._worker.show_skeleton = self.skeleton_checkbox.isChecked()

    def _on_recapture_clicked(self) -> None:
        if self._worker is not None:
            self._worker.request_recapture()

    def _on_record_clicked(self) -> None:
        if self._worker is None:
            return
        if self._recording_active:
            self._worker.stop_recording()
            self.record_button.setEnabled(False)  # re-enabled by _on_recording_stopped
        else:
            self._worker.start_recording()
            self.record_button.setEnabled(False)  # re-enabled by _on_recording_started

    def _on_open_folder_clicked(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._recordings_dir)))

    def _update_rec_indicator(self) -> None:
        if self._record_start_time is None:
            return
        elapsed = int(time.monotonic() - self._record_start_time)
        minutes, seconds = divmod(elapsed, 60)
        self.rec_indicator.setText(f"\u25cf REC {minutes}:{seconds:02d}")

    # -- worker signal handlers ------------------------------------------ #

    @Slot(QImage)
    def _on_frame_ready(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    @Slot(str)
    def _on_status_message(self, message: str) -> None:
        self.status_label.setText(message)

    @Slot()
    def _on_background_captured(self) -> None:
        self.status_label.setText("Ready.")

    @Slot(str)
    def _on_recording_started(self, path: str) -> None:
        self._recording_active = True
        self._record_start_time = time.monotonic()
        self._record_elapsed_timer.start()
        self.record_button.setText("\u25a0 Stop Recording")
        self.record_button.setProperty("recording", "true")
        self.record_button.style().unpolish(self.record_button)
        self.record_button.style().polish(self.record_button)
        self.record_button.setEnabled(True)
        self.status_label.setText(f"Recording to {path}")

    @Slot(str, float)
    def _on_recording_stopped(self, path: str, real_fps: float) -> None:
        self._recording_active = False
        self._record_start_time = None
        self._record_elapsed_timer.stop()
        self.rec_indicator.setText("")
        self.record_button.setText("\u25cf Start Recording")
        self.record_button.setProperty("recording", "false")
        self.record_button.style().unpolish(self.record_button)
        self.record_button.style().polish(self.record_button)
        self.record_button.setEnabled(True)
        self.status_label.setText(f"Saved recording: {path}" if path else "Recording stopped.")
        self._start_transcode(path, real_fps)

    def _start_transcode(self, path: str, real_fps: float) -> None:
        if not path:
            self.qr_container.setVisible(False)
            return

        if shutil.which("ffmpeg") is None:
            # No ffmpeg available -- share the raw mp4v file as-is. It'll
            # open fine in VLC/desktop players (at its original, possibly
            # sped-up, timing); note the caveat for phones.
            self.status_label.setText(
                "Saved (install ffmpeg for reliable phone playback + correct speed).",
            )
            self._show_qr_for_recording(path)
            return

        self.record_button.setEnabled(False)
        self.status_label.setText("Encoding for phone playback...")
        self._transcode_thread = TranscodeThread(path, real_fps)
        self._transcode_thread.finished_ok.connect(self._on_transcode_finished)
        self._transcode_thread.failed.connect(self._on_transcode_failed)
        self._transcode_thread.start()

    @Slot(str)
    def _on_transcode_finished(self, path: str) -> None:
        self.record_button.setEnabled(True)
        self.status_label.setText("Ready to share.")
        self._show_qr_for_recording(path)

    @Slot(str, str)
    def _on_transcode_failed(self, message: str, source_path: str) -> None:
        logger.warning("Transcode failed: %s", message)
        self.record_button.setEnabled(True)
        self.status_label.setText("Saved (playback on some phones may not work -- see log).")
        # The original mp4v file is untouched when encoding fails, so it's
        # still safe to share -- just with the same phone-compatibility
        # caveat as the no-ffmpeg path above.
        self._show_qr_for_recording(source_path)

    def _show_qr_for_recording(self, path: str) -> None:
        if not path:
            self.qr_container.setVisible(False)
            return

        if not self._file_server_ok:
            # Server couldn't bind its port -- no working link to share,
            # so just surface the local file path instead of a dead QR code.
            self.qr_label.setPixmap(QPixmap())
            self.qr_label.setText("Local server\nunavailable")
            self.qr_url_label.setText(path)
            self.qr_container.setVisible(True)
            return

        filename = Path(path).name
        url = f"http://{self._lan_ip}:{LOCAL_SERVER_PORT}/{filename}"
        self.qr_url_label.setText(url)

        pixmap = _make_qr_pixmap(url)
        if pixmap is not None:
            self.qr_label.setPixmap(
                pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation),
            )
        else:
            self.qr_label.setPixmap(QPixmap())
            self.qr_label.setText("Install\nqrcode[pil]\nfor a QR code")
        self.qr_container.setVisible(True)

    @Slot(str)
    def _on_recording_failed(self, message: str) -> None:
        self.record_button.setEnabled(True)
        self.status_label.setText(f"Recording error: {message}")

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        self.status_label.setText(f"Error: {message}")
        self._stop_worker()

    # -- lifecycle -------------------------------------------------------- #

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 -- Qt override
        self._stop_worker()
        self._file_server.stop()
        event.accept()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

    