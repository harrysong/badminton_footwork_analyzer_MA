"""
Video I/O utilities for reading and writing video files
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Video metadata"""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str

    def __repr__(self) -> str:
        return (
            f"VideoInfo({self.width}x{self.height}, "
            f"{self.fps:.2f}fps, {self.duration:.2f}s)"
        )


class VideoReader:
    """Video reader with frame-by-frame access"""

    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        self._cap = None
        self._info = None
        self._current_frame = 0

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self._open()

    def _open(self) -> None:
        """Open video file"""
        self._cap = cv2.VideoCapture(str(self.video_path))

        if not self._cap.isOpened():
            raise IOError(f"Failed to open video: {self.video_path}")

        # Extract metadata
        self._info = VideoInfo(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            / self._cap.get(cv2.CAP_PROP_FPS),
            codec=self._get_codec_name(),
        )

        logger.info(f"Opened video: {self._info}")

    def _get_codec_name(self) -> str:
        """Get codec name from fourcc"""
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    @property
    def info(self) -> VideoInfo:
        """Get video metadata"""
        return self._info

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame"""
        ret, frame = self._cap.read()
        if ret:
            self._current_frame += 1
            return frame
        return None

    def read_frames(self, start: int = 0, end: Optional[int] = None) -> Iterator[np.ndarray]:
        """Read frames from start to end"""
        self.set_frame_position(start)

        end = end or self._info.frame_count
        frame_idx = start

        while frame_idx < end:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame
            frame_idx += 1

    def set_frame_position(self, frame_number: int) -> None:
        """Seek to specific frame"""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self._current_frame = frame_number

    def get_frame_position(self) -> int:
        """Get current frame number"""
        return self._current_frame

    def get_timestamp(self) -> float:
        """Get current timestamp in seconds"""
        return self._current_frame / self._info.fps

    def __iter__(self) -> Iterator[np.ndarray]:
        """Make video reader iterable"""
        self.set_frame_position(0)
        return self

    def __next__(self) -> np.ndarray:
        """Get next frame"""
        frame = self.read_frame()
        if frame is None:
            raise StopIteration
        return frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def release(self) -> None:
        """Release video capture"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info(f"Released video: {self.video_path}")


class VideoWriter:
    """Video writer with automatic codec selection"""

    # Platform-specific codec preferences
    CODEC_PRIORITY = [
        "avc1",   # H.264 - widely compatible
        "H264",   # Alternative H.264
        "mp4v",   # MP4 container
        "X264",   # x264 encoder
        "MJPG",   # Motion JPEG - fallback
    ]

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        resolution: Tuple[int, int],
        codec: str = None,
    ):
        self.output_path = Path(output_path)
        self.fps = max(1.0, min(120.0, fps))  # Clamp fps to reasonable range
        self.resolution = (max(1, resolution[0]), max(1, resolution[1]))  # Ensure positive dimensions
        self.codec = codec
        self._writer = None

        logger.info(f"VideoWriter init: path={self.output_path}, fps={self.fps}, resolution={self.resolution}")

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._open()

    def _open(self) -> None:
        """Initialize video writer with codec fallback"""
        # Try specified codec or auto-select
        codecs_to_try = [self.codec] if self.codec else self.CODEC_PRIORITY
        codecs_to_try = [c for c in codecs_to_try if c is not None]

        last_error = None
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self._writer = cv2.VideoWriter(
                    str(self.output_path), fourcc, self.fps, self.resolution
                )

                if self._writer.isOpened():
                    self.codec = codec  # Store the working codec
                    logger.info(f"Created video writer with codec '{codec}': {self.resolution} @ {self.fps}fps")
                    return

                last_error = f"Codec '{codec}' failed to open writer"

            except Exception as e:
                last_error = f"Codec '{codec}' error: {str(e)}"
                logger.warning(last_error)
                continue

        # If we get here, all codecs failed
        error_msg = f"Failed to create video writer: {self.output_path}. Tried codecs: {codecs_to_try}. Last error: {last_error}"
        logger.error(error_msg)
        raise IOError(error_msg)

    def write(self, frame: np.ndarray) -> None:
        """Write a frame"""
        if frame.shape[:2] != (self.resolution[1], self.resolution[0]):
            frame = cv2.resize(frame, self.resolution)
        self._writer.write(frame)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def release(self) -> None:
        """Release video writer"""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(f"Saved video: {self.output_path}")


class FrameBuffer:
    """Circular buffer for frame processing"""

    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self._buffer: list = []
        self._timestamps: list = []
        self._frame_numbers: list = []

    def add(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        frame_number: Optional[int] = None,
    ) -> None:
        """Add frame to buffer"""
        self._buffer.append(frame.copy())
        self._timestamps.append(timestamp)
        self._frame_numbers.append(frame_number)

        if len(self._buffer) > self.max_size:
            self._buffer.pop(0)
            self._timestamps.pop(0)
            self._frame_numbers.pop(0)

    def get(
        self, index: int
    ) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int]]:
        """Get frame at index"""
        if 0 <= index < len(self._buffer):
            return (
                self._buffer[index],
                self._timestamps[index],
                self._frame_numbers[index],
            )
        return None, None, None

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int]]:
        """Get most recent frame"""
        if self._buffer:
            return (
                self._buffer[-1],
                self._timestamps[-1],
                self._frame_numbers[-1],
            )
        return None, None, None

    def get_all(self) -> list:
        """Get all frames"""
        return list(self._buffer)

    def clear(self) -> None:
        """Clear buffer"""
        self._buffer.clear()
        self._timestamps.clear()
        self._frame_numbers.clear()

    def __len__(self) -> int:
        return len(self._buffer)


def resize_frame(
    frame: np.ndarray,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    scale: Optional[float] = None,
) -> np.ndarray:
    """Resize frame while maintaining aspect ratio"""
    h, w = frame.shape[:2]

    if scale is not None:
        new_w, new_h = int(w * scale), int(h * scale)
    elif max_width is not None and max_height is not None:
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)
        new_w, new_h = int(w * scale), int(h * scale)
    elif max_width is not None:
        scale = max_width / w
        new_w, new_h = int(w * scale), int(h * scale)
    elif max_height is not None:
        scale = max_height / h
        new_w, new_h = int(w * scale), int(h * scale)
    else:
        return frame

    return cv2.resize(frame, (new_w, new_h))


def draw_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    bg_color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Draw text with optional background"""
    x, y = position

    if bg_color is not None:
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            frame, (x, y - text_h - 10), (x + text_w, y + 5), bg_color, -1
        )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return frame
