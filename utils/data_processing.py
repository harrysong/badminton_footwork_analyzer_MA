"""
Data processing utilities for smoothing and trajectory analysis
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from typing import List, Optional, Tuple
from collections import deque
import cv2


class KalmanFilter1D:
    """Simple 1D Kalman filter for smoothing positions"""

    def __init__(
        self,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-2,
        initial_value: float = 0.0,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate = initial_value
        self.error_estimate = 1.0

    def update(self, measurement: float) -> float:
        """Update filter with new measurement"""
        # Prediction update
        self.error_estimate += self.process_noise

        # Measurement update
        kalman_gain = self.error_estimate / (self.error_estimate + self.measurement_noise)
        self.estimate += kalman_gain * (measurement - self.estimate)
        self.error_estimate = (1 - kalman_gain) * self.error_estimate

        return self.estimate

    def reset(self, initial_value: float = 0.0) -> None:
        """Reset filter"""
        self.estimate = initial_value
        self.error_estimate = 1.0


class KalmanFilter2D:
    """2D Kalman filter for smoothing (x, y) positions"""

    def __init__(
        self,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-2,
        initial_position: Optional[Tuple[float, float]] = None,
    ):
        self.x_filter = KalmanFilter1D(process_noise, measurement_noise)
        self.y_filter = KalmanFilter1D(process_noise, measurement_noise)

        if initial_position is not None:
            self.x_filter.estimate = initial_position[0]
            self.y_filter.estimate = initial_position[1]

    def update(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Update filter with new position"""
        x = self.x_filter.update(position[0])
        y = self.y_filter.update(position[1])
        return (x, y)

    def reset(self, initial_position: Optional[Tuple[float, float]] = None) -> None:
        """Reset filter"""
        x0 = initial_position[0] if initial_position else 0.0
        y0 = initial_position[1] if initial_position else 0.0
        self.x_filter.reset(x0)
        self.y_filter.reset(y0)


class TrajectorySmoother:
    """Smooth trajectory using multiple methods"""

    def __init__(self, method: str = "kalman", window_size: int = 7, **kwargs):
        self.method = method
        self.window_size = window_size
        self.kwargs = kwargs
        self._buffer: deque = deque(maxlen=window_size * 2)
        self._kalman: Optional[KalmanFilter2D] = None

        if method == "kalman":
            process_noise = kwargs.get("process_noise", 1e-5)
            measurement_noise = kwargs.get("measurement_noise", 1e-2)
            self._kalman = KalmanFilter2D(process_noise, measurement_noise)

    def smooth(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Smooth a single position"""
        if self.method == "kalman":
            return self._kalman.update(position)

        self._buffer.append(position)
        return position

    def smooth_batch(
        self, positions: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Smooth a batch of positions"""
        if len(positions) < self.window_size:
            return positions

        x = np.array([p[0] for p in positions])
        y = np.array([p[1] for p in positions])

        if self.method == "savgol":
            # Savitzky-Golay filter
            polyorder = self.kwargs.get("polyorder", 3)
            x_smooth = savgol_filter(x, self.window_size, polyorder)
            y_smooth = savgol_filter(y, self.window_size, polyorder)

        elif self.method == "gaussian":
            # Gaussian smoothing
            sigma = self.kwargs.get("sigma", 2.0)
            x_smooth = gaussian_filter1d(x, sigma)
            y_smooth = gaussian_filter1d(y, sigma)

        elif self.method == "moving_avg":
            # Moving average
            x_smooth = np.convolve(
                x, np.ones(self.window_size) / self.window_size, mode="same"
            )
            y_smooth = np.convolve(
                y, np.ones(self.window_size) / self.window_size, mode="same"
            )

        else:
            return positions

        return list(zip(x_smooth, y_smooth))

    def reset(self) -> None:
        """Reset smoother state"""
        self._buffer.clear()
        if self._kalman is not None:
            self._kalman.reset()


def calculate_velocity(
    positions: List[Tuple[float, float]], fps: float = 30.0
) -> List[Tuple[float, float]]:
    """Calculate velocity from positions"""
    if len(positions) < 2:
        return [(0.0, 0.0)]

    dt = 1.0 / fps
    velocities = [(0.0, 0.0)]

    for i in range(1, len(positions)):
        vx = (positions[i][0] - positions[i - 1][0]) / dt
        vy = (positions[i][1] - positions[i - 1][1]) / dt
        velocities.append((vx, vy))

    return velocities


def calculate_acceleration(
    velocities: List[Tuple[float, float]], fps: float = 30.0
) -> List[Tuple[float, float]]:
    """Calculate acceleration from velocities"""
    if len(velocities) < 2:
        return [(0.0, 0.0)]

    dt = 1.0 / fps
    accelerations = [(0.0, 0.0)]

    for i in range(1, len(velocities)):
        ax = (velocities[i][0] - velocities[i - 1][0]) / dt
        ay = (velocities[i][1] - velocities[i - 1][1]) / dt
        accelerations.append((ax, ay))

    return accelerations


def calculate_magnitude(vectors: List[Tuple[float, float]]) -> np.ndarray:
    """Calculate magnitude of vectors"""
    return np.sqrt(np.sum(np.square(vectors), axis=1))


def detect_direction_changes(
    velocities: List[Tuple[float, float]],
    angle_threshold: float = 90.0,
    min_speed: float = 5.0,
) -> List[int]:
    """Detect direction change points in trajectory"""
    direction_changes = []

    for i in range(1, len(velocities)):
        v1 = np.array(velocities[i - 1])
        v2 = np.array(velocities[i])

        # Skip if speed is too low
        if np.linalg.norm(v1) < min_speed or np.linalg.norm(v2) < min_speed:
            continue

        # Calculate angle between velocity vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        if angle > angle_threshold:
            direction_changes.append(i)

    return direction_changes


def calculate_path_efficiency(
    actual_path: List[Tuple[float, float]],
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
) -> float:
    """
    Calculate path efficiency as optimal distance / actual distance
    Returns value between 0 and 1 (1 = perfectly efficient)
    """
    actual_distance = sum(
        np.linalg.norm(
            np.array(actual_path[i]) - np.array(actual_path[i - 1])
        )
        for i in range(1, len(actual_path))
    )

    optimal_distance = np.linalg.norm(np.array(end_point) - np.array(start_point))

    if actual_distance == 0:
        return 1.0

    return optimal_distance / actual_distance


def calculate_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate centroid of points"""
    if not points:
        return (0.0, 0.0)

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    return (np.mean(x_coords), np.mean(y_coords))


def calculate_bounding_box(
    points: List[Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Calculate bounding box of points"""
    if not points:
        return ((0.0, 0.0), (0.0, 0.0))

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    min_point = (min(x_coords), min(y_coords))
    max_point = (max(x_coords), max(y_coords))

    return (min_point, max_point)


def interpolate_missing_values(
    values: List[Optional[float]], method: str = "linear"
) -> List[float]:
    """Interpolate missing (None) values"""
    result = []
    last_valid_idx = -1

    for i, val in enumerate(values):
        if val is not None:
            if last_valid_idx >= 0 and i - last_valid_idx > 1:
                # Interpolate between last_valid_idx and i
                start_val = result[last_valid_idx]
                end_val = val
                num_steps = i - last_valid_idx

                for j in range(1, num_steps):
                    if method == "linear":
                        t = j / num_steps
                        interpolated = start_val + (end_val - start_val) * t
                    else:
                        interpolated = start_val
                    result.append(interpolated)

            result.append(val)
            last_valid_idx = len(result) - 1
        else:
            # Will be filled later or use last known value
            result.append(val if val is not None else (result[-1] if result else 0.0))

    return result


class CircularBuffer:
    """Fixed-size circular buffer for real-time processing"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)

    def add(self, value) -> None:
        """Add value to buffer"""
        self._buffer.append(value)

    def get(self, index: int = -1):
        """Get value at index (negative indices supported)"""
        try:
            return self._buffer[index]
        except IndexError:
            return None

    def get_all(self) -> list:
        """Get all values as list"""
        return list(self._buffer)

    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self._buffer) == self.max_size

    def clear(self) -> None:
        """Clear buffer"""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)
