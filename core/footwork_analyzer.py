"""
Footwork Analyzer for badminton movement analysis
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import LANDMARKS, ANALYSIS_CONFIG
from core.pose_tracker import Pose3D
from core.com_calculator import CenterOfMass
from utils.data_processing import (
    calculate_velocity, calculate_acceleration, detect_direction_changes,
    calculate_path_efficiency, CircularBuffer
)


class FootworkEvent(Enum):
    """Types of footwork events"""
    LANDING = "landing"
    TAKEOFF = "takeoff"
    DIRECTION_CHANGE = "direction_change"
    SPLIT_STEP = "split_step"
    RECOVERY_STEP = "recovery_step"


@dataclass
class FootPosition:
    """Foot position data"""
    left: Tuple[float, float]
    right: Tuple[float, float]
    left_confidence: float
    right_confidence: float
    timestamp: float
    frame_number: int


@dataclass
class FootworkEventData:
    """Footwork event data"""
    event_type: FootworkEvent
    timestamp: float
    frame_number: int
    position: Tuple[float, float]
    velocity: float
    description: str


@dataclass
class FootworkMetrics:
    """Comprehensive footwork metrics"""
    # Basic metrics
    total_steps: int = 0
    step_frequency: float = 0.0  # steps per second
    avg_step_length: float = 0.0  # meters
    max_step_length: float = 0.0

    # Movement metrics
    total_distance: float = 0.0  # meters
    avg_speed: float = 0.0  # m/s
    max_speed: float = 0.0
    acceleration_profile: List[float] = field(default_factory=list)

    # Efficiency metrics
    path_efficiency: float = 1.0  # 0-1 scale
    response_times: List[float] = field(default_factory=list)  # seconds
    avg_response_time: float = 0.0

    # Coverage metrics
    coverage_area: float = 0.0  # square meters
    coverage_ratio: float = 0.0  # percentage of court covered

    # Events
    events: List[FootworkEventData] = field(default_factory=list)
    jump_count: int = 0
    direction_changes: int = 0

    # CoM metrics
    com_stability: float = 0.0  # lower is more stable
    com_height_variation: float = 0.0

    # === Custom metrics for detailed analysis ===

    # Recovery speed (回中速度) - speed returning to center after shot
    recovery_speed: float = 0.0  # m/s
    recovery_time_avg: float = 0.0  # seconds
    recovery_events: int = 0

    # Net approach speed (上网速度) - speed moving towards net
    net_approach_speed: float = 0.0  # m/s
    net_approach_time_avg: float = 0.0  # seconds
    net_approach_events: int = 0

    # Backward movement speed (后退速度) - speed moving to backcourt
    backward_speed: float = 0.0  # m/s
    backward_events: int = 0

    # Lateral movement speed (左右移动速度)
    lateral_speed: float = 0.0  # m/s
    lateral_left_speed: float = 0.0  # m/s
    lateral_right_speed: float = 0.0  # m/s

    # Crouch depth (身体下蹲) - knee bend when preparing/receiving
    crouch_depth_avg: float = 0.0  # normalized 0-1 (lower is deeper)
    crouch_depth_min: float = 1.0  # deepest crouch
    crouch_events: int = 0

    # Split step timing (分腿跳时机)
    split_step_timing_accuracy: float = 0.0  # 0-1 scale
    split_step_before_movement: int = 0  # count of properly timed split steps

    # First step speed (第一步速度)
    first_step_speed: float = 0.0  # m/s
    first_step_time: float = 0.0  # seconds from decision to first step

    # Deceleration ability (减速能力)
    deceleration_rate: float = 0.0  # m/s²
    stopping_distance: float = 0.0  # meters


class FootworkAnalyzer:
    """Analyze footwork patterns from pose data"""

    def __init__(
        self,
        fps: float = 30.0,
        pixel_scale: Optional[float] = None,  # meters per pixel
    ):
        self.fps = fps
        self.pixel_scale = pixel_scale or 1.0
        self.dt = 1.0 / fps

        # Configuration
        self.config = ANALYSIS_CONFIG

        # State tracking
        self.foot_positions: List[FootPosition] = []
        self.com_positions: List[CenterOfMass] = []
        self.events: List[FootworkEventData] = []

        # Real-time buffers
        self._left_foot_buffer = CircularBuffer(30)
        self._right_foot_buffer = CircularBuffer(30)
        self._com_buffer = CircularBuffer(30)

        # Event detection state
        self._last_left_y = None
        self._last_right_y = None
        self._last_velocity = None
        self._in_air = False

        # Court calibration
        self.court_bounds = None  # Will be set from calibration

    def process_frame(
        self,
        pose: Pose3D,
        com: Optional[CenterOfMass] = None,
        frame_number: int = 0,
    ) -> List[FootworkEventData]:
        """
        Process a single frame and detect footwork events

        Returns:
            List of detected events in this frame
        """
        if pose is None:
            return []

        detected_events = []
        timestamp = frame_number / self.fps

        # Extract foot positions
        left_ankle = pose.get_landmark_2d("left_ankle")
        right_ankle = pose.get_landmark_2d("right_ankle")

        if left_ankle is None or right_ankle is None:
            return []

        # Create foot position record
        foot_pos = FootPosition(
            left=left_ankle,
            right=right_ankle,
            left_confidence=pose.visibility[LANDMARKS["left_ankle"]],
            right_confidence=pose.visibility[LANDMARKS["right_ankle"]],
            timestamp=timestamp,
            frame_number=frame_number,
        )
        self.foot_positions.append(foot_pos)

        # Add to buffers
        self._left_foot_buffer.add(left_ankle)
        self._right_foot_buffer.add(right_ankle)

        # Process CoM
        if com is not None:
            self.com_positions.append(com)
            self._com_buffer.add((com.x, com.y))

        # Detect events if we have enough history
        if len(self.foot_positions) >= 3:
            events = self._detect_events(foot_pos, frame_number, timestamp)
            detected_events.extend(events)
            self.events.extend(events)

        # Update state
        self._last_left_y = left_ankle[1]
        self._last_right_y = right_ankle[1]

        return detected_events

    def _detect_events(
        self,
        foot_pos: FootPosition,
        frame_number: int,
        timestamp: float,
    ) -> List[FootworkEventData]:
        """Detect footwork events"""
        events = []

        # Calculate foot velocities
        left_vel = self._calculate_foot_velocity("left")
        right_vel = self._calculate_foot_velocity("right")

        # Detect takeoff (both feet leaving ground)
        if self._detect_takeoff(foot_pos, left_vel, right_vel):
            events.append(FootworkEventData(
                event_type=FootworkEvent.TAKEOFF,
                timestamp=timestamp,
                frame_number=frame_number,
                position=self._get_center_position(foot_pos),
                velocity=(left_vel + right_vel) / 2 if left_vel and right_vel else 0,
                description="Player took off (jump)",
            ))
            self._in_air = True

        # Detect landing
        if self._detect_landing(foot_pos, left_vel, right_vel):
            events.append(FootworkEventData(
                event_type=FootworkEvent.LANDING,
                timestamp=timestamp,
                frame_number=frame_number,
                position=self._get_center_position(foot_pos),
                velocity=0,
                description="Player landed",
            ))
            self._in_air = False

        # Detect direction changes
        direction_change = self._detect_direction_change()
        if direction_change:
            events.append(FootworkEventData(
                event_type=FootworkEvent.DIRECTION_CHANGE,
                timestamp=timestamp,
                frame_number=frame_number,
                position=self._get_center_position(foot_pos),
                velocity=(left_vel + right_vel) / 2 if left_vel and right_vel else 0,
                description="Direction changed",
            ))

        # Detect split step
        split_step = self._detect_split_step(foot_pos)
        if split_step:
            events.append(FootworkEventData(
                event_type=FootworkEvent.SPLIT_STEP,
                timestamp=timestamp,
                frame_number=frame_number,
                position=self._get_center_position(foot_pos),
                velocity=0,
                description="Split step detected",
            ))

        return events

    def _calculate_foot_velocity(self, foot: str) -> Optional[float]:
        """Calculate vertical velocity of foot"""
        buffer = self._left_foot_buffer if foot == "left" else self._right_foot_buffer

        if len(buffer) < 2:
            return None

        pos1 = buffer.get(-2)
        pos2 = buffer.get(-1)

        if pos1 is None or pos2 is None:
            return None

        dy = pos2[1] - pos1[1]
        return dy / self.dt

    def _detect_takeoff(
        self,
        foot_pos: FootPosition,
        left_vel: Optional[float],
        right_vel: Optional[float],
    ) -> bool:
        """Detect if player took off (jump)"""
        if self._in_air:
            return False

        # Both feet moving up rapidly
        threshold = self.config["jump_threshold"]

        if left_vel is not None and right_vel is not None:
            # Negative velocity in y means going up (in image coordinates)
            if left_vel < -threshold and right_vel < -threshold:
                return True

        return False

    def _detect_landing(
        self,
        foot_pos: FootPosition,
        left_vel: Optional[float],
        right_vel: Optional[float],
    ) -> bool:
        """Detect if player landed"""
        if not self._in_air:
            return False

        # Feet stopped moving down (velocity close to zero or negative)
        threshold = self.config["landing_velocity_threshold"]

        if left_vel is not None and right_vel is not None:
            if abs(left_vel) < abs(threshold) and abs(right_vel) < abs(threshold):
                return True

        return False

    def _detect_direction_change(self) -> bool:
        """Detect direction change from CoM trajectory"""
        if len(self._com_buffer) < 5:
            return False

        positions = self._com_buffer.get_all()
        velocities = calculate_velocity(positions, self.fps)

        if len(velocities) < 2:
            return False

        # Check for significant angle change
        v1 = np.array(velocities[-2])
        v2 = np.array(velocities[-1])

        # Lower threshold for minimum velocity (from 5 to 1 pixels/frame)
        # This allows detection of direction changes even during slow movements
        min_velocity_threshold = 1.0  # pixels/frame

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm < min_velocity_threshold or v2_norm < min_velocity_threshold:
            return False

        # Also lower the direction change threshold (from 90° to 45°)
        # This makes detection more sensitive
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        # Use 45° threshold instead of 90° (more sensitive)
        return angle > 45

    def _detect_split_step(self, foot_pos: FootPosition) -> bool:
        """Detect split step (small jump in place for balance)"""
        # This is a simplified detection - can be enhanced
        if len(self.foot_positions) < 10:
            return False

        # Check for quick up-down movement in place
        recent = self.foot_positions[-10:]
        left_positions = [fp.left for fp in recent]

        # Calculate range of motion
        y_range = max(p[1] for p in left_positions) - min(p[1] for p in left_positions)
        x_range = max(p[0] for p in left_positions) - min(p[0] for p in left_positions)

        # Split step has vertical movement but minimal horizontal movement
        return y_range > 0.05 and x_range < 0.1

    def _get_center_position(self, foot_pos: FootPosition) -> Tuple[float, float]:
        """Get center position between feet"""
        return (
            (foot_pos.left[0] + foot_pos.right[0]) / 2,
            (foot_pos.left[1] + foot_pos.right[1]) / 2,
        )

    def calculate_metrics(self, debug: bool = False) -> FootworkMetrics:
        """Calculate comprehensive footwork metrics"""
        import sys
        metrics = FootworkMetrics()

        if len(self.foot_positions) < 2:
            if debug:
                print(f"\n[DEBUG METRICS] Insufficient data: {len(self.foot_positions)} frames", file=sys.stderr)
            return metrics

        # === AUTO-CALIBRATE pixel_scale if not set or seems wrong ===
        if self.pixel_scale is None or self.pixel_scale >= 0.5:
            # Estimate from movement range (assume player covers ~60-80% of court)
            center_positions = [self._get_center_position(fp) for fp in self.foot_positions]
            x_coords = [p[0] for p in center_positions]
            y_coords = [p[1] for p in center_positions]

            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)

            # Assume badminton half-court is ~6.7m x 6.1m
            # If player covers 70% of court, then:
            estimated_court_width = max(x_range, y_range) / 0.7
            self.pixel_scale = 6.7 / estimated_court_width  # meters per pixel

            if debug:
                print(f"[AUTO-CALIBRATION] Detected movement range: {x_range:.1f}x{y_range:.1f} pixels", file=sys.stderr)
                print(f"[AUTO-CALIBRATION] Estimated pixel_scale: {self.pixel_scale:.4f} m/pixel", file=sys.stderr)

        if debug:
            print(f"\n{'='*60} DEBUG METRICS CALCULATION {'='*60}", file=sys.stderr)
            print(f"Frames: {len(self.foot_positions)}", file=sys.stderr)
            print(f"Events detected: {len(self.events)}", file=sys.stderr)
            print(f"Event types: {[e.event_type.value for e in self.events]}", file=sys.stderr)
            print(f"Pixel scale: {self.pixel_scale:.4f} m/pixel", file=sys.stderr)

        # Calculate basic metrics
        duration = self.foot_positions[-1].timestamp - self.foot_positions[0].timestamp

        # Step count (from events)
        metrics.total_steps = len([e for e in self.events
                                    if e.event_type in [FootworkEvent.LANDING, FootworkEvent.SPLIT_STEP]])

        if duration > 0:
            # === Always use velocity peak detection for more accurate step counting ===
            # Event-based detection is unreliable (misses many small steps in badminton)
            center_positions = [self._get_center_position(fp) for fp in self.foot_positions]
            if len(center_positions) > 10:
                velocities = calculate_velocity(center_positions, self.fps)
                speeds = [np.linalg.norm(v) for v in velocities]

                # Count peaks in speed (each peak ≈ one step)
                if len(speeds) > 5:
                    # Simple peak detection: local maxima above threshold
                    from scipy.signal import find_peaks
                    # Use lower threshold for badminton (many small steps)
                    median_speed = np.median(speeds)
                    mean_speed = np.mean(speeds)
                    threshold = max(median_speed * 0.3, mean_speed * 0.25)

                    # Distance parameter: minimum frames between peaks (30fps = 10 frames = 0.33s between steps)
                    # At 3 steps/second, we need distance of ~10 frames
                    min_distance = max(5, int(self.fps / 3))

                    peaks, _ = find_peaks(speeds, height=threshold, distance=min_distance)

                    metrics.total_steps = len(peaks)
                    metrics.step_frequency = len(peaks) / duration

                    if debug:
                        print(f"[STEP DETECTION] Speed threshold: {threshold:.4f}, min_distance: {min_distance}", file=sys.stderr)
                        print(f"[STEP DETECTION] Found {len(peaks)} peaks in {len(speeds)} speed samples", file=sys.stderr)
                        print(f"[STEP DETECTION] Speed stats: min={min(speeds):.4f}, max={max(speeds):.4f}, mean={mean_speed:.4f}, median={median_speed:.4f}", file=sys.stderr)
            else:
                # Fallback to event-based if insufficient data
                metrics.step_frequency = metrics.total_steps / duration

        # Calculate distances and speeds
        center_positions = [self._get_center_position(fp) for fp in self.foot_positions]

        # Apply smoothing to reduce jitter in pose detection
        if len(center_positions) > 5:
            from scipy.ndimage import gaussian_filter1d
            x_coords = [p[0] for p in center_positions]
            y_coords = [p[1] for p in center_positions]

            # Apply stronger Gaussian filter (sigma=5 instead of 2)
            # This reduces jitter and improves path efficiency calculation
            x_smooth = gaussian_filter1d(x_coords, sigma=5)
            y_smooth = gaussian_filter1d(y_coords, sigma=5)

            center_positions = list(zip(x_smooth, y_smooth))

        # Total distance
        total_distance = 0.0
        step_lengths = []

        for i in range(1, len(center_positions)):
            dist = np.linalg.norm(
                np.array(center_positions[i]) - np.array(center_positions[i-1])
            )
            total_distance += dist
            step_lengths.append(dist)

        metrics.total_distance = total_distance * self.pixel_scale

        if step_lengths:
            step_lengths_m = [s * self.pixel_scale for s in step_lengths]
            metrics.avg_step_length = np.mean(step_lengths_m)
            metrics.max_step_length = np.max(step_lengths_m)

        # Speed calculations
        velocities = calculate_velocity(center_positions, self.fps)
        speeds = [np.linalg.norm(v) for v in velocities]
        speeds_m = [s * self.pixel_scale for s in speeds]

        if speeds_m:
            metrics.avg_speed = np.mean(speeds_m)
            metrics.max_speed = np.max(speeds_m)

        # Path efficiency - FIXED CALCULATION FOR BADMINTON
        if len(center_positions) > 2:
            # Calculate how efficiently the player moves by comparing actual path to optimal path
            # For badminton, we use the distance between the two farthest points as the optimal distance

            # Find the two farthest points in the trajectory (sample every 10th point for speed)
            sample_rate = 10
            sampled_positions = center_positions[::sample_rate]

            max_distance_pixels = 0
            for i in range(len(sampled_positions)):
                for j in range(i+1, len(sampled_positions)):
                    dist = np.linalg.norm(np.array(sampled_positions[i]) - np.array(sampled_positions[j]))
                    if dist > max_distance_pixels:
                        max_distance_pixels = dist

            # FIX: Convert both to meters consistently
            optimal_distance = max_distance_pixels * self.pixel_scale  # meters
            actual_distance = total_distance * self.pixel_scale  # meters (FIX: was using pixels!)

            if actual_distance > 0:
                baseline_efficiency = optimal_distance / actual_distance

                # For badminton, even perfect players have baseline_efficiency around 0.15-0.25
                # because they make many small movements and return to center constantly.
                # We scale this to a 0-1 range where 0.25 is considered "excellent"
                # Formula: score = baseline_efficiency / 0.25, capped at 1.0
                metrics.path_efficiency = min(1.0, baseline_efficiency / 0.25)
            else:
                metrics.path_efficiency = 1.0

            if debug:
                print(f"[PATH EFFICIENCY] optimal_distance: {optimal_distance:.2f}m, actual_distance: {actual_distance:.2f}m", file=sys.stderr)
                print(f"[PATH EFFICIENCY] baseline_efficiency: {baseline_efficiency:.4f}", file=sys.stderr)
                print(f"[PATH EFFICIENCY] scaled_efficiency: {metrics.path_efficiency:.4f} (baseline/0.25, capped at 1.0)", file=sys.stderr)

        # Event counts
        metrics.jump_count = len([e for e in self.events if e.event_type == FootworkEvent.TAKEOFF])
        metrics.direction_changes = len([e for e in self.events if e.event_type == FootworkEvent.DIRECTION_CHANGE])

        # CoM metrics
        if self.com_positions:
            com_y = [c.y for c in self.com_positions]
            metrics.com_height_variation = np.std(com_y)
            metrics.com_stability = self._calculate_com_stability()

        # Coverage metrics - CALCULATE MISSING METRIC
        if center_positions and len(center_positions) > 2:
            # Calculate bounding box of movement
            x_coords = [p[0] for p in center_positions]
            y_coords = [p[1] for p in center_positions]

            # Coverage area in pixels²
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            coverage_area_pixels = x_range * y_range

            # Convert to meters²
            metrics.coverage_area = coverage_area_pixels * (self.pixel_scale ** 2)

            # Coverage ratio - assume standard badminton court half (6.1m x 6.7m = ~40.87m²)
            standard_court_area = 40.87  # Half court area
            metrics.coverage_ratio = min(1.0, metrics.coverage_area / standard_court_area)

        # Response time - IMPROVED CALCULATION
        # Calculate from velocity peaks instead of relying on direction_change events
        # This is more reliable and doesn't require event detection
        if len(center_positions) > int(self.fps):
            # Get velocity data
            velocities = calculate_velocity(center_positions, self.fps)
            speeds = [np.linalg.norm(v) for v in velocities]

            if len(speeds) > int(self.fps):
                # Find speed peaks (acceleration moments)
                from scipy.signal import find_peaks
                median_speed = np.median(speeds)
                threshold = median_speed * 0.5  # Lower threshold for more peaks

                peaks, properties = find_peaks(speeds, height=threshold, distance=5)

                # For each peak (acceleration), calculate "response time"
                # Response time = time from low speed to peak speed
                response_times = []
                for peak_idx in peaks:
                    # Find the trough (minimum speed) before this peak
                    start_search = max(0, peak_idx - int(self.fps / 2))  # Look back up to 0.5 seconds
                    if start_search < peak_idx:
                        min_speed_idx = start_search + np.argmin(speeds[start_search:peak_idx + 1])
                        response_time = (peak_idx - min_speed_idx) / self.fps
                        if 0.05 < response_time < 1.0:  # Filter: 50ms to 1 second is reasonable
                            response_times.append(response_time)

                if response_times:
                    metrics.response_times = response_times
                    metrics.avg_response_time = np.mean(response_times)

        metrics.events = self.events

        # === Calculate custom metrics ===

        # Recovery speed (回中速度) - movements toward center
        recovery_speeds = self._calculate_directional_speed(center_positions, toward_center=True)
        if recovery_speeds:
            metrics.recovery_speed = np.mean(recovery_speeds) * self.pixel_scale
            metrics.recovery_events = len(recovery_speeds)

        # Net approach speed (上网速度) - moving upward in frame (assuming camera behind player)
        net_approach_speeds = self._calculate_vertical_speed(center_positions, direction="up")
        if net_approach_speeds:
            metrics.net_approach_speed = np.mean(net_approach_speeds) * self.pixel_scale
            metrics.net_approach_events = len(net_approach_speeds)

        # Backward movement speed (后退速度) - moving downward in frame
        backward_speeds = self._calculate_vertical_speed(center_positions, direction="down")
        if backward_speeds:
            metrics.backward_speed = np.mean(backward_speeds) * self.pixel_scale
            metrics.backward_events = len(backward_speeds)

        # Lateral movement speeds (左右移动速度)
        left_speeds, right_speeds = self._calculate_lateral_speeds(center_positions)
        if left_speeds:
            metrics.lateral_left_speed = np.mean(left_speeds) * self.pixel_scale
        if right_speeds:
            metrics.lateral_right_speed = np.mean(right_speeds) * self.pixel_scale
        all_lateral = left_speeds + right_speeds
        if all_lateral:
            metrics.lateral_speed = np.mean(all_lateral) * self.pixel_scale

        # Crouch depth (身体下蹲) - based on ankle positions relative to hips
        crouch_depths = self._calculate_crouch_depths()
        if crouch_depths:
            metrics.crouch_depth_avg = np.mean(crouch_depths)
            metrics.crouch_depth_min = np.min(crouch_depths)
            metrics.crouch_events = len(crouch_depths)

        # Split step timing accuracy
        metrics.split_step_timing_accuracy = self._calculate_split_step_timing()

        # First step speed (第一步速度)
        first_step_data = self._calculate_first_step_speed()
        if first_step_data:
            metrics.first_step_speed = first_step_data.get('speed', 0.0) * self.pixel_scale
            metrics.first_step_time = first_step_data.get('time', 0.0)

        # Deceleration ability (减速能力)
        decel_data = self._calculate_deceleration(center_positions)
        if decel_data:
            metrics.deceleration_rate = decel_data.get('rate', 0.0) * self.pixel_scale
            metrics.stopping_distance = decel_data.get('distance', 0.0) * self.pixel_scale

        if debug:
            import sys
            print(f"\nCalculated Metrics:", file=sys.stderr)
            print(f"  path_efficiency: {metrics.path_efficiency:.4f}", file=sys.stderr)
            print(f"  step_frequency: {metrics.step_frequency:.4f}", file=sys.stderr)
            print(f"  total_steps: {metrics.total_steps}", file=sys.stderr)
            print(f"  avg_response_time: {metrics.avg_response_time:.4f}", file=sys.stderr)
            print(f"  coverage_ratio: {metrics.coverage_ratio:.4f}", file=sys.stderr)
            print(f"  coverage_area: {metrics.coverage_area:.4f}", file=sys.stderr)
            print(f"  com_stability: {metrics.com_stability:.4f}", file=sys.stderr)
            print(f"  total_distance: {metrics.total_distance:.4f}", file=sys.stderr)
            print(f"  max_speed: {metrics.max_speed:.4f}", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)

        return metrics

    def _calculate_com_stability(self) -> float:
        """Calculate CoM stability (lower is more stable)

        Returns stability in meters (not pixels) for cross-video consistency.
        pixel_scale converts pixel coordinates to meters.
        """
        if len(self.com_positions) < 10:
            return 0.0

        # Calculate variance in CoM position (in meters for cross-video consistency)
        com_x = [c.x * self.pixel_scale for c in self.com_positions[-30:]]
        com_y = [c.y * self.pixel_scale for c in self.com_positions[-30:]]

        return np.std(com_x) + np.std(com_y)

    def calibrate_court(
        self,
        corner_points: List[Tuple[float, float]],  # 4 corners of court
        court_length_m: float = 13.4,  # Badminton singles length
        court_width_m: float = 5.18,   # Badminton singles width
    ) -> float:
        """
        Calibrate court dimensions from known points

        Returns:
            pixel_scale in meters/pixel
        """
        if len(corner_points) != 4:
            raise ValueError("Need exactly 4 corner points")

        # Calculate pixel distances
        width_pixels = np.linalg.norm(np.array(corner_points[1]) - np.array(corner_points[0]))
        length_pixels = np.linalg.norm(np.array(corner_points[3]) - np.array(corner_points[0]))

        # Average scale
        scale_x = court_width_m / width_pixels
        scale_y = court_length_m / length_pixels

        self.pixel_scale = (scale_x + scale_y) / 2
        self.court_bounds = corner_points

        return self.pixel_scale

    def _calculate_directional_speed(
        self,
        positions: List[Tuple[float, float]],
        toward_center: bool = True,
    ) -> List[float]:
        """Calculate speeds for movements toward/away from center"""
        if len(positions) < 3:
            return []

        # Estimate center as the mean position
        center_x = np.mean([p[0] for p in positions])
        center_y = np.mean([p[1] for p in positions])

        velocities = calculate_velocity(positions, self.fps)
        speeds = []

        for i, (pos, vel) in enumerate(zip(positions, velocities)):
            # Vector to/from center
            to_center = np.array([center_x - pos[0], center_y - pos[1]])
            to_center_norm = to_center / (np.linalg.norm(to_center) + 1e-6)

            # Current velocity direction
            vel_mag = np.linalg.norm(vel)
            if vel_mag < 0.001:
                continue

            vel_dir = vel / vel_mag

            # Dot product to check alignment
            alignment = np.dot(vel_dir, to_center_norm)

            # Include if moving in desired direction
            if toward_center and alignment > 0.3:  # Moving toward center
                speeds.append(vel_mag)
            elif not toward_center and alignment < -0.3:  # Moving away from center
                speeds.append(vel_mag)

        return speeds

    def _calculate_vertical_speed(
        self,
        positions: List[Tuple[float, float]],
        direction: str = "up",
    ) -> List[float]:
        """Calculate speeds for vertical movements"""
        if len(positions) < 3:
            return []

        velocities = calculate_velocity(positions, self.fps)
        speeds = []

        for vel in velocities:
            # Check vertical component
            vertical_vel = vel[1]  # Y component

            if direction == "up" and vertical_vel < -0.01:  # Moving up (negative Y)
                speeds.append(abs(vertical_vel))
            elif direction == "down" and vertical_vel > 0.01:  # Moving down (positive Y)
                speeds.append(abs(vertical_vel))

        return speeds

    def _calculate_lateral_speeds(
        self,
        positions: List[Tuple[float, float]],
    ) -> Tuple[List[float], List[float]]:
        """Calculate left and right lateral speeds separately"""
        if len(positions) < 3:
            return [], []

        velocities = calculate_velocity(positions, self.fps)
        left_speeds = []
        right_speeds = []

        for vel in velocities:
            # Check horizontal component
            horizontal_vel = vel[0]  # X component

            if horizontal_vel > 0.01:  # Moving right
                right_speeds.append(horizontal_vel)
            elif horizontal_vel < -0.01:  # Moving left
                left_speeds.append(abs(horizontal_vel))

        return left_speeds, right_speeds

    def _calculate_crouch_depths(self) -> List[float]:
        """Calculate crouch depth based on body geometry"""
        if len(self.foot_positions) < 2:
            return []

        depths = []

        # We need to estimate crouch from relative body positions
        # Use hip-knee-ankle vertical relationship
        for i in range(len(self.foot_positions) - 1):
            # This is a simplified estimate - in practice, we'd need full pose data
            # For now, use foot position variability as a proxy
            # (more stable/low = more crouched)

            # Check if we have CoM data
            if i < len(self.com_positions):
                # Lower CoM relative to foot position suggests deeper crouch
                com = self.com_positions[i]
                foot_y = (self.foot_positions[i].left[1] + self.foot_positions[i].right[1]) / 2

                # Normalize crouch depth (0-1, lower is deeper)
                crouch_ratio = abs(com.y - foot_y)
                depths.append(crouch_ratio)

        return depths

    def _calculate_split_step_timing(self) -> float:
        """Calculate split step timing accuracy"""
        split_steps = [e for e in self.events if e.event_type == FootworkEvent.SPLIT_STEP]

        if not split_steps:
            return 0.0

        # Check if split steps are followed by directional movements
        well_timed = 0
        for split_step in split_steps:
            # Look for movement events shortly after split step
            subsequent_events = [
                e for e in self.events
                if 0 < (e.timestamp - split_step.timestamp) < 0.5  # Within 0.5 seconds
                and e.event_type in [FootworkEvent.DIRECTION_CHANGE, FootworkEvent.TAKEOFF]
            ]
            if subsequent_events:
                well_timed += 1

        if split_steps:
            return well_timed / len(split_steps)

        return 0.0

    def _calculate_first_step_speed(self) -> Dict[str, float]:
        """Calculate first step speed after decision (direction change or landing)"""
        decision_events = [
            e for e in self.events
            if e.event_type in [FootworkEvent.DIRECTION_CHANGE, FootworkEvent.LANDING]
        ]

        if not decision_events or len(self.foot_positions) < 3:
            return {}

        first_step_speeds = []
        first_step_times = []

        for event in decision_events:
            # Find movement shortly after event
            event_idx = event.frame_number
            if event_idx < len(self.foot_positions) - 5:
                # Calculate displacement in first few frames after event
                start_pos = self._get_center_position(self.foot_positions[event_idx])
                end_pos = self._get_center_position(self.foot_positions[min(event_idx + 5, len(self.foot_positions) - 1)])

                displacement = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
                time_elapsed = 5 / self.fps

                if time_elapsed > 0:
                    speed = displacement / time_elapsed
                    first_step_speeds.append(speed)
                    first_step_times.append(time_elapsed)

        if first_step_speeds:
            return {
                'speed': np.mean(first_step_speeds),
                'time': np.mean(first_step_times),
            }

        return {}

    def _calculate_deceleration(
        self,
        positions: List[Tuple[float, float]],
    ) -> Dict[str, float]:
        """Calculate deceleration ability"""
        if len(positions) < 5:
            return {}

        velocities = calculate_velocity(positions, self.fps)
        accelerations = calculate_acceleration(velocities, self.fps)

        # Find moments of strong deceleration (negative acceleration)
        decelerations = [a[0] for a in accelerations if a[0] < -0.1]  # Threshold

        if decelerations:
            avg_decel = np.mean(decelerations)
            # Estimate stopping distance: v^2 / (2*a)
            return {
                'rate': abs(avg_decel),
                'distance': 0.0,  # Would need more context to calculate accurately
            }

        return {}

    def reset(self) -> None:
        """Reset analyzer state"""
        self.foot_positions.clear()
        self.com_positions.clear()
        self.events.clear()
        self._left_foot_buffer.clear()
        self._right_foot_buffer.clear()
        self._com_buffer.clear()
        self._in_air = False

    def get_foot_trajectory(self, foot: str = "center") -> List[Tuple[float, float]]:
        """Get foot trajectory for visualization"""
        if foot == "center":
            return [self._get_center_position(fp) for fp in self.foot_positions]
        elif foot == "left":
            return [fp.left for fp in self.foot_positions]
        elif foot == "right":
            return [fp.right for fp in self.foot_positions]
        return []
