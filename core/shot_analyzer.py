"""
Shot detection and biomechanics analysis for badminton
Analyzes hitting mechanics, tactical geometry, and rhythm control
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from core.pose_tracker import Pose3D
from core.com_calculator import CenterOfMass


class ShotType(Enum):
    """Types of badminton shots"""
    SMASH = "smash"
    CLEAR = "clear"
    DROP = "drop"
    DRIVE = "drive"
    NET_SHOT = "net_shot"
    LIFT = "lift"
    UNKNOWN = "unknown"


@dataclass
class ShotEvent:
    """Represents a single shot/hitting event"""
    timestamp: float
    frame_number: int
    shot_type: ShotType
    racket_position: Tuple[float, float]  # Racket head position (estimated)
    hit_point_height: float  # Normalized height from ground
    body_pose: Pose3D  # Full body pose at hit moment
    com_position: CenterOfMass

    # Biomechanics data
    kinetic_chain_score: float = 0.0  # 0-1, synchronization of body segments
    swing_angular_velocity: float = 0.0  # rad/s

    # Tactical data
    hit_position: Tuple[float, float] = (0.5, 0.5)  # Position on court (normalized)
    target_position: Optional[Tuple[float, float]] = None  # Where the ball went (if detected)


@dataclass
class BiomechanicsMetrics:
    """Biomechanics analysis metrics"""
    # Kinetic chain
    kinetic_chain_synchronization: float = 0.0  # 0-1, how well force transfers
    segment_timing_variance: float = 0.0  # Lower is better

    # Hit point analysis
    highest_hit_point: float = 0.0  # Normalized height
    avg_hit_point_height: float = 0.0
    hit_point_consistency: float = 0.0  # Lower is more consistent

    # Swing mechanics
    avg_swing_angular_velocity: float = 0.0  # rad/s
    max_swing_angular_velocity: float = 0.0
    swing_acceleration: float = 0.0

    # Power transfer
    power_transfer_efficiency: float = 0.0  # 0-1


@dataclass
class TacticalGeometryMetrics:
    """Tactical geometry and spatial analysis"""
    # Shot distribution (16-grid court zones)
    shot_distribution_16grid: Dict[str, int] = field(default_factory=dict)
    preferred_zones: List[str] = field(default_factory=list)

    # Shot line analysis
    optimal_line_accuracy: float = 0.0  # 0-1, how close to geometric optimal
    line_variance: float = 0.0

    # Court coverage exposure
    average_exposed_area: float = 0.0  # Square meters
    max_exposed_area: float = 0.0
    exposure_frequency: Dict[str, float] = field(default_factory=dict)


@dataclass
class RhythmControlMetrics:
    """Rhythm and timing control metrics"""
    # Inter-shot timing
    inter_shot_intervals: List[float] = field(default_factory=list)
    avg_rally_tempo: float = 0.0  # Seconds between shots
    rhythm_variance: float = 0.0

    # Shot consistency
    swing_consistency_score: float = 0.0  # 0-1
    position_shot_variance: Dict[str, float] = field(default_factory=dict)

    # Rally patterns
    avg_rally_length: float = 0.0  # Number of shots per rally
    attack_frequency: float = 0.0  # Percentage of attacking shots


class ShotDetector:
    """Detect shot events from pose sequence"""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.pose_history: List[Tuple[float, Pose3D, CenterOfMass]] = []
        self.max_history = 15  # Keep about 0.5 seconds of history

        # Detection thresholds
        self.arm_velocity_threshold = 2.0  # Normalized units per second
        self.position_change_threshold = 0.15

    def process_frame(
        self,
        pose: Pose3D,
        com: CenterOfMass,
        frame_number: int,
    ) -> Optional[ShotEvent]:
        """
        Process a frame and detect if a shot occurred

        Returns ShotEvent if shot detected, None otherwise
        """
        if pose is None or com is None:
            return None

        timestamp = frame_number / self.fps
        self.pose_history.append((timestamp, pose, com))

        # Keep history bounded
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)

        # Need enough history for detection
        if len(self.pose_history) < 5:
            return None

        # Check for shot characteristics
        shot_detected = self._detect_shot_movement()

        if shot_detected:
            shot_event = self._create_shot_event(
                timestamp, frame_number, pose, com
            )
            return shot_event

        return None

    def _detect_shot_movement(self) -> bool:
        """Detect if a shot-like movement occurred"""
        if len(self.pose_history) < 3:
            return False

        # Get recent poses
        recent = self.pose_history[-3:]
        poses = [p[1] for p in recent]

        # Check for rapid arm movement (indicative of swing)
        left_wrist_vel = self._calculate_velocity(poses, "left_wrist")
        right_wrist_vel = self._calculate_velocity(poses, "right_wrist")

        # Check for arm elevation change (preparation to hit)
        left_elbow_y = poses[-1].get_landmark_2d("left_elbow")
        right_elbow_y = poses[-1].get_landmark_2d("right_elbow")
        left_shoulder_y = poses[-1].get_landmark_2d("left_shoulder")
        right_shoulder_y = poses[-1].get_landmark_2d("right_shoulder")

        # Detect if one arm is raised (hitting position)
        arm_raised = False
        if left_elbow_y and left_shoulder_y:
            arm_raised = arm_raised or (left_elbow_y[1] < left_shoulder_y[1] - 0.05)
        if right_elbow_y and right_shoulder_y:
            arm_raised = arm_raised or (right_elbow_y[1] < right_shoulder_y[1] - 0.05)

        # Detect rapid arm motion
        rapid_motion = (left_wrist_vel > self.arm_velocity_threshold or
                       right_wrist_vel > self.arm_velocity_threshold)

        return rapid_motion and arm_raised

    def _calculate_velocity(self, poses: List[Pose3D], landmark: str) -> float:
        """Calculate velocity for a landmark"""
        if len(poses) < 2:
            return 0.0

        pos1 = poses[-2].get_landmark_2d(landmark)
        pos2 = poses[-1].get_landmark_2d(landmark)

        if pos1 is None or pos2 is None:
            return 0.0

        dt = 1.0 / self.fps
        displacement = np.linalg.norm(np.array(pos2) - np.array(pos1))
        return displacement / dt

    def _create_shot_event(
        self,
        timestamp: float,
        frame_number: int,
        pose: Pose3D,
        com: CenterOfMass,
    ) -> ShotEvent:
        """Create a shot event from current pose"""
        # Estimate racket position (beyond wrist)
        right_wrist = pose.get_landmark_2d("right_wrist")
        right_elbow = pose.get_landmark_2d("right_elbow")

        if right_wrist and right_elbow:
            # Extend beyond wrist to estimate racket head
            direction = np.array(right_wrist) - np.array(right_elbow)
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            racket_pos = tuple(np.array(right_wrist) + direction * 0.15)
        else:
            racket_pos = (0.5, 0.5)

        # Detect shot type based on body position
        shot_type = self._classify_shot_type(pose, com)

        # Calculate hit point height (from ankle to wrist)
        right_ankle = pose.get_landmark_2d("right_ankle")
        if right_wrist and right_ankle:
            hit_point_height = abs(right_wrist[1] - right_ankle[1])
        else:
            hit_point_height = 0.5

        return ShotEvent(
            timestamp=timestamp,
            frame_number=frame_number,
            shot_type=shot_type,
            racket_position=racket_pos,
            hit_point_height=hit_point_height,
            body_pose=pose,
            com_position=com,
        )

    def _classify_shot_type(self, pose: Pose3D, com: CenterOfMass) -> ShotType:
        """Classify shot type from body pose"""
        # This is a simplified classification
        # In practice, you'd use ML or more sophisticated rules

        right_wrist = pose.get_landmark_2d("right_wrist")
        right_shoulder = pose.get_landmark_2d("right_shoulder")

        if not right_wrist or not right_shoulder:
            return ShotType.UNKNOWN

        # Height of wrist relative to shoulder
        relative_height = right_shoulder[1] - right_wrist[1]

        # If wrist is significantly above shoulder, likely a smash or clear
        if relative_height > 0.1:
            return ShotType.SMASH if com.y < 0.5 else ShotType.CLEAR

        # If wrist is around shoulder level
        elif relative_height > -0.05:
            return ShotType.DRIVE

        # If wrist is below shoulder
        else:
            if com.y > 0.6:  # Near net
                return ShotType.NET_SHOT
            else:
                return ShotType.DROP

    def reset(self):
        """Reset detector state"""
        self.pose_history.clear()


class BiomechanicsAnalyzer:
    """Analyze hitting biomechanics"""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.shots: List[ShotEvent] = []

    def analyze_shots(
        self,
        shots: List[ShotEvent],
    ) -> BiomechanicsMetrics:
        """Analyze biomechanics from shot events"""
        if not shots:
            return BiomechanicsMetrics()

        metrics = BiomechanicsMetrics()

        # Calculate kinetic chain synchronization
        metrics.kinetic_chain_synchronization = self._calculate_kinetic_chain_sync(shots)

        # Analyze hit points
        hit_heights = [s.hit_point_height for s in shots]
        metrics.highest_hit_point = max(hit_heights) if hit_heights else 0.0
        metrics.avg_hit_point_height = np.mean(hit_heights) if hit_heights else 0.0
        metrics.hit_point_consistency = np.std(hit_heights) if len(hit_heights) > 1 else 0.0

        # Swing velocities (would need to track over time)
        # For now, use simplified calculation
        metrics.avg_swing_angular_velocity = 15.0  # Placeholder rad/s
        metrics.max_swing_angular_velocity = 25.0  # Placeholder rad/s

        # Power transfer efficiency (simplified)
        metrics.power_transfer_efficiency = self._calculate_power_transfer(shots)

        return metrics

    def _calculate_kinetic_chain_sync(self, shots: List[ShotEvent]) -> float:
        """Calculate how well body segments synchronize during shots"""
        if not shots:
            return 0.0

        sync_scores = []
        for shot in shots[:10]:  # Analyze first 10 shots
            pose = shot.body_pose

            # Get key landmarks
            right_ankle = pose.get_landmark_3d("right_ankle")
            right_knee = pose.get_landmark_3d("right_knee")
            right_hip = pose.get_landmark_3d("right_hip")
            right_shoulder = pose.get_landmark_3d("right_shoulder")
            right_elbow = pose.get_landmark_3d("right_elbow")
            right_wrist = pose.get_landmark_3d("right_wrist")

            if all([right_ankle, right_knee, right_hip, right_shoulder, right_elbow, right_wrist]):
                # Calculate timing alignment (simplified - in practice would use velocity timing)
                # Check if joints form a proper kinetic chain
                chain = [
                    np.array(right_ankle),
                    np.array(right_knee),
                    np.array(right_hip),
                    np.array(right_shoulder),
                    np.array(right_elbow),
                    np.array(right_wrist),
                ]

                # Check alignment (lower variance = better synchronization)
                angles = []
                for i in range(len(chain) - 1):
                    for j in range(i + 2, len(chain)):
                        vec1 = chain[i+1] - chain[i]
                        vec2 = chain[j] - chain[j-1]
                        if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                            cos_angle = np.dot(vec1, vec2) / (
                                np.linalg.norm(vec1) * np.linalg.norm(vec2)
                            )
                            angles.append(np.arccos(np.clip(cos_angle, -1, 1)))

                if angles:
                    sync_scores.append(1.0 / (1.0 + np.std(angles)))

        return np.mean(sync_scores) if sync_scores else 0.5

    def _calculate_power_transfer(self, shots: List[ShotEvent]) -> float:
        """Calculate power transfer efficiency through kinetic chain"""
        # Simplified: based on body extension and hit point height
        if not shots:
            return 0.0

        power_scores = []
        for shot in shots:
            pose = shot.body_pose

            # Check body extension (are legs extended for power?)
            right_ankle = pose.get_landmark_2d("right_ankle")
            right_hip = pose.get_landmark_2d("right_hip")

            if right_ankle and right_hip:
                extension = abs(right_hip[1] - right_ankle[1])
                power = min(1.0, extension * 2.0)  # Normalize to 0-1
                power_scores.append(power)

        return np.mean(power_scores) if power_scores else 0.5


class TacticalGeometryAnalyzer:
    """Analyze tactical geometry and spatial patterns"""

    def __init__(self):
        self.shots: List[ShotEvent] = []

    def analyze_tactics(self, shots: List[ShotEvent]) -> TacticalGeometryMetrics:
        """Analyze tactical patterns"""
        if not shots:
            return TacticalGeometryMetrics()

        metrics = TacticalGeometryMetrics()

        # 16-grid shot distribution
        metrics.shot_distribution_16grid = self._calculate_16grid_distribution(shots)
        metrics.preferred_zones = self._get_preferred_zones(metrics.shot_distribution_16grid)

        # Line accuracy (simplified - would need opponent position data)
        metrics.optimal_line_accuracy = 0.65  # Placeholder

        # Exposure analysis
        exposure_data = self._calculate_exposure(shots)
        metrics.average_exposed_area = exposure_data.get('avg', 0.0)
        metrics.max_exposed_area = exposure_data.get('max', 0.0)

        return metrics

    def _calculate_16grid_distribution(
        self,
        shots: List[ShotEvent],
    ) -> Dict[str, int]:
        """Calculate shot distribution across 16 court zones"""
        # Divide court into 4x4 grid
        distribution = defaultdict(int)

        for shot in shots:
            x, y = shot.hit_position
            row = int(min(3, y * 4))
            col = int(min(3, x * 4))
            zone = f"{row}{col}"
            distribution[zone] += 1

        return dict(distribution)

    def _get_preferred_zones(
        self,
        distribution: Dict[str, int],
    ) -> List[str]:
        """Get most frequently used zones"""
        if not distribution:
            return []

        sorted_zones = sorted(distribution.items(), key=lambda x: -x[1])
        total = sum(distribution.values())

        # Return zones that account for top 50% of shots
        cumulative = 0
        preferred = []
        for zone, count in sorted_zones:
            preferred.append(zone)
            cumulative += count
            if cumulative >= total * 0.5:
                break

        return preferred

    def _calculate_exposure(
        self,
        shots: List[ShotEvent],
    ) -> Dict[str, float]:
        """Calculate court exposure after shots"""
        # Simplified: estimate based on shot position
        exposures = []

        for shot in shots:
            # Distance from center represents potential exposure
            x, y = shot.hit_position
            center_dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)

            # Exposure increases with distance from center
            exposed_area = center_dist * 20.0  # Simplified conversion to m²
            exposures.append(exposed_area)

        if exposures:
            return {
                'avg': np.mean(exposures),
                'max': max(exposures),
            }

        return {'avg': 0.0, 'max': 0.0}


class RhythmControlAnalyzer:
    """Analyze rhythm and timing control"""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.shot_timestamps: List[float] = []

    def analyze_rhythm(
        self,
        shots: List[ShotEvent],
    ) -> RhythmControlMetrics:
        """Analyze rhythm patterns"""
        if not shots:
            return RhythmControlMetrics()

        metrics = RhythmControlMetrics()

        # Inter-shot intervals
        if len(shots) > 1:
            intervals = [
                shots[i].timestamp - shots[i-1].timestamp
                for i in range(1, len(shots))
            ]
            metrics.inter_shot_intervals = intervals
            metrics.avg_rally_tempo = np.mean(intervals) if intervals else 0.0
            metrics.rhythm_variance = np.std(intervals) if len(intervals) > 1 else 0.0

        # Swing consistency (simplified)
        metrics.swing_consistency_score = 0.7  # Placeholder

        # Rally patterns
        rallies = self._split_into_rallies(shots)
        if rallies:
            rally_lengths = [len(r) for r in rallies]
            metrics.avg_rally_length = np.mean(rally_lengths) if rally_lengths else 0.0

        # Attack frequency
        attack_shots = [s for s in shots if s.shot_type in [ShotType.SMASH, ShotType.DRIVE]]
        metrics.attack_frequency = len(attack_shots) / len(shots) if shots else 0.0

        return metrics

    def _split_into_rallies(
        self,
        shots: List[ShotEvent],
        gap_threshold: float = 3.0,
    ) -> List[List[ShotEvent]]:
        """Split shots into rallies based on time gaps"""
        if not shots:
            return []

        rallies = []
        current_rally = [shots[0]]

        for shot in shots[1:]:
            if shot.timestamp - current_rally[-1].timestamp > gap_threshold:
                rallies.append(current_rally)
                current_rally = [shot]
            else:
                current_rally.append(shot)

        if current_rally:
            rallies.append(current_rally)

        return rallies
