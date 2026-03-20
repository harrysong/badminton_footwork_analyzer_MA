"""
Center of Mass (CoM) Calculator for badminton player analysis
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import BODY_WEIGHTS, LANDMARKS, COM_LANDMARK_GROUPS
from core.pose_tracker import Pose3D


@dataclass
class CenterOfMass:
    """Center of Mass data"""
    x: float
    y: float
    z: float
    confidence: float
    segment_contributions: Dict[str, Tuple[float, float, float]]

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def to_2d(self) -> Tuple[float, float]:
        return (self.x, self.y)


class CoMCalculator:
    """Calculate Center of Mass based on anthropometric data"""

    def __init__(self, use_segment_weights: bool = True):
        self.use_segment_weights = use_segment_weights

        # Segment weights (fraction of total body mass)
        # Based on Winter, D.A. (2009) Biomechanics and Motor Control of Human Movement
        self.segment_weights = {
            "head": 0.081,
            "neck": 0.028,
            "torso_upper": 0.166,
            "torso_mid": 0.116,
            "torso_lower": 0.146,
            "left_upper_arm": 0.028,
            "left_forearm": 0.016,
            "left_hand": 0.006,
            "right_upper_arm": 0.028,
            "right_forearm": 0.016,
            "right_hand": 0.006,
            "left_thigh": 0.100,
            "left_shin": 0.047,
            "left_foot": 0.014,
            "right_thigh": 0.100,
            "right_shin": 0.047,
            "right_foot": 0.014,
        }

        # Center of mass position within each segment (as fraction from proximal end)
        # From proximal joint to distal joint
        self.segment_com_ratio = {
            "head": 0.5,  # Center of head
            "neck": 0.5,
            "torso_upper": 0.5,
            "torso_mid": 0.5,
            "torso_lower": 0.5,
            "left_upper_arm": 0.436,  # 43.6% from shoulder
            "left_forearm": 0.430,    # 43.0% from elbow
            "left_hand": 0.506,       # 50.6% from wrist
            "right_upper_arm": 0.436,
            "right_forearm": 0.430,
            "right_hand": 0.506,
            "left_thigh": 0.433,      # 43.3% from hip
            "left_shin": 0.433,       # 43.3% from knee
            "left_foot": 0.500,       # 50.0% from ankle
            "right_thigh": 0.433,
            "right_shin": 0.433,
            "right_foot": 0.500,
        }

        # History for smoothing
        self._history: List[CenterOfMass] = []
        self._max_history = 5

    def _get_segment_position(
        self,
        pose: Pose3D,
        segment_name: str,
    ) -> Optional[Tuple[float, float, float]]:
        """Calculate CoM position for a body segment"""
        landmarks_needed = COM_LANDMARK_GROUPS.get(segment_name, [])

        if not landmarks_needed:
            return None

        # Get landmark positions
        positions = []
        visibilities = []

        for lm_name in landmarks_needed:
            pos = pose.get_landmark(lm_name)
            if pos is not None:
                positions.append(pos)
                visibilities.append(pose.visibility[LANDMARKS[lm_name]])

        if not positions:
            return None

        # Special handling for different segment types
        if segment_name == "head":
            # Head uses just nose position
            return positions[0]

        elif "torso" in segment_name:
            # Torso segments use center of their landmarks
            return tuple(np.mean(positions, axis=0))

        elif len(positions) >= 2:
            # Limb segments - interpolate between joints
            com_ratio = self.segment_com_ratio.get(segment_name, 0.5)

            proximal = np.array(positions[0])  # Closer to center
            distal = np.array(positions[1])    # Further from center

            # Calculate CoM position
            com_pos = proximal + com_ratio * (distal - proximal)

            return tuple(com_pos)

        elif positions:
            return positions[0]

        return None

    def calculate_com(self, pose: Pose3D) -> Optional[CenterOfMass]:
        """
        Calculate overall Center of Mass from pose data

        Returns:
            CenterOfMass object or None if calculation fails
        """
        if pose is None:
            return None

        weighted_positions = []
        weights = []
        segment_contributions = {}

        for segment_name, weight in self.segment_weights.items():
            pos = self._get_segment_position(pose, segment_name)

            if pos is not None:
                weighted_positions.append(np.array(pos) * weight)
                weights.append(weight)
                segment_contributions[segment_name] = pos

        if not weighted_positions:
            return None

        # Calculate weighted average
        total_weight = sum(weights)
        com_position = sum(weighted_positions) / total_weight

        # Calculate confidence based on visibility of key landmarks
        key_landmarks = ["nose", "left_hip", "right_hip", "left_shoulder", "right_shoulder"]
        visibility_scores = [
            pose.visibility[LANDMARKS[lm]]
            for lm in key_landmarks
            if lm in LANDMARKS
        ]
        confidence = np.mean(visibility_scores) if visibility_scores else 0.0

        com = CenterOfMass(
            x=com_position[0],
            y=com_position[1],
            z=com_position[2],
            confidence=confidence,
            segment_contributions=segment_contributions,
        )

        # Add to history for smoothing
        self._history.append(com)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return self._smooth_com(com)

    def _smooth_com(self, com: CenterOfMass) -> CenterOfMass:
        """Apply temporal smoothing to CoM"""
        if len(self._history) < 2:
            return com

        # Simple moving average
        x_vals = [c.x for c in self._history]
        y_vals = [c.y for c in self._history]
        z_vals = [c.z for c in self._history]

        return CenterOfMass(
            x=np.mean(x_vals),
            y=np.mean(y_vals),
            z=np.mean(z_vals),
            confidence=com.confidence,
            segment_contributions=com.segment_contributions,
        )

    def calculate_com_velocity(
        self,
        com_history: List[CenterOfMass],
        fps: float = 30.0,
    ) -> Tuple[float, float, float]:
        """Calculate CoM velocity from history"""
        if len(com_history) < 2:
            return (0.0, 0.0, 0.0)

        dt = 1.0 / fps

        # Calculate velocity from last two points
        prev = com_history[-2]
        curr = com_history[-1]

        vx = (curr.x - prev.x) / dt
        vy = (curr.y - prev.y) / dt
        vz = (curr.z - prev.z) / dt

        return (vx, vy, vz)

    def calculate_com_acceleration(
        self,
        com_history: List[CenterOfMass],
        fps: float = 30.0,
    ) -> Tuple[float, float, float]:
        """Calculate CoM acceleration from history"""
        if len(com_history) < 3:
            return (0.0, 0.0, 0.0)

        dt = 1.0 / fps

        # Calculate velocities
        v1 = self.calculate_com_velocity(com_history[:-1], fps)
        v2 = self.calculate_com_velocity(com_history, fps)

        ax = (v2[0] - v1[0]) / dt
        ay = (v2[1] - v1[1]) / dt
        az = (v2[2] - v1[2]) / dt

        return (ax, ay, az)

    def get_com_height_variation(
        self,
        com_history: List[CenterOfMass],
        window_size: int = 30,
    ) -> Dict[str, float]:
        """Analyze CoM height variation (for detecting jumps/crouches)"""
        if len(com_history) < window_size:
            window_size = len(com_history)

        recent_y = [c.y for c in com_history[-window_size:]]

        return {
            "mean": np.mean(recent_y),
            "std": np.std(recent_y),
            "min": np.min(recent_y),
            "max": np.max(recent_y),
            "range": np.max(recent_y) - np.min(recent_y),
        }

    def clear_history(self) -> None:
        """Clear CoM history"""
        self._history.clear()

    def get_history(self) -> List[CenterOfMass]:
        """Get CoM history"""
        return list(self._history)


class SimpleCoMCalculator(CoMCalculator):
    """Simplified CoM calculator using key landmarks only"""

    def __init__(self):
        super().__init__(use_segment_weights=False)

        # Simplified weights for just 5 key landmarks
        self.key_landmarks = {
            "nose": 0.081,           # Head
            "left_shoulder": 0.108,  # Upper body (shared)
            "right_shoulder": 0.108,
            "left_hip": 0.196,       # Lower body (shared)
            "right_hip": 0.196,
            "left_ankle": 0.155,     # Legs (shared)
            "right_ankle": 0.155,
        }

    def calculate_com(self, pose: Pose3D) -> Optional[CenterOfMass]:
        """Calculate simplified CoM from key landmarks"""
        if pose is None:
            return None

        weighted_positions = []
        weights = []

        for landmark_name, weight in self.key_landmarks.items():
            pos = pose.get_landmark(landmark_name)
            if pos is not None and pose.is_visible(landmark_name, threshold=0.5):
                weighted_positions.append(np.array(pos) * weight)
                weights.append(weight)

        if not weighted_positions:
            return None

        total_weight = sum(weights)
        com_position = sum(weighted_positions) / total_weight

        # Calculate confidence
        confidence = min(1.0, len(weights) / len(self.key_landmarks))

        return CenterOfMass(
            x=com_position[0],
            y=com_position[1],
            z=com_position[2],
            confidence=confidence,
            segment_contributions={},
        )
