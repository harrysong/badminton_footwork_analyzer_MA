"""Analysis Agent - wraps footwork and shot analysis for LangGraph multi-agent system."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from core.pose_tracker import Pose3D
from core.com_calculator import CoMCalculator, CenterOfMass
from core.footwork_analyzer import (
    FootworkAnalyzer,
    FootworkEvent,
    FootworkEventData,
    FootworkMetrics,
)
from core.shot_analyzer import ShotDetector, ShotEvent


class AnalysisAgent(BaseAgent):
    """
    Agent responsible for biomechanical analysis including:
    - Center of Mass calculation
    - Footwork event detection
    - Shot detection and classification
    - Metric calculation
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, agent_name="AnalysisAgent")
        self.com_calculator: Optional[CoMCalculator] = None
        self.footwork_analyzer: Optional[FootworkAnalyzer] = None
        self.shot_detector: Optional[ShotDetector] = None

    def _initialize(self) -> None:
        """Initialize analysis components with configuration."""
        fps = self.config.get("frame_rate", 30.0)
        pixel_scale = self.config.get("pixel_scale")

        self.com_calculator = CoMCalculator()
        self.footwork_analyzer = FootworkAnalyzer(fps=fps, pixel_scale=pixel_scale)
        self.shot_detector = ShotDetector(fps=fps)

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze poses and extract footwork/shot metrics.

        Args:
            state: Must contain:
                - poses: List of detected Pose3D objects
                - frame_rate: Video frame rate

        Returns:
            Updated state with:
                - com_positions: Center of mass positions
                - footwork_events: Detected footwork events
                - shot_events: Detected shot events
                - footwork_metrics: Calculated metrics
        """
        poses = state.get("poses", [])
        frame_rate = state.get("frame_rate", 30.0)
        fps = self.config.get("frame_rate", frame_rate)

        if not poses:
            raise ValueError("No poses provided for analysis")

        # Calculate Center of Mass for each frame
        com_positions: List[CenterOfMass] = []
        valid_pose_count = 0

        for i, pose in enumerate(poses):
            if pose is not None:
                com = self.com_calculator.calculate(pose)
                com_positions.append(com)
                valid_pose_count += 1
            else:
                # Add None for frames without detected pose
                com_positions.append(None)

        state["com_positions"] = com_positions

        # Update analyzer FPS in case it differs
        self.footwork_analyzer.fps = fps

        # Detect footwork events
        footwork_events: List[FootworkEventData] = []
        for i, (pose, com) in enumerate(zip(poses, com_positions)):
            if pose is not None:
                events = self.footwork_analyzer.process_frame(pose, com, i)
                footwork_events.extend(events)

        # Calculate footwork metrics
        if self.footwork_analyzer.foot_positions:
            footwork_metrics = self.footwork_analyzer.calculate_metrics()
            state["footwork_metrics"] = footwork_metrics
        else:
            footwork_metrics = FootworkMetrics()
            state["footwork_metrics"] = footwork_metrics

        state["footwork_events"] = footwork_events
        state["current_stage"] = "analyzing"

        # Detect shot events
        shot_events = self._detect_shots(poses, footwork_events)
        state["shot_events"] = shot_events

        # Calculate biomechanics metrics
        biomechanics_metrics = self._calculate_biomechanics(
            com_positions, footwork_events, fps
        )
        state["biomechanics_metrics"] = biomechanics_metrics

        # Add custom metrics
        self.metrics.custom_metrics = {
            "valid_poses": valid_pose_count,
            "total_frames": len(poses),
            "footwork_events_count": len(footwork_events),
            "shot_events_count": len(shot_events),
            "avg_com_confidence": self._calculate_com_confidence(com_positions),
        }

        return state

    def _detect_shots(
        self,
        poses: List[Optional[Pose3D]],
        footwork_events: List[FootworkEventData],
    ) -> List[ShotEvent]:
        """Detect shot events from poses and footwork events."""
        shot_events = []

        # Find landing events (likely shots)
        landings = [
            e for e in footwork_events
            if e.event_type == FootworkEvent.LANDING
        ]

        for landing in landings:
            # Use shot detector to classify the shot
            frame_idx = landing.frame_number
            if frame_idx < len(poses) and poses[frame_idx] is not None:
                shot = self.shot_detector.detect_shot(poses[frame_idx], frame_idx)
                if shot:
                    shot_events.append(shot)

        return shot_events

    def _calculate_biomechanics(
        self,
        com_positions: List[CenterOfMass],
        footwork_events: List[FootworkEventData],
        fps: float,
    ) -> Dict[str, Any]:
        """Calculate biomechanics metrics from CoM positions."""
        # Filter out None values
        valid_coms = [c for c in com_positions if c is not None]

        if len(valid_coms) < 2:
            return {
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "avg_acceleration": 0.0,
                "max_acceleration": 0.0,
            }

        # Calculate velocities
        velocities = []
        for i in range(1, len(valid_coms)):
            dt = 1.0 / fps
            dx = valid_coms[i].x - valid_coms[i-1].x
            dy = valid_coms[i].y - valid_coms[i-1].y
            velocity = np.sqrt(dx**2 + dy**2) / dt
            velocities.append(velocity)

        # Calculate accelerations
        accelerations = []
        for i in range(1, len(velocities)):
            dt = 1.0 / fps
            acc = (velocities[i] - velocities[i-1]) / dt
            accelerations.append(acc)

        return {
            "avg_speed": np.mean(velocities) if velocities else 0.0,
            "max_speed": np.max(velocities) if velocities else 0.0,
            "avg_acceleration": np.mean(accelerations) if accelerations else 0.0,
            "max_acceleration": np.max(accelerations) if accelerations else 0.0,
            "total_movement_distance": sum(velocities) / fps if velocities else 0.0,
        }

    def _calculate_com_confidence(self, com_positions: List[CenterOfMass]) -> float:
        """Calculate average confidence of CoM calculations."""
        valid_coms = [c for c in com_positions if c is not None]
        if not valid_coms:
            return 0.0
        # Use position variance as a proxy for confidence
        # Lower variance = higher confidence
        x_coords = [c.x for c in valid_coms]
        y_coords = [c.y for c in valid_coms]
        variance = np.var(x_coords) + np.var(y_coords)
        # Normalize to 0-1 (lower variance = higher confidence)
        confidence = 1.0 / (1.0 + variance / 1000)
        return float(confidence)

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        if self.footwork_analyzer:
            self.footwork_analyzer.foot_positions = []
            self.footwork_analyzer.com_positions = []
            self.footwork_analyzer.events = []
        if self.shot_detector:
            self.shot_detector.shot_events = []
