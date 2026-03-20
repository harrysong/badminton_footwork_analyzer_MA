"""Detection Agent - wraps PoseTracker for LangGraph multi-agent system."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentMetrics
from core.pose_tracker import PoseTracker, Pose3D
from utils.video_io import VideoReader


class DetectionAgent(BaseAgent):
    """
    Agent responsible for pose detection and tracking.

    Wraps the existing PoseTracker to integrate with the LangGraph workflow.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, agent_name="DetectionAgent")
        self.pose_tracker: Optional[PoseTracker] = None
        self.video_reader: Optional[VideoReader] = None

    def _initialize(self) -> None:
        """Initialize the pose tracker with configuration."""
        model_complexity = self.config.get("model_complexity", 1)
        enable_smoothing = self.config.get("enable_smoothing", True)
        min_detection_confidence = self.config.get("min_detection_confidence", 0.5)
        min_tracking_confidence = self.config.get("min_tracking_confidence", 0.5)

        self.pose_tracker = PoseTracker(
            model_complexity=model_complexity,
            enable_smoothing=enable_smoothing,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process video and detect poses.

        Args:
            state: Must contain either:
                - video_path: Path to video file
                - video_frames: Pre-loaded frames

        Returns:
            Updated state with:
                - poses: List of detected Pose3D objects
                - detection_stats: Detection quality statistics
                - video_frames: Frames (if loaded from video_path)
        """
        # Get frames from state
        video_path = state.get("video_path")
        video_frames = state.get("video_frames", [])

        if video_path and not video_frames:
            # Load frames from video
            self.video_reader = VideoReader(video_path)
            video_frames = self.video_reader.read_all_frames()
            state["video_frames"] = video_frames

        if not video_frames:
            raise ValueError("No video frames provided")

        # Set frame rate
        frame_rate = self.config.get("frame_rate", 30.0)
        if self.video_reader:
            frame_rate = self.video_reader.fps
        state["frame_rate"] = frame_rate

        # Detect poses in each frame
        poses: List[Optional[Pose3D]] = []
        detected_count = 0
        total_confidence = 0.0

        for frame in video_frames:
            pose = self.pose_tracker.process(frame)
            poses.append(pose)

            if pose is not None:
                detected_count += 1
                # Calculate average visibility as confidence
                avg_visibility = float(np.mean(pose.visibility))
                total_confidence += avg_visibility

        # Calculate detection statistics
        total_frames = len(video_frames)
        detection_rate = detected_count / total_frames if total_frames > 0 else 0.0
        avg_confidence = total_confidence / detected_count if detected_count > 0 else 0.0

        detection_stats = {
            "total_frames": total_frames,
            "detected_frames": detected_count,
            "detection_rate": detection_rate,
            "avg_confidence": avg_confidence,
            "missing_frames": total_frames - detected_count,
        }

        # Update state with results
        state["poses"] = poses
        state["detection_stats"] = detection_stats
        state["current_stage"] = "detecting"

        # Add custom metrics
        self.metrics.custom_metrics = {
            "detection_rate": detection_rate,
            "avg_confidence": avg_confidence,
            "frames_processed": total_frames,
        }

        # Warn if detection rate is low
        if detection_rate < 0.7:
            self.metrics.warnings.append(
                f"Low detection rate: {detection_rate:.2%}"
            )

        return state

    def process_frame(self, frame: np.ndarray) -> Optional[Pose3D]:
        """Process a single frame and return pose."""
        if self.pose_tracker is None:
            raise RuntimeError("Pose tracker not initialized")
        return self.pose_tracker.process(frame)

    def release(self) -> None:
        """Release resources."""
        if self.pose_tracker:
            self.pose_tracker.release()
        if self.video_reader:
            self.video_reader.release()

    def reset(self) -> None:
        """Reset the agent state."""
        super().reset()
        if self.pose_tracker:
            self.pose_tracker.reset()
