"""Visualization Agent - wraps visualization modules for LangGraph multi-agent system."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from core.pose_tracker import Pose3D
from core.com_calculator import CenterOfMass
from visualization.heatmap_generator import HeatmapGenerator
from visualization.trajectory_plotter import TrajectoryPlotter


class VisualizationAgent(BaseAgent):
    """
    Agent responsible for generating visualizations.

    Wraps existing visualization modules to:
    - Generate heatmaps
    - Generate trajectory plots
    - Create annotated frames
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, agent_name="VisualizationAgent")
        self.heatmap_generator: Optional[HeatmapGenerator] = None
        self.trajectory_plotter: Optional[TrajectoryPlotter] = None

    def _initialize(self) -> None:
        """Initialize visualization components."""
        self.heatmap_generator = HeatmapGenerator()
        self.trajectory_plotter = TrajectoryPlotter()

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualizations from analysis results.

        Args:
            state: Must contain:
                - poses: List of detected poses
                - com_positions: Center of mass positions
                - footwork_events: Footwork events
                - video_frames: Original video frames (optional)

        Returns:
            Updated state with:
                - heatmap_data: 2D heatmap array
                - trajectory_data: Trajectory points
                - visualizations: Dict of visualization outputs
                - annotated_frames: Frames with pose overlay
        """
        poses = state.get("poses", [])
        com_positions = state.get("com_positions", [])
        footwork_events = state.get("footwork_events", [])
        video_frames = state.get("video_frames", [])

        visualizations = {}

        # Generate heatmap
        if com_positions:
            heatmap_data = self._generate_heatmap(com_positions)
            state["heatmap_data"] = heatmap_data
            visualizations["heatmap"] = heatmap_data

        # Generate trajectory
        if com_positions:
            trajectory_data = self._generate_trajectory(com_positions, poses)
            state["trajectory_data"] = trajectory_data
            visualizations["trajectory"] = trajectory_data

        # Generate annotated frames (if video frames provided)
        if video_frames and poses:
            annotated_frames = self._generate_annotated_frames(
                video_frames, poses, footwork_events
            )
            state["annotated_frames"] = annotated_frames
            visualizations["annotated_video"] = annotated_frames

        # Generate court overlay
        if com_positions:
            court_overlay = self._generate_court_overlay(com_positions)
            visualizations["court_overlay"] = court_overlay

        state["visualizations"] = visualizations
        state["current_stage"] = "visualizing"

        # Add custom metrics
        self.metrics.custom_metrics = {
            "heatmap_generated": "heatmap_data" in state,
            "trajectory_points": len(state.get("trajectory_data", [])),
            "frames_annotated": len(state.get("annotated_frames", [])),
        }

        return state

    def _generate_heatmap(
        self, com_positions: List[CenterOfMass]
    ) -> np.ndarray:
        """Generate heatmap from center of mass positions."""
        # Filter out None values
        valid_coms = [c for c in com_positions if c is not None]

        if not valid_coms:
            return np.zeros((100, 100))

        # Extract 2D positions (x, y)
        positions = np.array([[c.x, c.y] for c in valid_coms])

        # Generate heatmap
        heatmap = self.heatmap_generator.generate(positions)

        return heatmap

    def _generate_trajectory(
        self,
        com_positions: List[CenterOfMass],
        poses: List[Optional[Pose3D]],
    ) -> List[tuple]:
        """Generate trajectory data for plotting."""
        trajectory = []

        # Use CoM positions if available
        valid_coms = [c for c in com_positions if c is not None]

        for com in valid_coms:
            trajectory.append((com.x, com.y))

        # Fallback to pose positions if no CoM
        if not trajectory and poses:
            for pose in poses:
                if pose is not None:
                    # Use hip center as fallback
                    hip_center = pose.get_landmark_2d("left_hip")
                    if hip_center:
                        trajectory.append(hip_center)

        return trajectory

    def _generate_annotated_frames(
        self,
        video_frames: List[np.ndarray],
        poses: List[Optional[Pose3D]],
        footwork_events: List[Any],
    ) -> List[np.ndarray]:
        """Generate frames with pose overlay."""
        annotated = []

        # Limit to key frames for performance
        max_frames = min(len(video_frames), len(poses), 100)

        step = max(1, max_frames // 50)  # Sample up to 50 frames

        for i in range(0, max_frames, step):
            if i < len(video_frames) and i < len(poses):
                frame = video_frames[i].copy()
                pose = poses[i]

                if pose is not None:
                    # Draw pose landmarks on frame
                    annotated_frame = self._draw_pose(frame, pose)
                    annotated.append(annotated_frame)

        return annotated

    def _draw_pose(
        self, frame: np.ndarray, pose: Pose3D
    ) -> np.ndarray:
        """Draw pose landmarks on frame."""
        import cv2

        h, w = frame.shape[:2]

        # Draw connections
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
        ]

        # Draw landmarks
        for landmark in pose.landmarks:
            x, y = int(landmark[0] * w), int(landmark[1] * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Draw connections
        for start, end in connections:
            if start < len(pose.landmarks) and end < len(pose.landmarks):
                x1 = int(pose.landmarks[start][0] * w)
                y1 = int(pose.landmarks[start][1] * h)
                x2 = int(pose.landmarks[end][0] * w)
                y2 = int(pose.landmarks[end][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame

    def _generate_court_overlay(
        self, com_positions: List[CenterOfMass]
    ) -> np.ndarray:
        """Generate court overlay with movement path."""
        # Create a blank court image
        court = np.ones((500, 300, 3), dtype=np.uint8) * 255

        # Draw court lines
        import cv2

        # Court boundaries
        cv2.rectangle(court, (10, 10), (290, 490), (0, 0, 0), 2)

        # Net line
        cv2.line(court, (10, 250), (290, 250), (0, 0, 255), 2)

        # Service lines
        cv2.rectangle(court, (60, 10), (240, 150), (0, 0, 0), 1)
        cv2.rectangle(court, (60, 350), (240, 490), (0, 0, 0), 1)

        # Plot trajectory if available
        if com_positions:
            valid_coms = [c for c in com_positions if c is not None]
            if valid_coms:
                # Normalize positions to court size
                xs = [c.x for c in valid_coms]
                ys = [c.y for c in valid_coms]

                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                # Scale to court coordinates
                scale_x = 270 / (max_x - min_x) if max_x != min_x else 1
                scale_y = 470 / (max_y - min_y) if max_y != min_y else 1

                points = []
                for com in valid_coms:
                    px = int(20 + (com.x - min_x) * scale_x)
                    py = int(20 + (com.y - min_y) * scale_y)
                    points.append((px, py))

                # Draw trajectory
                for i in range(1, len(points)):
                    cv2.line(
                        court, points[i-1], points[i],
                        (255, 0, 0), 2
                    )

        return court

    def save_visualizations(
        self,
        visualizations: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Save visualizations to files."""
        import cv2

        output_paths = {}

        output_dir.mkdir(parents=True, exist_ok=True)

        if "heatmap" in visualizations:
            path = output_dir / "heatmap.png"
            cv2.imwrite(str(path), visualizations["heatmap"])
            output_paths["heatmap"] = path

        if "court_overlay" in visualizations:
            path = output_dir / "court_overlay.png"
            cv2.imwrite(str(path), visualizations["court_overlay"])
            output_paths["court_overlay"] = path

        return output_paths
