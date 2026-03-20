"""
Trajectory plotting and animation for movement visualization
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import VIZ_CONFIG, LANDMARKS
from core.pose_tracker import Pose3D
from core.com_calculator import CenterOfMass


class TrajectoryPlotter:
    """Plot movement trajectories on video frames"""

    def __init__(
        self,
        max_trajectory_length: int = 30,
        trajectory_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ):
        self.max_trajectory_length = max_trajectory_length

        # Default colors
        self.colors = trajectory_colors or {
            "left_foot": (0, 255, 0),      # Green
            "right_foot": (255, 0, 0),     # Blue
            "center": (0, 0, 255),         # Red
            "com": (255, 255, 0),          # Cyan
            "skeleton": (0, 255, 0),       # Green
        }

        # Trajectory history
        self.trajectories: Dict[str, List[Tuple[float, float]]] = {
            "left_foot": [],
            "right_foot": [],
            "center": [],
            "com": [],
        }

    def update_trajectories(
        self,
        pose: Optional[Pose3D] = None,
        com: Optional[CenterOfMass] = None,
    ) -> None:
        """Update trajectory history with new data"""
        if pose is not None:
            # Left foot
            left = pose.get_landmark_2d("left_ankle")
            if left:
                self.trajectories["left_foot"].append(left)

            # Right foot
            right = pose.get_landmark_2d("right_ankle")
            if right:
                self.trajectories["right_foot"].append(right)

            # Center between feet
            if left and right:
                center = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
                self.trajectories["center"].append(center)

        if com is not None:
            self.trajectories["com"].append((com.x, com.y))

        # Trim to max length
        for key in self.trajectories:
            if len(self.trajectories[key]) > self.max_trajectory_length:
                self.trajectories[key] = self.trajectories[key][-self.max_trajectory_length:]

    def draw_on_frame(
        self,
        frame: np.ndarray,
        draw_com: bool = True,
        draw_feet: bool = True,
        draw_center: bool = True,
        fade_trajectory: bool = True,
    ) -> np.ndarray:
        """Draw trajectories on frame"""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw CoM trajectory
        if draw_com and self.trajectories["com"]:
            annotated = self._draw_trajectory(
                annotated,
                self.trajectories["com"],
                self.colors["com"],
                fade=fade_trajectory,
            )

            # Draw current CoM position
            if self.trajectories["com"]:
                com_pos = self.trajectories["com"][-1]
                x, y = int(com_pos[0] * w), int(com_pos[1] * h)
                cv2.circle(annotated, (x, y), 8, self.colors["com"], -1)
                cv2.circle(annotated, (x, y), 10, (255, 255, 255), 2)

        # Draw feet trajectories
        if draw_feet:
            annotated = self._draw_trajectory(
                annotated,
                self.trajectories["left_foot"],
                self.colors["left_foot"],
                fade=fade_trajectory,
            )
            annotated = self._draw_trajectory(
                annotated,
                self.trajectories["right_foot"],
                self.colors["right_foot"],
                fade=fade_trajectory,
            )

        # Draw center trajectory
        if draw_center:
            annotated = self._draw_trajectory(
                annotated,
                self.trajectories["center"],
                self.colors["center"],
                fade=fade_trajectory,
                thickness=3,
            )

        return annotated

    def _draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: List[Tuple[float, float]],
        color: Tuple[int, int, int],
        fade: bool = True,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw a single trajectory"""
        if len(trajectory) < 2:
            return frame

        h, w = frame.shape[:2]

        for i in range(1, len(trajectory)):
            # Calculate fade
            if fade:
                alpha = i / len(trajectory)
                faded_color = tuple(int(c * alpha + 255 * (1 - alpha) * 0.3) for c in color)
                line_thickness = max(1, int(thickness * alpha))
            else:
                faded_color = color
                line_thickness = thickness

            # Convert normalized coordinates to pixel coordinates
            p1 = (
                int(trajectory[i - 1][0] * w),
                int(trajectory[i - 1][1] * h),
            )
            p2 = (
                int(trajectory[i][0] * w),
                int(trajectory[i][1] * h),
            )

            cv2.line(frame, p1, p2, faded_color, line_thickness)

        return frame

    def draw_static_trajectory(
        self,
        trajectory: List[Tuple[float, float]],
        size: Tuple[int, int] = (600, 600),
        background: Optional[np.ndarray] = None,
        color: Tuple[int, int, int] = (255, 0, 0),
        show_points: bool = False,
        show_direction: bool = True,
    ) -> np.ndarray:
        """Draw static trajectory on white/court background"""
        if background is None:
            canvas = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        else:
            canvas = background.copy()
            if canvas.shape[:2] != (size[1], size[0]):
                canvas = cv2.resize(canvas, size)

        if len(trajectory) < 2:
            return canvas

        # Scale trajectory to canvas size
        w, h = size
        points = [(int(x * w), int(y * h)) for x, y in trajectory]

        # Draw trajectory with gradient
        for i in range(1, len(points)):
            # Color gradient from blue to red
            ratio = i / len(points)
            r = int(color[0] * ratio + 50 * (1 - ratio))
            g = int(color[1] * ratio + 100 * (1 - ratio))
            b = int(color[2] * ratio + 200 * (1 - ratio))
            line_color = (b, g, r)

            cv2.line(canvas, points[i - 1], points[i], line_color, 3)

        # Draw points
        if show_points:
            for i, point in enumerate(points):
                if i % 5 == 0:  # Draw every 5th point
                    cv2.circle(canvas, point, 3, (0, 0, 0), -1)

        # Draw direction arrow
        if show_direction and len(points) >= 2:
            start = points[-5] if len(points) >= 5 else points[0]
            end = points[-1]
            cv2.arrowedLine(canvas, start, end, (0, 0, 255), 3, tipLength=0.3)

        # Mark start and end
        cv2.circle(canvas, points[0], 8, (0, 255, 0), -1)  # Green start
        cv2.circle(canvas, points[-1], 8, (0, 0, 255), -1)  # Red end

        # Labels
        cv2.putText(canvas, "Start", (points[0][0] + 10, points[0][1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(canvas, "End", (points[-1][0] + 10, points[-1][1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return canvas

    def create_comparison_plot(
        self,
        player_trajectory: List[Tuple[float, float]],
        reference_trajectory: List[Tuple[float, float]],
        size: Tuple[int, int] = (800, 600),
    ) -> np.ndarray:
        """Create side-by-side trajectory comparison"""
        canvas = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255

        # Split canvas
        mid_x = size[0] // 2

        # Draw dividing line
        cv2.line(canvas, (mid_x, 0), (mid_x, size[1]), (200, 200, 200), 2)

        # Left side - Player
        left_canvas = self.draw_static_trajectory(
            player_trajectory,
            size=(mid_x - 20, size[1] - 40),
            color=(255, 100, 100),
        )
        canvas[20:size[1]-20, 10:mid_x-10] = left_canvas

        # Right side - Reference
        right_canvas = self.draw_static_trajectory(
            reference_trajectory,
            size=(mid_x - 20, size[1] - 40),
            color=(100, 100, 255),
        )
        canvas[20:size[1]-20, mid_x+10:size[0]-10] = right_canvas

        # Labels
        cv2.putText(canvas, "Player", (mid_x // 2 - 40, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(canvas, "Reference", (mid_x + mid_x // 2 - 50, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        return canvas

    def reset(self) -> None:
        """Clear all trajectories"""
        for key in self.trajectories:
            self.trajectories[key].clear()


class PoseVisualizer:
    """Visualize pose skeleton"""

    def __init__(self):
        self.skeleton_connections = [
            ("nose", "left_shoulder"),
            ("nose", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("right_shoulder", "right_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("right_hip", "right_knee"),
            ("left_knee", "left_ankle"),
            ("right_knee", "right_ankle"),
        ]

    def draw_pose(
        self,
        frame: np.ndarray,
        pose: Pose3D,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        draw_keypoints: bool = True,
    ) -> np.ndarray:
        """Draw pose skeleton on frame"""
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # Draw connections
        for start_name, end_name in self.skeleton_connections:
            start_pos = pose.get_landmark_2d(start_name)
            end_pos = pose.get_landmark_2d(end_name)

            if start_pos and end_pos:
                # Check visibility
                start_vis = pose.is_visible(start_name)
                end_vis = pose.is_visible(end_name)

                if start_vis and end_vis:
                    p1 = (int(start_pos[0] * w), int(start_pos[1] * h))
                    p2 = (int(end_pos[0] * w), int(end_pos[1] * h))
                    cv2.line(annotated, p1, p2, color, thickness)

        # Draw keypoints
        if draw_keypoints:
            key_points = [
                "nose", "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                "left_hip", "right_hip", "left_knee", "right_knee",
                "left_ankle", "right_ankle",
            ]

            for name in key_points:
                pos = pose.get_landmark_2d(name)
                if pos and pose.is_visible(name):
                    x, y = int(pos[0] * w), int(pos[1] * h)
                    cv2.circle(annotated, (x, y), 5, color, -1)
                    cv2.circle(annotated, (x, y), 7, (255, 255, 255), 1)

        # Draw CoM if available
        # (This would need CoM passed separately)

        return annotated

    def draw_court_overlay(
        self,
        frame: np.ndarray,
        court_bounds: List[Tuple[float, float]],
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw court boundary overlay"""
        if len(court_bounds) != 4:
            return frame

        h, w = frame.shape[:2]
        annotated = frame.copy()

        # Convert normalized coordinates to pixel coordinates
        points = [(int(x * w), int(y * h)) for x, y in court_bounds]

        # Draw court rectangle
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            cv2.line(annotated, p1, p2, color, thickness)

        # Draw center line
        mid_top = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
        mid_bottom = ((points[2][0] + points[3][0]) // 2, (points[2][1] + points[3][1]) // 2)
        cv2.line(annotated, mid_top, mid_bottom, color, thickness)

        return annotated
