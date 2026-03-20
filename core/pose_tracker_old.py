"""
Pose tracking using MediaPipe
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))
from config import MEDIAPIPE_POSE_CONFIG, LANDMARKS
from utils.data_processing import TrajectorySmoother, KalmanFilter2D


@dataclass
class Pose3D:
    """3D pose data for a single frame"""
    landmarks: np.ndarray  # Shape: (33, 3) - x, y, z
    visibility: np.ndarray  # Shape: (33,) - visibility score
    world_landmarks: Optional[np.ndarray] = None  # Shape: (33, 3) - real-world coords

    def get_landmark(self, name: str) -> Optional[Tuple[float, float, float]]:
        """Get specific landmark by name"""
        if name in LANDMARKS:
            idx = LANDMARKS[name]
            if idx < len(self.landmarks):
                return tuple(self.landmarks[idx])
        return None

    def get_landmark_2d(self, name: str) -> Optional[Tuple[float, float]]:
        """Get 2D landmark (x, y)"""
        lm = self.get_landmark(name)
        return (lm[0], lm[1]) if lm else None

    def is_visible(self, name: str, threshold: float = 0.5) -> bool:
        """Check if landmark is visible"""
        if name in LANDMARKS:
            idx = LANDMARKS[name]
            if idx < len(self.visibility):
                return self.visibility[idx] > threshold
        return False


class PoseTracker:
    """MediaPipe pose tracker with smoothing and automatic fallback"""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_smoothing: bool = True,
        smoothing_method: str = "kalman",
        min_detection_confidence: float = 0.3,  # 降低阈值以提高检测率
        min_tracking_confidence: float = 0.3,     # 降低阈值以提高检测率
    ):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_smoothing = enable_smoothing
        self.smoothing_method = smoothing_method
        self._initialize_count = 0  # Track initialization attempts

        # Smoothing filters for each landmark
        self._kalman_filters: Dict[int, KalmanFilter2D] = {}
        self._trajectory_smoothers: Dict[int, TrajectorySmoother] = {}

        # Tracking state
        self.is_tracking = False
        self.last_pose: Optional[Pose3D] = None
        self.frame_count = 0

        # Detection statistics
        self.total_frames = 0
        self.detected_frames = 0
        self.missing_frames = 0
        self.last_detection_confidence = 0.0

        # Initialize MediaPipe Pose with fallback
        self.mp_pose = mp.solutions.pose
        self.pose = self._initialize_pose(
            static_image_mode,
            model_complexity,
            smooth_landmarks,
            min_detection_confidence,
            min_tracking_confidence,
        )

        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def _initialize_pose(
        self,
        static_image_mode: bool,
        model_complexity: int,
        smooth_landmarks: bool,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> Any:
        """Initialize MediaPipe Pose with automatic fallback for complex models"""

        import logging
        logger = logging.getLogger(__name__)

        # Try requested complexity
        try:
            pose = self.mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

            # Test if it actually works by processing a dummy frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            pose.process(test_frame_rgb)

            self._initialize_count += 1
            if model_complexity == 2:
                logger.warning("⚠️ 模型复杂度 2 已启用，但可能在某些系统上不稳定。如果遇到问题，请尝试复杂度 1。")

            return pose

        except RuntimeError as e:
            if model_complexity == 2 and "InferenceCalculator" in str(e):
                logger.warning(f"⚠️ 模型复杂度 {model_complexity} 不支持，自动降级到复杂度 1")

                # Fallback to complexity 1
                return self._initialize_pose(
                    static_image_mode=static_image_mode,
                    model_complexity=1,
                    smooth_landmarks=smooth_landmarks,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
            else:
                raise
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Pose: {e}")
            raise

    def _init_filters(self, num_landmarks: int = 33) -> None:
        """Initialize smoothing filters"""
        if self.smoothing_method == "kalman":
            for i in range(num_landmarks):
                self._kalman_filters[i] = KalmanFilter2D(
                    process_noise=1e-5,
                    measurement_noise=1e-2,
                )
        else:
            for i in range(num_landmarks):
                self._trajectory_smoothers[i] = TrajectorySmoother(
                    method=self.smoothing_method,
                    window_size=7,
                )

    def _smooth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply smoothing to landmarks"""
        if not self.enable_smoothing:
            return landmarks

        smoothed = landmarks.copy()

        if self.smoothing_method == "kalman":
            for i in range(len(landmarks)):
                if i not in self._kalman_filters:
                    self._kalman_filters[i] = KalmanFilter2D()

                pos = (landmarks[i, 0], landmarks[i, 1])
                smoothed_pos = self._kalman_filters[i].update(pos)
                smoothed[i, 0] = smoothed_pos[0]
                smoothed[i, 1] = smoothed_pos[1]

        return smoothed

    def process(self, frame: np.ndarray) -> Optional[Pose3D]:
        """
        Process a frame and return pose data

        Args:
            frame: BGR image from OpenCV

        Returns:
            Pose3D object or None if no pose detected
        """
        import logging
        logger = logging.getLogger(__name__)

        self.total_frames += 1

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            results = self.pose.process(rgb_frame)

            if not results.pose_landmarks:
                self.is_tracking = False
                self.missing_frames += 1
                return None

            self.is_tracking = True
            self.detected_frames += 1
            self.frame_count += 1

            # Extract landmarks
            landmarks = []
            visibility = []

            for lm in results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
                visibility.append(lm.visibility)

            # Track average confidence for debugging
            avg_visibility = sum(visibility) / len(visibility)
            self.last_detection_confidence = avg_visibility

            landmarks_array = np.array(landmarks)
            visibility_array = np.array(visibility)

            # Apply smoothing
            if self.enable_smoothing and self.frame_count > 1:
                landmarks_array = self._smooth_landmarks(landmarks_array)

            # Extract world landmarks if available
            world_landmarks = None
            if results.pose_world_landmarks:
                world_landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark]
                )

            self.last_pose = Pose3D(
                landmarks=landmarks_array,
                visibility=visibility_array,
                world_landmarks=world_landmarks,
            )

            return self.last_pose

        except RuntimeError as e:
            if "InferenceCalculator" in str(e):
                logger.error(f"MediaPipe 运行时错误: {e}")
                logger.info("建议：尝试降低模型复杂度到 1 或 0")
                return None
            raise
        except Exception as e:
            logger.warning(f"姿态检测异常: {e}")
            self.missing_frames += 1
            return None

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics for debugging"""
        if self.total_frames == 0:
            return {
                "detection_rate": 0.0,
                "total_frames": 0,
                "detected_frames": 0,
                "missing_frames": 0,
                "last_confidence": 0.0,
            }

        return {
            "detection_rate": self.detected_frames / self.total_frames,
            "total_frames": self.total_frames,
            "detected_frames": self.detected_frames,
            "missing_frames": self.missing_frames,
            "last_confidence": self.last_detection_confidence,
        }

    def draw_landmarks(
        self,
        frame: np.ndarray,
        pose: Optional[Pose3D] = None,
        draw_connections: bool = True,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        radius: int = 3,
    ) -> np.ndarray:
        """Draw pose landmarks on frame"""
        if pose is None:
            pose = self.last_pose

        if pose is None:
            return frame

        # Create a copy to draw on
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        landmarks_normalized = pose.landmarks.copy()
        landmarks_normalized[:, 0] *= w  # x
        landmarks_normalized[:, 1] *= h  # y

        # Draw connections
        if draw_connections:
            mp_landmarks = self._create_mp_landmarks(pose.landmarks)
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                mp_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        else:
            # Draw only key points
            for i, (x, y, z) in enumerate(landmarks_normalized):
                if pose.visibility[i] > 0.5:
                    cv2.circle(
                        annotated_frame,
                        (int(x), int(y)),
                        radius,
                        landmark_color,
                        -1,
                    )

        return annotated_frame

    def _create_mp_landmarks(self, landmarks: np.ndarray):
        """Create MediaPipe landmarks object from array"""
        mp_landmarks = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()

        for x, y, z in landmarks:
            lm = mp_landmarks.landmark.add()
            lm.x = x
            lm.y = y
            lm.z = z

        return mp_landmarks

    def get_body_bounding_box(
        self,
        pose: Optional[Pose3D] = None,
        padding: float = 0.1,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box of body (x, y, w, h)

        Returns:
            (x, y, width, height) in pixels or None
        """
        if pose is None:
            pose = self.last_pose

        if pose is None:
            return None

        visible_landmarks = [
            (int(lm[0] * 1000), int(lm[1] * 1000))  # Scale to avoid floating point issues
            for i, lm in enumerate(pose.landmarks)
            if pose.visibility[i] > 0.5
        ]

        if not visible_landmarks:
            return None

        xs = [p[0] for p in visible_landmarks]
        ys = [p[1] for p in visible_landmarks]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Apply padding
        width = x_max - x_min
        height = y_max - y_min

        x_min -= int(width * padding)
        x_max += int(width * padding)
        y_min -= int(height * padding)
        y_max += int(height * padding)

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def calculate_movement_score(self, pose: Optional[Pose3D] = None) -> float:
        """
        Calculate movement score based on limb positions
        Higher score = more active pose
        """
        if pose is None:
            pose = self.last_pose

        if pose is None:
            return 0.0

        # Calculate average distance of limbs from center
        center = pose.get_landmark("nose") or (0.5, 0.5, 0)
        if center is None:
            return 0.0

        limb_landmarks = [
            "left_wrist", "right_wrist",
            "left_ankle", "right_ankle",
            "left_knee", "right_knee",
        ]

        total_distance = 0.0
        count = 0

        for name in limb_landmarks:
            lm = pose.get_landmark(name)
            if lm and pose.is_visible(name):
                dist = np.sqrt(
                    (lm[0] - center[0]) ** 2 +
                    (lm[1] - center[1]) ** 2
                )
                total_distance += dist
                count += 1

        return total_distance / count if count > 0 else 0.0

    def reset(self) -> None:
        """Reset tracker state"""
        # Safely clear filters if they exist
        if hasattr(self, '_kalman_filters'):
            self._kalman_filters.clear()
        if hasattr(self, '_trajectory_smoothers'):
            self._trajectory_smoothers.clear()

        self.is_tracking = False
        self.last_pose = None
        self.frame_count = 0

        # Reset detection statistics
        self.total_frames = 0
        self.detected_frames = 0
        self.missing_frames = 0
        self.last_detection_confidence = 0.0

    def release(self) -> None:
        """Release resources"""
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class RealtimePoseTracker(PoseTracker):
    """Pose tracker optimized for real-time video"""

    def __init__(
        self,
        target_fps: int = 30,
        skip_frames: int = 0,
        **kwargs
    ):
        super().__init__(
            static_image_mode=False,  # Must be False for video
            smooth_landmarks=True,
            **kwargs
        )
        self.target_fps = target_fps
        self.skip_frames = skip_frames
        self._frame_counter = 0

    def process(self, frame: np.ndarray) -> Optional[Pose3D]:
        """Process frame with optional skipping"""
        self._frame_counter += 1

        # Skip frames for performance
        if self.skip_frames > 0 and self._frame_counter % (self.skip_frames + 1) != 0:
            return self.last_pose

        return super().process(frame)


class BatchPoseTracker(PoseTracker):
    """Pose tracker for batch processing video files"""

    def __init__(self, **kwargs):
        super().__init__(
            static_image_mode=False,
            smooth_landmarks=True,
            **kwargs
        )
        self.all_poses: List[Pose3D] = []

    def process(self, frame: np.ndarray) -> Optional[Pose3D]:
        """Process frame and store result"""
        pose = super().process(frame)
        self.all_poses.append(pose)
        return pose

    def get_trajectory(self, landmark_name: str) -> List[Optional[Tuple[float, float]]]:
        """Get trajectory of a specific landmark"""
        return [
            pose.get_landmark_2d(landmark_name) if pose else None
            for pose in self.all_poses
        ]

    def get_visibility_history(self, landmark_name: str) -> List[float]:
        """Get visibility history of a specific landmark"""
        result = []
        for pose in self.all_poses:
            if pose and landmark_name in LANDMARKS:
                idx = LANDMARKS[landmark_name]
                result.append(pose.visibility[idx])
            else:
                result.append(0.0)
        return result

    def clear_history(self) -> None:
        """Clear stored poses"""
        self.all_poses.clear()
