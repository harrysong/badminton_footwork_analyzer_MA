"""
Pose tracking using MediaPipe Tasks API (new API for mediapipe >= 0.10)
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import sys
import urllib.request
import os

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))
from config import MEDIAPIPE_POSE_CONFIG, LANDMARKS
from utils.data_processing import TrajectorySmoother, KalmanFilter2D


# Model file path
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"


def download_model():
    """Download pose landmarker model if not exists"""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"Downloading pose landmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    return MODEL_PATH


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
    """MediaPipe pose tracker using new Tasks API"""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_smoothing: bool = True,
        smoothing_method: str = "kalman",
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
    ):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_smoothing = enable_smoothing
        self.smoothing_method = smoothing_method

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

        # Download model if needed
        model_path = download_model()

        # Initialize MediaPipe PoseLandmarker
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        running_mode = VisionRunningMode.IMAGE if static_image_mode else VisionRunningMode.VIDEO

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = PoseLandmarker.create_from_options(options)
        self.running_mode = running_mode
        self._timestamp_ms = 0

        # For drawing - define pose connections manually based on MediaPipe pose topology
        # Format: (start_landmark_index, end_landmark_index)
        self.POSE_CONNECTIONS = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),  # Right eye
            (0, 4), (4, 5), (5, 6), (6, 8),  # Left eye
            (9, 10),  # Mouth
            # Torso
            (11, 12),  # Shoulders
            (11, 23), (12, 24),  # Shoulders to hips
            (23, 24),  # Hips
            # Right arm
            (11, 13), (13, 15),  # Right arm
            (15, 17), (15, 19), (15, 21),  # Right hand
            (17, 19),  # Right hand connection
            # Left arm
            (12, 14), (14, 16),  # Left arm
            (16, 18), (16, 20), (16, 22),  # Left hand
            (18, 20),  # Left hand connection
            # Right leg
            (23, 25), (25, 27),  # Right leg
            (27, 29), (27, 31),  # Right foot
            (29, 31),  # Right foot connection
            # Left leg
            (24, 26), (26, 28),  # Left leg
            (28, 30), (28, 32),  # Left foot
            (30, 32),  # Left foot connection
        ]

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

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Process frame based on running mode
            if self.running_mode == mp.tasks.vision.RunningMode.VIDEO:
                self._timestamp_ms += 33  # ~30fps
                results = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
            else:
                results = self.landmarker.detect(mp_image)

            if not results.pose_landmarks or len(results.pose_landmarks) == 0:
                self.is_tracking = False
                self.missing_frames += 1
                return None

            self.is_tracking = True
            self.detected_frames += 1
            self.frame_count += 1

            # Get first detected pose
            pose_landmarks = results.pose_landmarks[0]

            # Extract landmarks
            landmarks = []
            visibility = []

            for lm in pose_landmarks:
                landmarks.append([lm.x, lm.y, lm.z])
                # Use presence as visibility proxy if visibility not available
                visibility.append(getattr(lm, 'visibility', getattr(lm, 'presence', 1.0)))

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
            if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
                world_landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks[0]]
                )

            self.last_pose = Pose3D(
                landmarks=landmarks_array,
                visibility=visibility_array,
                world_landmarks=world_landmarks,
            )

            return self.last_pose

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
            # Draw skeleton connections
            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection

                if pose.visibility[start_idx] > 0.5 and pose.visibility[end_idx] > 0.5:
                    start_point = (int(landmarks_normalized[start_idx, 0]),
                                   int(landmarks_normalized[start_idx, 1]))
                    end_point = (int(landmarks_normalized[end_idx, 0]),
                                 int(landmarks_normalized[end_idx, 1]))

                    cv2.line(annotated_frame, start_point, end_point,
                             connection_color, thickness)

        # Draw landmarks
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
            (int(lm[0] * 1000), int(lm[1] * 1000))
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
        """Calculate movement score based on limb positions"""
        if pose is None:
            pose = self.last_pose

        if pose is None:
            return 0.0

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
        if hasattr(self, '_kalman_filters'):
            self._kalman_filters.clear()
        if hasattr(self, '_trajectory_smoothers'):
            self._trajectory_smoothers.clear()

        self.is_tracking = False
        self.last_pose = None
        self.frame_count = 0
        self._timestamp_ms = 0

        # Reset detection statistics
        self.total_frames = 0
        self.detected_frames = 0
        self.missing_frames = 0
        self.last_detection_confidence = 0.0

    def release(self) -> None:
        """Release resources"""
        self.landmarker.close()

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
            static_image_mode=False,
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
