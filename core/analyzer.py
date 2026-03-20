"""
Main analyzer that orchestrates all components
"""

import numpy as np
import cv2
from typing import Optional, Dict, List, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
import time
import sys
import logging

# Setup logger
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent.parent))

from config import VIDEO_CONFIG, VIZ_CONFIG, OUTPUT_DIR
from core.pose_tracker import PoseTracker, RealtimePoseTracker, BatchPoseTracker, Pose3D
from core.com_calculator import CoMCalculator, SimpleCoMCalculator, CenterOfMass
from core.footwork_analyzer import FootworkAnalyzer, FootworkMetrics, FootworkEventData
from core.efficiency_model import EfficiencyModel, EfficiencyScore, ComparisonResult
from core.shot_analyzer import (
    ShotDetector, BiomechanicsAnalyzer, TacticalGeometryAnalyzer,
    RhythmControlAnalyzer, BiomechanicsMetrics, TacticalGeometryMetrics,
    RhythmControlMetrics, ShotEvent
)
from utils.video_io import VideoReader, VideoWriter, draw_text
from visualization.heatmap_generator import HeatmapGenerator, TemporalHeatmap
from visualization.trajectory_plotter import TrajectoryPlotter, PoseVisualizer


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    metrics: FootworkMetrics
    efficiency_score: EfficiencyScore
    comparisons: Dict[str, ComparisonResult]
    recommendations: List[Dict[str, str]]
    trajectory: List[Tuple[float, float]]
    heatmap: Optional[np.ndarray] = None
    duration: float = 0.0
    frame_count: int = 0

    # Advanced metrics
    shots: List[ShotEvent] = field(default_factory=list)
    biomechanics: BiomechanicsMetrics = field(default_factory=BiomechanicsMetrics)
    tactical_geometry: TacticalGeometryMetrics = field(default_factory=TacticalGeometryMetrics)
    rhythm_control: RhythmControlMetrics = field(default_factory=RhythmControlMetrics)


class BadmintonAnalyzer:
    """
    Main analyzer for badminton footwork analysis
    """

    def __init__(
        self,
        model_complexity: int = 1,
        enable_smoothing: bool = True,
        fps: float = 30.0,
        pixel_scale: Optional[float] = None,
        reference_level: str = "professional",
    ):
        # Core components
        self.pose_tracker = PoseTracker(
            model_complexity=model_complexity,
            enable_smoothing=enable_smoothing,
        )
        self.com_calculator = CoMCalculator()
        self.footwork_analyzer = FootworkAnalyzer(fps=fps, pixel_scale=pixel_scale)
        self.efficiency_model = EfficiencyModel()

        # Advanced analysis components
        self.shot_detector = ShotDetector(fps=fps)
        self.biomechanics_analyzer = BiomechanicsAnalyzer(fps=fps)
        self.tactical_analyzer = TacticalGeometryAnalyzer()
        self.rhythm_analyzer = RhythmControlAnalyzer(fps=fps)

        # Visualization components
        self.heatmap_generator = HeatmapGenerator()
        self.trajectory_plotter = TrajectoryPlotter()
        self.pose_visualizer = PoseVisualizer()
        self.temporal_heatmap = TemporalHeatmap()

        # Shot tracking
        self.detected_shots: List[ShotEvent] = []

        # Configuration
        self.reference_level = reference_level
        self.fps = fps

        # State
        self.is_processing = False
        self.current_frame = 0
        self.start_time = 0.0

    def process_video(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        show_progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> AnalysisResult:
        """
        Process a video file and generate analysis

        Args:
            video_path: Path to input video
            output_path: Optional path for annotated output video
            show_progress: Whether to show progress
            progress_callback: Optional callback(frame_idx, total_frames)

        Returns:
            AnalysisResult with all metrics
        """
        video_path = Path(video_path)

        # Reset state
        self.reset()
        self.is_processing = True
        self.start_time = time.time()

        # Open video
        with VideoReader(video_path) as reader:
            video_info = reader.info
            total_frames = video_info.frame_count

            # Setup writer if output path specified
            writer = None
            if output_path:
                output_path = Path(output_path)
                logger.info(f"Creating video writer: {output_path}, fps={video_info.fps}, resolution={video_info.width}x{video_info.height}")
                writer = VideoWriter(
                    output_path,
                    fps=video_info.fps,
                    resolution=(video_info.width, video_info.height),
                )

            # Process frames
            for frame_idx, frame in enumerate(reader):
                self.current_frame = frame_idx

                # Process frame
                annotated_frame = self._process_frame(frame, frame_idx)

                # Write output
                if writer:
                    writer.write(annotated_frame)

                # Progress callback
                if progress_callback:
                    progress_callback(frame_idx, total_frames)

            if writer:
                writer.release()

        self.is_processing = False
        duration = time.time() - self.start_time

        # Generate final analysis
        return self._generate_result(duration)

    def _process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Process a single frame"""
        # Detect pose
        pose = self.pose_tracker.process(frame)

        if pose is None:
            return frame

        # Calculate CoM
        com = self.com_calculator.calculate_com(pose)

        # Analyze footwork
        events = self.footwork_analyzer.process_frame(pose, com, frame_number)

        # Detect shots
        shot = self.shot_detector.process_frame(pose, com, frame_number)
        if shot:
            self.detected_shots.append(shot)

        # Update visualizations
        self.trajectory_plotter.update_trajectories(pose, com)

        if com:
            self.temporal_heatmap.update((com.x, com.y))

        # Create annotated frame
        annotated = self._annotate_frame(frame, pose, com, events, shot if shot else None)

        return annotated

    def _annotate_frame(
        self,
        frame: np.ndarray,
        pose: Pose3D,
        com: Optional[CenterOfMass],
        events: List[FootworkEventData],
        shot: Optional = None,
    ) -> np.ndarray:
        """Create annotated visualization frame"""
        # Draw pose
        annotated = self.pose_visualizer.draw_pose(frame, pose)

        # Draw trajectories
        annotated = self.trajectory_plotter.draw_on_frame(annotated)

        # Draw CoM
        if com:
            h, w = frame.shape[:2]
            cx, cy = int(com.x * w), int(com.y * h)
            cv2.circle(annotated, (cx, cy), 8, VIZ_CONFIG["com_color"], -1)
            cv2.circle(annotated, (cx, cy), 10, (255, 255, 255), 2)

        # Draw events
        for event in events:
            h, w = frame.shape[:2]
            x, y = int(event.position[0] * w), int(event.position[1] * h)

            if event.event_type.value == "takeoff":
                color = (0, 165, 255)  # Orange
                text = "JUMP"
            elif event.event_type.value == "landing":
                color = (0, 255, 0)  # Green
                text = "LAND"
            elif event.event_type.value == "direction_change":
                color = (255, 0, 255)  # Magenta
                text = "TURN"
            else:
                color = (255, 255, 0)  # Cyan
                text = event.event_type.value.upper()

            cv2.circle(annotated, (x, y), 15, color, 2)
            cv2.putText(
                annotated, text, (x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # Draw shot indicator if detected
        if shot:
            h, w = frame.shape[:2]
            rx, ry = int(shot.racket_position[0] * w), int(shot.racket_position[1] * h)

            # Draw racket position
            cv2.circle(annotated, (rx, ry), 12, (255, 255, 0), 3)

            # Draw shot type label
            shot_label = shot.shot_type.value.upper()
            cv2.putText(
                annotated, f"HIT: {shot_label}", (rx + 20, ry),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )

        # Add info overlay
        annotated = self._add_info_overlay(annotated)

        return annotated

    def _add_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add information overlay to frame"""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent background
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Text info
        y_pos = 40
        cv2.putText(
            frame, f"Frame: {self.current_frame}", (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        if self.footwork_analyzer.foot_positions:
            metrics = self.footwork_analyzer.calculate_metrics(debug=False)
            y_pos += 25
            cv2.putText(
                frame, f"Steps: {metrics.total_steps}", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            y_pos += 25
            cv2.putText(
                frame, f"Distance: {metrics.total_distance:.1f}m", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        return frame

    def _generate_result(self, duration: float) -> AnalysisResult:
        """Generate final analysis result"""
        # Calculate footwork metrics
        metrics = self.footwork_analyzer.calculate_metrics(debug=True)

        # Calculate efficiency score
        efficiency_score = self.efficiency_model.calculate_efficiency_score(
            metrics, self.reference_level
        )

        # Compare with reference
        comparisons = self.efficiency_model.compare_with_reference(
            metrics, self.reference_level
        )

        # Generate recommendations
        recommendations = self.efficiency_model.generate_recommendations(
            efficiency_score, comparisons, metrics, self.reference_level
        )

        # Get trajectory
        trajectory = self.footwork_analyzer.get_foot_trajectory("center")

        # Generate heatmap
        heatmap = None
        if trajectory:
            heatmap = self.heatmap_generator.generate_heatmap(trajectory)

        # === Advanced Metrics Analysis ===

        # Analyze shots and biomechanics
        biomechanics = self.biomechanics_analyzer.analyze_shots(self.detected_shots)

        # Analyze tactical geometry
        tactical_geometry = self.tactical_analyzer.analyze_tactics(self.detected_shots)

        # Analyze rhythm control
        rhythm_control = self.rhythm_analyzer.analyze_rhythm(self.detected_shots)

        return AnalysisResult(
            metrics=metrics,
            efficiency_score=efficiency_score,
            comparisons=comparisons,
            recommendations=recommendations,
            trajectory=trajectory,
            heatmap=heatmap,
            duration=duration,
            frame_count=self.current_frame,
            shots=self.detected_shots,
            biomechanics=biomechanics,
            tactical_geometry=tactical_geometry,
            rhythm_control=rhythm_control,
        )

    def process_frame_realtime(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[AnalysisResult]]:
        """
        Process a frame in real-time mode

        Returns:
            (annotated_frame, partial_result or None)
        """
        annotated = self._process_frame(frame, self.current_frame)
        self.current_frame += 1

        # Generate partial result every 30 frames
        if self.current_frame % 30 == 0:
            partial_result = self._generate_result(0)
            return annotated, partial_result

        return annotated, None

    def reset(self) -> None:
        """Reset all analyzers"""
        self.pose_tracker.reset()
        self.com_calculator.clear_history()
        self.footwork_analyzer.reset()
        self.trajectory_plotter.reset()
        self.temporal_heatmap.reset()
        self.shot_detector.reset()
        self.detected_shots.clear()
        self.current_frame = 0
        self.is_processing = False

    def release(self) -> None:
        """Release resources"""
        self.pose_tracker.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class RealtimeAnalyzer(BadmintonAnalyzer):
    """Analyzer optimized for real-time processing"""

    def __init__(
        self,
        model_complexity: int = 0,  # Use lighter model for real-time
        skip_frames: int = 1,  # Process every other frame
        **kwargs
    ):
        super().__init__(model_complexity=model_complexity, **kwargs)
        self.skip_frames = skip_frames
        self.frame_counter = 0

    def process_frame_realtime(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[AnalysisResult]]:
        """Process frame with skipping for performance"""
        self.frame_counter += 1

        # Skip frames for pose estimation
        if self.frame_counter % (self.skip_frames + 1) == 0:
            return super().process_frame_realtime(frame)

        # Just annotate with existing data
        # (In a real implementation, you'd use the last known pose)
        return frame, None
