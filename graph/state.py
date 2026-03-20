"""State definitions for LangGraph multi-agent system."""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from typing_extensions import NotRequired


class ProcessingStage(Enum):
    """Processing stage enumeration for workflow tracking"""
    INITIALIZED = "initialized"
    DETECTING = "detecting"
    ANALYZING = "analyzing"
    EVALUATING = "evaluating"
    VISUALIZING = "visualizing"
    VALIDATING = "validating"
    HUMAN_REVIEW = "human_review"
    COMPLETED = "completed"
    FAILED = "failed"


class ReviewDecision(Enum):
    """Human review decision types"""
    APPROVED = "approve"
    REVISED = "revise"
    REJECTED = "reject"


# For forward compatibility with actual data types from core modules
class Pose3D:
    """Placeholder for core.Pose3D - will be imported at runtime"""
    pass


class CenterOfMass:
    """Placeholder for core.CenterOfMass - will be imported at runtime"""
    pass


class FootworkEvent:
    """Placeholder for core.FootworkEvent - will be imported at runtime"""
    pass


class ShotEvent:
    """Placeholder for core.ShotEvent - will be imported at runtime"""
    pass


class FootworkMetrics:
    """Placeholder for core.FootworkMetrics - will be imported at runtime"""
    pass


class EfficiencyScore:
    """Placeholder for core.EfficiencyScore - will be imported at runtime"""
    pass


class AgentState(TypedDict):
    """
    Main state structure that flows through the LangGraph multi-agent system.

    This TypedDict defines all the data that can be passed between agents
    in the workflow pipeline.
    """

    # ==================== INPUT DATA ====================
    """Input video path"""
    video_path: NotRequired[str]

    """List of video frames (numpy arrays)"""
    video_frames: NotRequired[List[np.ndarray]]

    """Video frame rate"""
    frame_rate: NotRequired[float]

    """Calibration data (pixel to meter conversion, etc.)"""
    calibration_data: NotRequired[Dict[str, Any]]

    # ==================== CONFIGURATION ====================
    """Analysis configuration (model_complexity, reference_level, etc.)"""
    config: NotRequired[Dict[str, Any]]

    # ==================== DETECTION RESULTS (DetectionAgent) ====================
    """Per-frame pose data from detection"""
    poses: NotRequired[List[Optional[Pose3D]]]

    """Detection quality statistics"""
    detection_stats: NotRequired[Dict[str, float]]

    # ==================== ANALYSIS RESULTS (AnalysisAgent) ====================
    """Center of mass positions per frame"""
    com_positions: NotRequired[List[CenterOfMass]]

    """Detected footwork events (jumps, landings, etc.)"""
    footwork_events: NotRequired[List[FootworkEvent]]

    """Detected shot events"""
    shot_events: NotRequired[List[ShotEvent]]

    """Calculated footwork metrics"""
    footwork_metrics: NotRequired[FootworkMetrics]

    """Biomechanics analysis results"""
    biomechanics_metrics: NotRequired[Dict[str, Any]]

    """Tactical analysis results"""
    tactical_metrics: NotRequired[Dict[str, Any]]

    """Rhythm control metrics"""
    rhythm_metrics: NotRequired[Dict[str, Any]]

    # ==================== EVALUATION RESULTS (EvaluationAgent) ====================
    """Overall efficiency score"""
    efficiency_score: NotRequired[EfficiencyScore]

    """Comparison results vs reference levels"""
    comparisons: NotRequired[Dict[str, Any]]

    """Personalized recommendations"""
    recommendations: NotRequired[List[Dict[str, Any]]]

    # ==================== VISUALIZATION DATA (VisualizationAgent) ====================
    """Trajectory data for plotting"""
    trajectory_data: NotRequired[List[tuple]]

    """Heatmap data (2D array)"""
    heatmap_data: NotRequired[np.ndarray]

    """Annotated video frames with pose overlay"""
    annotated_frames: NotRequired[List[np.ndarray]]

    """Dictionary of visualization outputs"""
    visualizations: NotRequired[Dict[str, np.ndarray]]

    # ==================== VALIDATION RESULTS (ValidationAgent) ====================
    """Validation results dictionary"""
    validation_results: NotRequired[Dict[str, Any]]

    """Quality scores for different aspects"""
    quality_scores: NotRequired[Dict[str, float]]

    """Detected anomalies"""
    anomalies: NotRequired[List[Dict[str, Any]]]

    """Whether human review is required"""
    requires_human_review: NotRequired[bool]

    """Reasons for requiring human review"""
    review_reasons: NotRequired[List[str]]

    # ==================== HUMAN-IN-THE-LOOP ====================
    """Human feedback received"""
    human_feedback: NotRequired[Dict[str, Any]]

    """Whether the current result is approved"""
    approved: NotRequired[bool]

    """Number of revision attempts"""
    revision_count: NotRequired[int]

    """Maximum number of revisions allowed"""
    max_revisions: NotRequired[int]

    # ==================== WORKFLOW STATE ====================
    """Current processing stage"""
    current_stage: NotRequired[ProcessingStage]

    """History of processing steps"""
    processing_history: NotRequired[List[Dict[str, Any]]]

    """Errors encountered during processing"""
    errors: NotRequired[List[Dict[str, str]]]

    """Warnings encountered during processing"""
    warnings: NotRequired[List[str]]

    # ==================== METADATA ====================
    """Unique session identifier"""
    session_id: NotRequired[str]

    """Timestamp of workflow start"""
    timestamp: NotRequired[float]

    """Total processing duration"""
    duration: NotRequired[float]

    """Output path for results"""
    output_path: NotRequired[str]


def create_initial_state(
    video_path: str,
    config: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> AgentState:
    """
    Create an initial state for the workflow.

    Args:
        video_path: Path to input video
        config: Analysis configuration
        session_id: Optional session identifier

    Returns:
        Initial AgentState dictionary
    """
    import uuid
    import time

    default_config = {
        "model_complexity": 1,
        "reference_level": "professional",
        "enable_smoothing": True,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "max_revisions": 3,
    }

    if config:
        default_config.update(config)

    return {
        "video_path": video_path,
        "video_frames": [],
        "frame_rate": 30.0,
        "calibration_data": {},
        "config": default_config,
        "poses": [],
        "detection_stats": {},
        "com_positions": [],
        "footwork_events": [],
        "shot_events": [],
        "footwork_metrics": None,
        "biomechanics_metrics": {},
        "tactical_metrics": {},
        "rhythm_metrics": {},
        "efficiency_score": None,
        "comparisons": {},
        "recommendations": [],
        "trajectory_data": [],
        "heatmap_data": None,
        "annotated_frames": [],
        "visualizations": {},
        "validation_results": {},
        "quality_scores": {},
        "anomalies": [],
        "requires_human_review": False,
        "review_reasons": [],
        "human_feedback": {},
        "approved": False,
        "revision_count": 0,
        "max_revisions": default_config.get("max_revisions", 3),
        "current_stage": ProcessingStage.INITIALIZED,
        "processing_history": [],
        "errors": [],
        "warnings": [],
        "session_id": session_id or str(uuid.uuid4()),
        "timestamp": time.time(),
        "duration": 0.0,
        "output_path": "",
    }
