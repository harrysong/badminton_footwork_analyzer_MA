"""Evaluation metrics for agents and system."""

from typing import Dict, Any, Optional, Protocol
from dataclasses import dataclass, field
from enum import Enum


class MetricType(Enum):
    """Types of evaluation metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    QUALITY = "quality"
    CUSTOM = "custom"


class ComparisonDirection(Enum):
    """Direction for metric comparison"""
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"


@dataclass
class EvaluationMetric:
    """Individual evaluation metric definition"""
    name: str
    description: str
    weight: float
    threshold: float
    metric_type: MetricType = MetricType.CUSTOM
    comparison_direction: ComparisonDirection = ComparisonDirection.HIGHER_BETTER
    unit: str = ""

    def evaluate(self, actual: float) -> Dict[str, Any]:
        """Evaluate actual value against threshold"""
        if self.comparison_direction == ComparisonDirection.HIGHER_BETTER:
            passed = actual >= self.threshold
            score = min(1.0, actual / self.threshold) if self.threshold > 0 else 0.0
        else:
            passed = actual <= self.threshold
            score = min(1.0, self.threshold / actual) if actual > 0 else 0.0

        return {
            "name": self.name,
            "actual": actual,
            "threshold": self.threshold,
            "passed": passed,
            "score": score,
            "unit": self.unit,
        }


@dataclass
class AgentMetrics:
    """Evaluation metrics for individual agents"""

    # DetectionAgent Metrics
    DETECTION_RATE = EvaluationMetric(
        name="detection_rate",
        description="Percentage of frames with successful pose detection",
        weight=0.3,
        threshold=0.80,
        metric_type=MetricType.QUALITY,
    )

    POSE_ACCURACY = EvaluationMetric(
        name="pose_accuracy",
        description="Accuracy of keypoint positions vs ground truth",
        weight=0.4,
        threshold=0.85,
        metric_type=MetricType.ACCURACY,
    )

    TRACKING_SMOOTHNESS = EvaluationMetric(
        name="tracking_smoothness",
        description="Temporal consistency of pose tracking",
        weight=0.3,
        threshold=0.90,
        metric_type=MetricType.QUALITY,
    )

    # AnalysisAgent Metrics
    EVENT_DETECTION_F1 = EvaluationMetric(
        name="event_detection_f1",
        description="F1 score for footwork event detection",
        weight=0.5,
        threshold=0.75,
        metric_type=MetricType.F1,
    )

    SHOT_CLASSIFICATION_ACCURACY = EvaluationMetric(
        name="shot_classification_accuracy",
        description="Accuracy of shot type classification",
        weight=0.3,
        threshold=0.70,
        metric_type=MetricType.ACCURACY,
    )

    METRIC_CORRELATION = EvaluationMetric(
        name="metric_correlation",
        description="Correlation with expert annotations",
        weight=0.2,
        threshold=0.80,
        metric_type=MetricType.CUSTOM,
    )

    # EvaluationAgent Metrics
    SCORE_CONSISTENCY = EvaluationMetric(
        name="score_consistency",
        description="Consistency of efficiency scores across similar videos",
        weight=0.4,
        threshold=0.85,
        metric_type=MetricType.CUSTOM,
    )

    RECOMMENDATION_RELEVANCE = EvaluationMetric(
        name="recommendation_relevance",
        description="Relevance of recommendations (human evaluation)",
        weight=0.6,
        threshold=0.80,
        metric_type=MetricType.CUSTOM,
    )

    # VisualizationAgent Metrics
    VISUALIZATION_RENDER_TIME = EvaluationMetric(
        name="visualization_render_time",
        description="Time to generate visualizations",
        weight=0.2,
        threshold=5.0,
        metric_type=MetricType.LATENCY,
        comparison_direction=ComparisonDirection.LOWER_BETTER,
        unit="seconds",
    )

    # ValidationAgent Metrics
    VALIDATION_ACCURACY = EvaluationMetric(
        name="validation_accuracy",
        description="Accuracy of validation decisions vs human judgment",
        weight=0.5,
        threshold=0.85,
        metric_type=MetricType.ACCURACY,
    )

    FALSE_POSITIVE_RATE = EvaluationMetric(
        name="false_positive_rate",
        description="Rate of false anomaly detections",
        weight=0.3,
        threshold=0.1,
        metric_type=MetricType.CUSTOM,
        comparison_direction=ComparisonDirection.LOWER_BETTER,
    )

    @classmethod
    def get_detection_metrics(cls) -> Dict[str, EvaluationMetric]:
        """Get all detection-related metrics"""
        return {
            "detection_rate": cls.DETECTION_RATE,
            "pose_accuracy": cls.POSE_ACCURACY,
            "tracking_smoothness": cls.TRACKING_SMOOTHNESS,
        }

    @classmethod
    def get_analysis_metrics(cls) -> Dict[str, EvaluationMetric]:
        """Get all analysis-related metrics"""
        return {
            "event_detection_f1": cls.EVENT_DETECTION_F1,
            "shot_classification_accuracy": cls.SHOT_CLASSIFICATION_ACCURACY,
            "metric_correlation": cls.METRIC_CORRELATION,
        }

    @classmethod
    def get_evaluation_metrics(cls) -> Dict[str, EvaluationMetric]:
        """Get all evaluation-related metrics"""
        return {
            "score_consistency": cls.SCORE_CONSISTENCY,
            "recommendation_relevance": cls.RECOMMENDATION_RELEVANCE,
        }


@dataclass
class SystemMetrics:
    """End-to-end system evaluation metrics"""

    LATENCY_P95 = EvaluationMetric(
        name="latency_p95",
        description="95th percentile processing latency",
        weight=0.2,
        threshold=5.0,
        metric_type=MetricType.LATENCY,
        comparison_direction=ComparisonDirection.LOWER_BETTER,
        unit="seconds",
    )

    LATENCY_P99 = EvaluationMetric(
        name="latency_p99",
        description="99th percentile processing latency",
        weight=0.1,
        threshold=10.0,
        metric_type=MetricType.LATENCY,
        comparison_direction=ComparisonDirection.LOWER_BETTER,
        unit="seconds",
    )

    THROUGHPUT = EvaluationMetric(
        name="throughput",
        description="Frames processed per second",
        weight=0.2,
        threshold=30.0,
        metric_type=MetricType.THROUGHPUT,
        unit="fps",
    )

    END_TO_END_ACCURACY = EvaluationMetric(
        name="end_to_end_accuracy",
        description="Overall accuracy vs human expert",
        weight=0.6,
        threshold=0.80,
        metric_type=MetricType.ACCURACY,
    )

    ERROR_RATE = EvaluationMetric(
        name="error_rate",
        description="Rate of processing failures",
        weight=0.2,
        threshold=0.05,
        metric_type=MetricType.CUSTOM,
        comparison_direction=ComparisonDirection.LOWER_BETTER,
    )

    @classmethod
    def get_all(cls) -> Dict[str, EvaluationMetric]:
        """Get all system metrics"""
        return {
            "latency_p95": cls.LATENCY_P95,
            "latency_p99": cls.LATENCY_P99,
            "throughput": cls.THROUGHPUT,
            "end_to_end_accuracy": cls.END_TO_END_ACCURACY,
            "error_rate": cls.ERROR_RATE,
        }


@dataclass
class EvaluationResult:
    """Result of evaluating metrics"""
    metric_name: str
    actual_value: float
    expected_value: float
    passed: bool
    score: float
    weight: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
            "passed": self.passed,
            "score": self.score,
            "weight": self.weight,
            "details": self.details,
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a set of evaluations"""
    total_score: float
    weighted_score: float
    passed_count: int
    failed_count: int
    total_count: int
    results: list = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total_count if self.total_count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "weighted_score": self.weighted_score,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_count": self.total_count,
            "pass_rate": self.pass_rate,
            "results": [r.to_dict() for r in self.results],
        }
