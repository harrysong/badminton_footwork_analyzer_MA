"""Validation Agent - quality control and validation for LangGraph multi-agent system."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentMetrics


class ValidationAgent(BaseAgent):
    """
    Agent responsible for quality control and validation.

    Performs:
    - Result quality assessment
    - Anomaly detection
    - Confidence scoring
    - Human review triggering
    """

    # Quality thresholds
    MIN_DETECTION_RATE = 0.7
    MIN_CONFIDENCE = 0.5
    MAX_ANOMALY_SCORE = 0.3
    MIN_EFFICIENCY_SCORE = 0.0
    MAX_EFFICIENCY_SCORE = 100.0

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, agent_name="ValidationAgent")
        self.quality_thresholds = {
            "detection_rate": self.config.get("min_detection_rate", self.MIN_DETECTION_RATE),
            "confidence": self.config.get("min_confidence", self.MIN_CONFIDENCE),
            "anomaly_score": self.config.get("max_anomaly_score", self.MAX_ANOMALY_SCORE),
        }

    def _initialize(self) -> None:
        """Initialize validation components."""
        pass

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate analysis results and assess quality.

        Args:
            state: Must contain:
                - detection_stats: Detection quality statistics
                - poses: Detected poses
                - footwork_metrics: Footwork metrics
                - efficiency_score: Efficiency score (optional)

        Returns:
            Updated state with:
                - validation_results: Validation results dict
                - quality_scores: Quality scores for different aspects
                - anomalies: Detected anomalies
                - requires_human_review: Whether human review is needed
                - review_reasons: Reasons for review
        """
        # Collect quality metrics
        quality_scores = {}
        anomalies = []
        review_reasons = []

        # Validate detection quality
        detection_stats = state.get("detection_stats", {})
        detection_score = self._validate_detection(detection_stats)
        quality_scores["detection"] = detection_score

        if detection_score < self.quality_thresholds["detection_rate"]:
            anomalies.append({
                "type": "low_detection_rate",
                "severity": "high",
                "value": detection_stats.get("detection_rate", 0),
                "threshold": self.quality_thresholds["detection_rate"],
            })
            review_reasons.append("检测率过低，可能影响分析准确性")

        # Validate pose confidence
        poses = state.get("poses", [])
        confidence_score = self._validate_confidence(poses)
        quality_scores["confidence"] = confidence_score

        if confidence_score < self.quality_thresholds["confidence"]:
            anomalies.append({
                "type": "low_confidence",
                "severity": "medium",
                "value": confidence_score,
                "threshold": self.quality_thresholds["confidence"],
            })
            review_reasons.append("姿态检测置信度较低")

        # Validate metrics reasonableness
        footwork_metrics = state.get("footwork_metrics")
        metrics_score = self._validate_metrics(footwork_metrics)
        quality_scores["metrics"] = metrics_score

        if metrics_score < 0.5:
            anomalies.append({
                "type": "unreasonable_metrics",
                "severity": "medium",
                "value": metrics_score,
            })
            review_reasons.append("步法指标异常")

        # Validate efficiency score
        efficiency_score = state.get("efficiency_score")
        efficiency_score_validation = self._validate_efficiency_score(efficiency_score)
        quality_scores["efficiency"] = efficiency_score_validation

        # Overall quality score
        overall_score = np.mean([
            quality_scores.get("detection", 0),
            quality_scores.get("confidence", 0),
            quality_scores.get("metrics", 0),
            quality_scores.get("efficiency", 0),
        ])
        quality_scores["overall"] = overall_score

        # Determine if human review is needed
        requires_human_review = self._should_require_review(
            quality_scores, anomalies, review_reasons
        )

        # Determine if validation passed
        quality_passed = overall_score >= 0.6 and len(anomalies) == 0

        # Compile validation results
        validation_results = {
            "quality_passed": quality_passed,
            "overall_score": overall_score,
            "thresholds_met": self._check_thresholds(quality_scores),
            "failed_checks": self._get_failed_checks(quality_scores),
        }

        state["validation_results"] = validation_results
        state["quality_scores"] = quality_scores
        state["anomalies"] = anomalies
        state["requires_human_review"] = requires_human_review
        state["review_reasons"] = review_reasons
        state["current_stage"] = "validating"

        # Add custom metrics
        self.metrics.custom_metrics = {
            "overall_quality_score": overall_score,
            "anomaly_count": len(anomalies),
            "requires_review": requires_human_review,
        }

        return state

    def _validate_detection(self, detection_stats: Dict[str, float]) -> float:
        """Validate detection quality."""
        detection_rate = detection_stats.get("detection_rate", 0.0)

        # Linear scoring: 0.7 = 0.5 score, 1.0 = 1.0 score
        if detection_rate >= 1.0:
            return 1.0
        elif detection_rate >= 0.7:
            return 0.5 + 0.5 * (detection_rate - 0.7) / 0.3
        else:
            return 0.5 * detection_rate / 0.7

    def _validate_confidence(self, poses: List) -> float:
        """Validate pose detection confidence."""
        if not poses:
            return 0.0

        valid_poses = [p for p in poses if p is not None]
        if not valid_poses:
            return 0.0

        # Calculate average visibility
        confidences = []
        for pose in valid_poses:
            if hasattr(pose, "visibility"):
                confidences.append(np.mean(pose.visibility))

        if not confidences:
            return 0.0

        avg_confidence = np.mean(confidences)

        # Scale: 0.5 = 0.5 score, 1.0 = 1.0 score
        if avg_confidence >= 1.0:
            return 1.0
        elif avg_confidence >= 0.5:
            return 0.5 + 0.5 * (avg_confidence - 0.5) / 0.5
        else:
            return 0.5 * avg_confidence / 0.5

    def _validate_metrics(self, metrics) -> float:
        """Validate that metrics are reasonable."""
        if metrics is None:
            return 0.0

        score = 1.0

        # Check if metrics have reasonable values
        if hasattr(metrics, "total_steps"):
            if metrics.total_steps < 0:
                score = 0.0
            elif metrics.total_steps > 1000:
                score *= 0.5  # Too many steps might indicate noise

        if hasattr(metrics, "path_efficiency"):
            if not 0 <= metrics.path_efficiency <= 1:
                score *= 0.5

        if hasattr(metrics, "avg_speed"):
            if metrics.avg_speed > 10:  # Unreasonably fast
                score *= 0.5

        return max(0.0, min(1.0, score))

    def _validate_efficiency_score(self, efficiency_score) -> float:
        """Validate efficiency score."""
        if efficiency_score is None:
            return 0.5  # Neutral if not calculated

        if not hasattr(efficiency_score, "overall"):
            return 0.5

        overall = efficiency_score.overall

        # Check if score is in valid range
        if not self.MIN_EFFICIENCY_SCORE <= overall <= self.MAX_EFFICIENCY_SCORE:
            return 0.0

        # Score is valid
        return 1.0

    def _should_require_review(
        self,
        quality_scores: Dict[str, float],
        anomalies: List[Dict],
        reasons: List[str],
    ) -> bool:
        """Determine if human review is required."""
        # Always review if there are high severity anomalies
        high_severity = [a for a in anomalies if a.get("severity") == "high"]
        if high_severity:
            return True

        # Review if overall quality is low
        if quality_scores.get("overall", 0) < 0.5:
            return True

        # Review if detection rate is critically low
        if quality_scores.get("detection", 0) < 0.4:
            return True

        # Review if multiple anomalies
        if len(anomalies) >= 3:
            return True

        return False

    def _check_thresholds(self, quality_scores: Dict[str, float]) -> Dict[str, bool]:
        """Check which thresholds are met."""
        return {
            "detection": quality_scores.get("detection", 0) >= self.quality_thresholds["detection_rate"],
            "confidence": quality_scores.get("confidence", 0) >= self.quality_thresholds["confidence"],
            "metrics": quality_scores.get("metrics", 0) >= 0.5,
            "efficiency": quality_scores.get("efficiency", 0) >= 0.5,
        }

    def _get_failed_checks(self, quality_scores: Dict[str, float]) -> List[str]:
        """Get list of failed quality checks."""
        failed = []

        if quality_scores.get("detection", 0) < self.quality_thresholds["detection_rate"]:
            failed.append("detection_rate")

        if quality_scores.get("confidence", 0) < self.quality_thresholds["confidence"]:
            failed.append("confidence")

        if quality_scores.get("metrics", 0) < 0.5:
            failed.append("metrics_quality")

        if quality_scores.get("efficiency", 0) < 0.5:
            failed.append("efficiency_score")

        return failed

    def set_thresholds(
        self,
        detection_rate: Optional[float] = None,
        confidence: Optional[float] = None,
        anomaly_score: Optional[float] = None,
    ) -> None:
        """Update quality thresholds."""
        if detection_rate is not None:
            self.quality_thresholds["detection_rate"] = detection_rate
        if confidence is not None:
            self.quality_thresholds["confidence"] = confidence
        if anomaly_score is not None:
            self.quality_thresholds["anomaly_score"] = anomaly_score
