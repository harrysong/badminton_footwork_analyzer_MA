"""Evaluation Agent - wraps EfficiencyModel for LangGraph multi-agent system."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from core.efficiency_model import EfficiencyModel, EfficiencyScore, ComparisonResult
from core.footwork_analyzer import FootworkMetrics


class EvaluationAgent(BaseAgent):
    """
    Agent responsible for performance evaluation and comparison.

    Wraps EfficiencyModel to:
    - Calculate efficiency scores
    - Compare with reference profiles
    - Generate recommendations
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, agent_name="EvaluationAgent")
        self.efficiency_model: Optional[EfficiencyModel] = None
        self.reference_level: str = "professional"

    def _initialize(self) -> None:
        """Initialize the efficiency model."""
        self.efficiency_model = EfficiencyModel()
        self.reference_level = self.config.get("reference_level", "professional")

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate footwork metrics and generate scores.

        Args:
            state: Must contain:
                - footwork_metrics: FootworkMetrics object
                - biomechanics_metrics: Biomechanics metrics dict

        Returns:
            Updated state with:
                - efficiency_score: EfficiencyScore object
                - comparisons: Comparison results dict
                - recommendations: List of recommendations
        """
        footwork_metrics = state.get("footwork_metrics")
        biomechanics_metrics = state.get("biomechanics_metrics", {})

        if footwork_metrics is None:
            raise ValueError("No footwork metrics provided for evaluation")

        # Calculate efficiency score
        efficiency_score = self.efficiency_model.calculate_score(
            footwork_metrics,
            biomechanics_metrics,
            self.reference_level,
        )
        state["efficiency_score"] = efficiency_score

        # Compare with reference profiles
        comparisons = self._generate_comparisons(footwork_metrics)
        state["comparisons"] = {
            level: self.efficiency_model.compare_with_reference(
                footwork_metrics, level
            )
            for level in ["professional", "advanced", "intermediate"]
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(
            efficiency_score, comparisons, footwork_metrics
        )
        state["recommendations"] = recommendations

        state["current_stage"] = "evaluating"

        # Add custom metrics
        self.metrics.custom_metrics = {
            "overall_score": efficiency_score.overall,
            "movement_efficiency": efficiency_score.movement_efficiency,
            "reference_level": self.reference_level,
        }

        return state

    def _generate_comparisons(
        self, footwork_metrics: FootworkMetrics
    ) -> Dict[str, Any]:
        """Generate comparison results for each metric."""
        comparisons = {}

        # Path efficiency comparison
        if hasattr(footwork_metrics, "path_efficiency"):
            comparisons["path_efficiency"] = {
                "value": footwork_metrics.path_efficiency,
                "reference": self.efficiency_model.reference_profiles.get(
                    self.reference_level
                ),
            }

        # Step frequency comparison
        if hasattr(footwork_metrics, "step_frequency"):
            comparisons["step_frequency"] = {
                "value": footwork_metrics.step_frequency,
                "reference": self.efficiency_model.reference_profiles.get(
                    self.reference_level
                ),
            }

        # Response time comparison
        if hasattr(footwork_metrics, "avg_response_time"):
            comparisons["avg_response_time"] = {
                "value": footwork_metrics.avg_response_time,
                "reference": self.efficiency_model.reference_profiles.get(
                    self.reference_level
                ),
            }

        return comparisons

    def _generate_recommendations(
        self,
        efficiency_score: EfficiencyScore,
        comparisons: Dict[str, Any],
        footwork_metrics: FootworkMetrics,
    ) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on scores."""
        recommendations = []

        # Movement efficiency recommendations
        if efficiency_score.movement_efficiency < 60:
            recommendations.append({
                "category": "movement_efficiency",
                "priority": "high",
                "title": "提升移动效率",
                "description": "您的路径效率较低，建议优化跑动路线，减少不必要的移动。",
                "metric": "path_efficiency",
                "current_value": getattr(footwork_metrics, "path_efficiency", 0),
                "target_value": 0.45,
            })
        elif efficiency_score.movement_efficiency < 80:
            recommendations.append({
                "category": "movement_efficiency",
                "priority": "medium",
                "title": "优化移动节奏",
                "description": "您的移动效率有提升空间，注意预判球路以优化移动路线。",
                "metric": "path_efficiency",
                "current_value": getattr(footwork_metrics, "path_efficiency", 0),
            })

        # Response time recommendations
        if efficiency_score.response_time < 60:
            recommendations.append({
                "category": "response_time",
                "priority": "high",
                "title": "提升反应速度",
                "description": "您的反应时间较长，建议加强快速启动训练。",
                "metric": "avg_response_time",
                "current_value": getattr(footwork_metrics, "avg_response_time", 0),
                "target_value": 0.5,
            })

        # Court coverage recommendations
        if efficiency_score.court_coverage < 60:
            recommendations.append({
                "category": "court_coverage",
                "priority": "medium",
                "title": "扩大场地覆盖",
                "description": "您的场地覆盖面积有限，注意加强前后场的移动能力。",
                "metric": "coverage_ratio",
                "current_value": getattr(footwork_metrics, "coverage_ratio", 0),
                "target_value": 0.4,
            })

        # Balance recommendations
        if efficiency_score.balance_stability < 60:
            recommendations.append({
                "category": "balance_stability",
                "priority": "medium",
                "title": "提升身体平衡",
                "description": "建议加强平衡训练和重心控制练习。",
                "metric": "com_stability",
                "current_value": 0,
            })

        # Add positive feedback if scores are good
        if efficiency_score.overall >= 85:
            recommendations.append({
                "category": "excellent",
                "priority": "low",
                "title": "优秀表现",
                "description": "您的步法表现出色！继续保持当前的训练方式。",
            })
        elif efficiency_score.overall >= 70:
            recommendations.append({
                "category": "good",
                "priority": "low",
                "title": "良好表现",
                "description": "您的步法整体良好，针对薄弱环节进行针对性训练可进一步提升。",
            })

        return recommendations

    def set_reference_level(self, level: str) -> None:
        """Change the reference level for comparison."""
        if level in ["professional", "advanced", "intermediate"]:
            self.reference_level = level
        else:
            raise ValueError(f"Invalid reference level: {level}")

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self.reference_level = self.config.get("reference_level", "professional")
