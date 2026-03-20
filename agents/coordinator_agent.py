"""Coordinator Agent - orchestrates the LangGraph multi-agent workflow."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from agents.detection_agent import DetectionAgent
from agents.analysis_agent import AnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.visualization_agent import VisualizationAgent
from agents.validation_agent import ValidationAgent
from graph.state import AgentState, create_initial_state, ProcessingStage


class CoordinatorAgent(BaseAgent):
    """
    Central coordinator for multi-agent orchestration.

    Manages the LangGraph workflow and handles:
    - Task orchestration
    - Error handling and recovery
    - State management
    - Human-in-the-loop coordination
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, agent_name="CoordinatorAgent")

        # Initialize all agents
        self.detection_agent: Optional[DetectionAgent] = None
        self.analysis_agent: Optional[AnalysisAgent] = None
        self.evaluation_agent: Optional[EvaluationAgent] = None
        self.visualization_agent: Optional[VisualizationAgent] = None
        self.validation_agent: Optional[ValidationAgent] = None

        # Workflow configuration
        self.max_revisions = config.get("max_revisions", 3)
        self.enable_human_review = config.get("enable_human_review", True)

        # Callbacks for human-in-the-loop
        self.on_review_request: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

    def _initialize(self) -> None:
        """Initialize all agents with shared configuration."""
        self.detection_agent = DetectionAgent(self.config)
        self.analysis_agent = AnalysisAgent(self.config)
        self.evaluation_agent = EvaluationAgent(self.config)
        self.visualization_agent = VisualizationAgent(self.config)
        self.validation_agent = ValidationAgent(self.config)

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete analysis pipeline.

        This is the main entry point that runs the full workflow:
        Detection -> Analysis -> Evaluation -> Visualization -> Validation

        Args:
            state: Initial state with video_path or video_frames

        Returns:
            Final state with all results
        """
        import time

        start_time = time.time()
        initial_state = state.copy()

        try:
            # Stage 1: Detection
            state = self._run_detection(state)
            if self._has_errors(state):
                return self._handle_error(state, "detection")

            # Stage 2: Analysis
            state = self._run_analysis(state)
            if self._has_errors(state):
                return self._handle_error(state, "analysis")

            # Stage 3: Evaluation
            state = self._run_evaluation(state)
            if self._has_errors(state):
                return self._handle_error(state, "evaluation")

            # Stage 4: Visualization
            state = self._run_visualization(state)
            if self._has_errors(state):
                return self._handle_error(state, "visualization")

            # Stage 5: Validation
            state = self._run_validation(state)
            if self._has_errors(state):
                return self._handle_error(state, "validation")

            # Handle validation results
            state = self._handle_validation_result(state)

            # Mark completion
            state["current_stage"] = ProcessingStage.COMPLETED
            state["duration"] = time.time() - start_time

            if self.on_complete:
                self.on_complete(state)

            return state

        except Exception as e:
            state["errors"] = state.get("errors", [])
            state["errors"].append({
                "stage": "coordinator",
                "error": str(e),
                "timestamp": time.time(),
            })
            state["current_stage"] = ProcessingStage.FAILED

            if self.on_error:
                self.on_error(state, e)

            raise

    def _run_detection(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run detection agent."""
        state["current_stage"] = ProcessingStage.DETECTING
        return self.detection_agent.execute(state)

    def _run_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis agent."""
        state["current_stage"] = ProcessingStage.ANALYZING
        return self.analysis_agent.execute(state)

    def _run_evaluation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation agent."""
        state["current_stage"] = ProcessingStage.EVALUATING
        return self.evaluation_agent.execute(state)

    def _run_visualization(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run visualization agent."""
        state["current_stage"] = ProcessingStage.VISUALIZING
        return self.visualization_agent.execute(state)

    def _run_validation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation agent."""
        state["current_stage"] = ProcessingStage.VALIDATING
        return self.validation_agent.execute(state)

    def _handle_validation_result(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validation results and determine next action."""
        requires_review = state.get("requires_human_review", False)
        validation_passed = state.get("validation_results", {}).get("quality_passed", False)

        # Check if we should request human review
        if requires_review and self.enable_human_review:
            state["current_stage"] = ProcessingStage.HUMAN_REVIEW

            if self.on_review_request:
                # Request human review through callback
                feedback = self.on_review_request(state)
                state = self._process_human_feedback(state, feedback)
            else:
                # No callback, auto-approve if validation mostly passed
                if validation_passed:
                    state["approved"] = True
                else:
                    # Try revision if not maxed out
                    revision_count = state.get("revision_count", 0)
                    if revision_count < self.max_revisions:
                        state["revision_count"] = revision_count + 1
                        # Could implement auto-revision logic here
                        state["approved"] = False
                    else:
                        state["approved"] = True  # Give up after max revisions

        return state

    def _process_human_feedback(
        self, state: Dict[str, Any], feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process human feedback and update state."""
        state["human_feedback"] = feedback

        action = feedback.get("action", "approve")

        if action == "approve":
            state["approved"] = True
        elif action == "revise":
            state["approved"] = False
            # Apply adjustments if provided
            adjustments = feedback.get("adjustments", {})
            if adjustments:
                self.config.update(adjustments)
            # Increment revision count
            state["revision_count"] = state.get("revision_count", 0) + 1
        else:  # reject
            state["approved"] = False

        return state

    def _has_errors(self, state: Dict[str, Any]) -> bool:
        """Check if state contains errors."""
        errors = state.get("errors", [])
        return len(errors) > 0

    def _handle_error(self, state: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Handle errors and attempt recovery."""
        state["current_stage"] = ProcessingStage.FAILED

        # Log error
        self.metrics.warnings.append(f"Error at stage: {stage}")

        return state

    def run_step(self, state: Dict[str, Any], step: str) -> Dict[str, Any]:
        """Run a single step of the workflow."""
        step_handlers = {
            "detection": self._run_detection,
            "analysis": self._run_analysis,
            "evaluation": self._run_evaluation,
            "visualization": self._run_visualization,
            "validation": self._run_validation,
        }

        if step not in step_handlers:
            raise ValueError(f"Unknown step: {step}")

        return step_handlers[step](state)

    def get_agent_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all agents."""
        return {
            "detection": self.detection_agent.get_metrics() if self.detection_agent else {},
            "analysis": self.analysis_agent.get_metrics() if self.analysis_agent else {},
            "evaluation": self.evaluation_agent.get_metrics() if self.evaluation_agent else {},
            "visualization": self.visualization_agent.get_metrics() if self.visualization_agent else {},
            "validation": self.validation_agent.get_metrics() if self.validation_agent else {},
        }

    def set_review_callback(self, callback: Callable) -> None:
        """Set callback for human review requests."""
        self.on_review_request = callback

    def set_complete_callback(self, callback: Callable) -> None:
        """Set callback for workflow completion."""
        self.on_complete = callback

    def set_error_callback(self, callback: Callable) -> None:
        """Set callback for errors."""
        self.on_error = callback

    def reset(self) -> None:
        """Reset all agents."""
        super().reset()
        if self.detection_agent:
            self.detection_agent.reset()
        if self.analysis_agent:
            self.analysis_agent.reset()
        if self.evaluation_agent:
            self.evaluation_agent.reset()
        if self.visualization_agent:
            pass  # No reset method
        if self.validation_agent:
            pass  # No reset method


def create_coordinator(config: Optional[Dict[str, Any]] = None) -> CoordinatorAgent:
    """Factory function to create a coordinator with default config."""
    default_config = {
        "model_complexity": 1,
        "reference_level": "professional",
        "enable_smoothing": True,
        "min_detection_confidence": 0.5,
        "max_revisions": 3,
        "enable_human_review": True,
    }

    if config:
        default_config.update(config)

    return CoordinatorAgent(default_config)
