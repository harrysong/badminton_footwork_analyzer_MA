"""LangGraph workflow builder for badminton footwork analysis."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Literal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState, ProcessingStage


def build_badminton_analysis_graph() -> StateGraph:
    """
    Build the LangGraph state graph for badminton analysis.

    Workflow:
    START -> DETECTION -> ANALYSIS -> EVALUATION -> VISUALIZATION -> VALIDATION
                                                              |
                         +--------------------------------------+
                         |                                      |
                         v                                      v
                    COMPLETE                              HUMAN_REVIEW
                         |                                      |
                         +------------------+-------------------+
                                            |
                                            v
                                       APPROVED/REVISED
                                            |
                                            v
                                          END

    Returns:
        Compiled StateGraph for the multi-agent workflow
    """
    from agents.detection_agent import DetectionAgent
    from agents.analysis_agent import AnalysisAgent
    from agents.evaluation_agent import EvaluationAgent
    from agents.visualization_agent import VisualizationAgent
    from agents.validation_agent import ValidationAgent

    # Create configuration
    config = {
        "model_complexity": 1,
        "reference_level": "professional",
        "enable_smoothing": True,
    }

    # Initialize agents
    detection_agent = DetectionAgent(config)
    analysis_agent = AnalysisAgent(config)
    evaluation_agent = EvaluationAgent(config)
    visualization_agent = VisualizationAgent(config)
    validation_agent = ValidationAgent(config)

    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("detection", detection_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("visualization", visualization_node)
    graph.add_node("validation", validation_node)
    graph.add_node("human_review", human_review_node)

    # Set entry point
    graph.set_entry_point("detection")

    # Add sequential edges
    graph.add_edge("detection", "analysis")
    graph.add_edge("analysis", "evaluation")
    graph.add_edge("evaluation", "visualization")
    graph.add_edge("visualization", "validation")

    # Add conditional edge from validation
    graph.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "complete": END,
            "revise": "analysis",
            "human_review": "human_review",
        }
    )

    # Add conditional edge from human review
    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "approved": END,
            "revise": "analysis",
            "rejected": END,
        }
    )

    # Compile with checkpointer for state persistence
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def detection_node(state: AgentState) -> AgentState:
    """Detection node - runs pose detection."""
    from agents.detection_agent import DetectionAgent
    from graph.state import ProcessingStage

    config = state.get("config", {})
    agent = DetectionAgent(config)

    state = agent.execute(state)
    state["current_stage"] = ProcessingStage.DETECTING

    return state


def analysis_node(state: AgentState) -> AgentState:
    """Analysis node - runs footwork and shot analysis."""
    from agents.analysis_agent import AnalysisAgent
    from graph.state import ProcessingStage

    config = state.get("config", {})
    agent = AnalysisAgent(config)

    state = agent.execute(state)
    state["current_stage"] = ProcessingStage.ANALYZING

    return state


def evaluation_node(state: AgentState) -> AgentState:
    """Evaluation node - runs efficiency evaluation."""
    from agents.evaluation_agent import EvaluationAgent
    from graph.state import ProcessingStage

    config = state.get("config", {})
    agent = EvaluationAgent(config)

    state = agent.execute(state)
    state["current_stage"] = ProcessingStage.EVALUATING

    return state


def visualization_node(state: AgentState) -> AgentState:
    """Visualization node - generates visualizations."""
    from agents.visualization_agent import VisualizationAgent
    from graph.state import ProcessingStage

    config = state.get("config", {})
    agent = VisualizationAgent(config)

    state = agent.execute(state)
    state["current_stage"] = ProcessingStage.VISUALIZING

    return state


def validation_node(state: AgentState) -> AgentState:
    """Validation node - validates results and quality."""
    from agents.validation_agent import ValidationAgent
    from graph.state import ProcessingStage

    config = state.get("config", {})
    agent = ValidationAgent(config)

    state = agent.execute(state)
    state["current_stage"] = ProcessingStage.VALIDATING

    return state


def human_review_node(state: AgentState) -> AgentState:
    """Human review node - handles human-in-the-loop."""
    from graph.state import ProcessingStage

    # This is a placeholder - in real implementation,
    # this would integrate with a UI for human feedback
    state["current_stage"] = ProcessingStage.HUMAN_REVIEW

    # If no human feedback yet, just pass through
    if not state.get("human_feedback"):
        state["approved"] = False

    return state


def route_after_validation(state: AgentState) -> Literal["complete", "revise", "human_review"]:
    """
    Determine next step after validation.

    Returns:
        - "complete": Validation passed, workflow complete
        - "revise": Validation failed, retry analysis
        - "human_review": Requires human review
    """
    validation_results = state.get("validation_results", {})
    requires_review = state.get("requires_human_review", False)
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)

    # Check for critical errors first
    errors = state.get("errors", [])
    if errors:
        if revision_count < max_revisions:
            return "revise"
        else:
            return "complete"  # Give up after max revisions

    # Check if human review is needed
    if requires_review:
        return "human_review"

    # Check if validation passed
    if validation_results.get("quality_passed", False):
        return "complete"

    # Validation failed but no critical issues
    if revision_count < max_revisions:
        state["revision_count"] = revision_count + 1
        return "revise"

    return "complete"


def route_after_review(state: AgentState) -> Literal["approved", "revise", "rejected"]:
    """
    Determine next step after human review.

    Returns:
        - "approved": Human approved the results
        - "revise": Human requested revision
        - "rejected": Human rejected the results
    """
    human_feedback = state.get("human_feedback", {})
    action = human_feedback.get("action", "approve")

    if action == "approve":
        return "approved"
    elif action == "revise":
        return "revise"
    else:
        return "rejected"


# Convenience function to run the graph
def run_analysis(
    video_path: str,
    config: Optional[Dict[str, Any]] = None,
    checkpointer: Optional[Any] = None,
) -> AgentState:
    """
    Run the complete analysis workflow.

    Args:
        video_path: Path to input video
        config: Optional configuration overrides
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Final AgentState with all results
    """
    from graph.state import create_initial_state

    # Create initial state
    state = create_initial_state(video_path, config)

    # Build and run the graph
    graph = build_badminton_analysis_graph()

    # Run with optional checkpointer
    if checkpointer:
        config_runtime = {"configurable": {"thread_id": "default"}}
        result = graph.invoke(state, config_runtime)
    else:
        result = graph.invoke(state)

    return result
