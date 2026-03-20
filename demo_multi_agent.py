#!/usr/bin/env python3
"""
Demo script for LangGraph Multi-Agent System for Badminton Footwork Analysis.

This script demonstrates:
1. How to use the CoordinatorAgent for end-to-end analysis
2. How to run individual agents
3. How to use the evaluation framework
4. How to use the LangGraph workflow

Usage:
    python demo_multi_agent.py [--video PATH] [--config CONFIG]

Example:
    python demo_multi_agent.py --video data/videos/sample.mp4
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import json

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from agents.coordinator_agent import CoordinatorAgent, create_coordinator
from agents.detection_agent import DetectionAgent
from agents.analysis_agent import AnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.visualization_agent import VisualizationAgent
from agents.validation_agent import ValidationAgent
from evaluation.evaluators import EvalDrivenTestPipeline
from evaluation.benchmarks import create_sample_benchmark_dataset
from graph.state import create_initial_state


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_state_summary(state: Dict[str, Any]):
    """Print a summary of the analysis state."""
    print("\n📊 Analysis Results Summary:")
    print("-" * 40)

    # Detection results
    if "detection_stats" in state:
        stats = state["detection_stats"]
        print(f"  Detection Rate: {stats.get('detection_rate', 0):.1%}")
        print(f"  Avg Confidence: {stats.get('avg_confidence', 0):.2f}")
        print(f"  Frames Processed: {stats.get('total_frames', 0)}")

    # Analysis results
    if "footwork_metrics" in state and state["footwork_metrics"]:
        metrics = state["footwork_metrics"]
        print(f"\n  Footwork Metrics:")
        print(f"    Total Steps: {metrics.total_steps}")
        print(f"    Step Frequency: {metrics.step_frequency:.2f} steps/s")
        print(f"    Path Efficiency: {metrics.path_efficiency:.2f}")
        print(f"    Avg Response Time: {metrics.avg_response_time:.3f}s")

    # Evaluation results
    if "efficiency_score" in state and state["efficiency_score"]:
        score = state["efficiency_score"]
        print(f"\n  Efficiency Score:")
        print(f"    Overall: {score.overall:.1f}")
        print(f"    Movement Efficiency: {score.movement_efficiency:.1f}")
        print(f"    Response Time: {score.response_time:.1f}")
        print(f"    Court Coverage: {score.court_coverage:.1f}")
        print(f"    Balance Stability: {score.balance_stability:.1f}")

    # Recommendations
    if "recommendations" in state and state["recommendations"]:
        print(f"\n  Recommendations:")
        for rec in state["recommendations"][:3]:  # Show top 3
            priority = rec.get("priority", "low")
            title = rec.get("title", "Unknown")
            print(f"    [{priority.upper()}] {title}")

    # Validation results
    if "validation_results" in state:
        validation = state["validation_results"]
        print(f"\n  Validation:")
        print(f"    Quality Passed: {validation.get('quality_passed', False)}")
        print(f"    Overall Score: {validation.get('overall_score', 0):.2f}")

    if "quality_scores" in state:
        scores = state["quality_scores"]
        print(f"    Detection Score: {scores.get('detection', 0):.2f}")
        print(f"    Confidence Score: {scores.get('confidence', 0):.2f}")

    print()


def demo_individual_agents(video_path: str):
    """Demonstrate using individual agents."""
    print_section("Demo 1: Individual Agents")

    config = {
        "model_complexity": 1,
        "reference_level": "professional",
        "enable_smoothing": True,
    }

    # Detection Agent
    print("\n🔍 Running DetectionAgent...")
    detection_agent = DetectionAgent(config)
    # Note: In real usage, you'd pass actual video frames
    # For demo, we'll just initialize
    print(f"  - Initialized: {detection_agent.name}")
    print(f"  - Pose tracker ready")

    # Analysis Agent
    print("\n📈 Running AnalysisAgent...")
    analysis_agent = AnalysisAgent(config)
    print(f"  - Initialized: {analysis_agent.name}")
    print(f"  - CoM calculator ready")
    print(f"  - Footwork analyzer ready")

    # Evaluation Agent
    print("\n🎯 Running EvaluationAgent...")
    evaluation_agent = EvaluationAgent(config)
    print(f"  - Initialized: {evaluation_agent.name}")
    print(f"  - Efficiency model ready")
    print(f"  - Reference level: {evaluation_agent.reference_level}")

    print("\n✅ Individual agents demo complete!")


def demo_coordinator_agent(video_path: str):
    """Demonstrate using the CoordinatorAgent."""
    print_section("Demo 2: Coordinator Agent (Full Pipeline)")

    config = {
        "model_complexity": 1,
        "reference_level": "professional",
        "enable_smoothing": True,
        "min_detection_confidence": 0.5,
        "max_revisions": 3,
        "enable_human_review": False,  # Disable for demo
    }

    print("\n🚀 Creating CoordinatorAgent...")
    coordinator = create_coordinator(config)
    print(f"  - Initialized: {coordinator.name}")
    print(f"  - Sub-agents initialized")

    # Show agent metrics
    print("\n📋 Registered Agents:")
    metrics = coordinator.get_agent_metrics()
    for agent_name in metrics:
        print(f"  - {agent_name}")

    # Note: Actual video processing requires a real video file
    # For demo, we just show the structure
    print("\n⚠️  Note: Full pipeline requires a real video file.")
    print(f"  Video path: {video_path}")

    print("\n✅ Coordinator agent demo complete!")


def demo_langgraph_workflow(video_path: str):
    """Demonstrate using the LangGraph workflow."""
    print_section("Demo 3: LangGraph Workflow")

    try:
        from src.graph.graph_builder import build_badminton_analysis_graph

        print("\n🔄 Building LangGraph workflow...")

        # Build the graph
        graph = build_badminton_analysis_graph()

        print("  - Graph built successfully")
        print("  - Nodes: detection -> analysis -> evaluation -> visualization -> validation")
        print("  - Conditional edges for revision/review")

        # Show graph structure
        print("\n📐 Graph Structure:")
        print("  START")
        print("   │")
        print("   ▼")
        print("  DETECTION")
        print("   │")
        print("   ▼")
        print("  ANALYSIS")
        print("   │")
        print("   ▼")
        print("  EVALUATION")
        print("   │")
        print("   ▼")
        print("  VISUALIZATION")
        print("   │")
        print("   ▼")
        print("  VALIDATION")
        print("   │")
        print("   ├──────────────┐")
        print("   ▼              ▼")
        print(" COMPLETE    HUMAN_REVIEW")
        print("   │              │")
        print("   └──────────────┘")
        print("        END")

        print("\n✅ LangGraph workflow demo complete!")

    except ImportError as e:
        print(f"  ⚠️  LangGraph not available: {e}")


def demo_evaluation_framework():
    """Demonstrate the evaluation framework."""
    print_section("Demo 4: Evaluation Framework")

    # Create test pipeline
    print("\n📐 Creating EvalDrivenTestPipeline...")
    pipeline = EvalDrivenTestPipeline()

    print("  - Pipeline created")

    # Create sample benchmark dataset
    print("\n📊 Creating sample benchmark dataset...")
    dataset = create_sample_benchmark_dataset()
    print(f"  - Dataset: {dataset.name}")
    print(f"  - Samples: {len(dataset.samples)}")
    print(f"  - Train: {len(dataset.train_samples)}, Val: {len(dataset.val_samples)}, Test: {len(dataset.test_samples)}")

    # Show sample
    if dataset.samples:
        sample = dataset.samples[0]
        print(f"\n  Sample: {sample.sample_id}")
        print(f"    Difficulty: {sample.difficulty}")
        print(f"    Duration: {sample.duration}s")
        print(f"    Expected Score: {sample.efficiency_scores.get('overall', 'N/A')}")

    print("\n✅ Evaluation framework demo complete!")


def demo_optimization():
    """Demonstrate the optimization framework."""
    print_section("Demo 5: Optimization Framework")

    from src.evaluation.optimizer import ParameterTuner, ABTestRunner

    print("\n🔧 Creating ParameterTuner...")

    # Example evaluation function
    def evaluate_config(config: Dict) -> float:
        """Example evaluation function."""
        # Simulate evaluation
        base_score = 70.0

        # Simple scoring based on parameters
        complexity_bonus = config.get("model_complexity", 0) * 2
        smoothing_bonus = 5 if config.get("enable_smoothing", False) else 0

        return base_score + complexity_bonus + smoothing_bonus

    tuner = ParameterTuner(evaluate_config)

    # Simple grid search
    print("\n🔍 Running grid search...")
    param_grid = {
        "model_complexity": [0, 1, 2],
        "enable_smoothing": [True, False],
    }

    best_params, best_score = tuner.grid_search(param_grid)
    print(f"  - Best params: {best_params}")
    print(f"  - Best score: {best_score:.2f}")

    # A/B Testing
    print("\n🧪 A/B Testing...")
    ab_runner = ABTestRunner(evaluate_config)

    variant_a = {"model_complexity": 1, "enable_smoothing": True}
    variant_b = {"model_complexity": 2, "enable_smoothing": True}

    print(f"  - Variant A: {variant_a}")
    print(f"  - Variant B: {variant_b}")
    print("  (Running simulated test...)")

    print("\n✅ Optimization framework demo complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Demo for LangGraph Multi-Agent System"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/videos/sample.mp4",
        help="Path to video file",
    )
    parser.add_argument(
        "--demo",
        type=str,
        choices=["all", "agents", "coordinator", "langgraph", "evaluation", "optimization"],
        default="all",
        help="Which demo to run",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🏸 Badminton Footwork Analyzer")
    print("  📡 LangGraph Multi-Agent System Demo")
    print("=" * 60)

    video_path = args.video

    # Run selected demos
    if args.demo in ["all", "agents"]:
        demo_individual_agents(video_path)

    if args.demo in ["all", "coordinator"]:
        demo_coordinator_agent(video_path)

    if args.demo in ["all", "langgraph"]:
        demo_langgraph_workflow(video_path)

    if args.demo in ["all", "evaluation"]:
        demo_evaluation_framework()

    if args.demo in ["all", "optimization"]:
        demo_optimization()

    print_section("Demo Complete!")
    print("\n📚 For more information, see:")
    print("  - src/agents/ - Agent implementations")
    print("  - src/graph/ - LangGraph workflow")
    print("  - src/evaluation/ - Evaluation framework")
    print("  - tests/ - Test suites")


if __name__ == "__main__":
    main()
