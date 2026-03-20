"""Evaluators for running agent and system evaluations."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import time
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    AgentMetrics, SystemMetrics, EvaluationMetric,
    EvaluationResult, AggregatedMetrics
)
from evaluation.benchmarks import BenchmarkSample, BenchmarkDataset


class AgentEvaluator:
    """Evaluator for individual agent performance"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.results: List[EvaluationResult] = []

    def evaluate_detection(
        self,
        actual_stats: Dict[str, float],
        ground_truth: Optional[Dict[str, float]] = None,
    ) -> Dict[str, EvaluationResult]:
        """Evaluate detection agent metrics"""
        results = {}

        # Detection rate
        detection_rate = actual_stats.get("detection_rate", 0.0)
        metric = AgentMetrics.DETECTION_RATE
        results["detection_rate"] = EvaluationResult(
            metric_name="detection_rate",
            actual_value=detection_rate,
            expected_value=metric.threshold,
            passed=detection_rate >= metric.threshold,
            score=min(1.0, detection_rate / metric.threshold) if metric.threshold > 0 else 0.0,
            weight=metric.weight,
        )

        # Average confidence
        avg_confidence = actual_stats.get("avg_confidence", 0.0)
        results["avg_confidence"] = EvaluationResult(
            metric_name="avg_confidence",
            actual_value=avg_confidence,
            expected_value=0.7,
            passed=avg_confidence >= 0.7,
            score=min(1.0, avg_confidence / 0.7),
            weight=0.3,
        )

        self.results.extend(results.values())
        return results

    def evaluate_analysis(
        self,
        actual_metrics: Any,
        ground_truth_events: Optional[List[Dict]] = None,
    ) -> Dict[str, EvaluationResult]:
        """Evaluate analysis agent metrics"""
        results = {}

        # Event count sanity check
        event_count = len(getattr(actual_metrics, "events", []))
        results["event_count"] = EvaluationResult(
            metric_name="event_count",
            actual_value=event_count,
            expected_value=0,
            passed=event_count > 0,
            score=min(1.0, event_count / 10),
            weight=0.3,
        )

        # Metric reasonableness
        total_steps = getattr(actual_metrics, "total_steps", 0)
        results["total_steps"] = EvaluationResult(
            metric_name="total_steps",
            actual_value=total_steps,
            expected_value=100,  # Reasonable range
            passed=0 < total_steps < 1000,
            score=1.0 if 0 < total_steps < 1000 else 0.0,
            weight=0.3,
        )

        self.results.extend(results.values())
        return results

    def evaluate_evaluation(
        self,
        actual_score: Any,
        ground_truth_score: Optional[float] = None,
    ) -> Dict[str, EvaluationResult]:
        """Evaluate evaluation agent metrics"""
        results = {}

        overall = getattr(actual_score, "overall", 0.0) if actual_score else 0.0

        # Score in valid range
        results["score_validity"] = EvaluationResult(
            metric_name="score_validity",
            actual_value=overall,
            expected_value=100.0,
            passed=0 <= overall <= 100,
            score=1.0 if 0 <= overall <= 100 else 0.0,
            weight=0.5,
        )

        # Score consistency (if ground truth available)
        if ground_truth_score is not None:
            score_diff = abs(overall - ground_truth_score)
            results["score_accuracy"] = EvaluationResult(
                metric_name="score_accuracy",
                actual_value=score_diff,
                expected_value=10.0,
                passed=score_diff <= 10.0,
                score=max(0, 1.0 - score_diff / 20),
                weight=0.5,
            )

        self.results.extend(results.values())
        return results

    def get_aggregated_results(self) -> AggregatedMetrics:
        """Get aggregated results"""
        if not self.results:
            return AggregatedMetrics(
                total_score=0.0,
                weighted_score=0.0,
                passed_count=0,
                failed_count=0,
                total_count=0,
            )

        total_score = sum(r.score for r in self.results)
        weighted_score = sum(r.score * r.weight for r in self.results)
        passed_count = sum(1 for r in self.results if r.passed)
        failed_count = len(self.results) - passed_count

        return AggregatedMetrics(
            total_score=total_score / len(self.results),
            weighted_score=weighted_score,
            passed_count=passed_count,
            failed_count=failed_count,
            total_count=len(self.results),
            results=self.results,
        )

    def reset(self):
        """Reset evaluator"""
        self.results = []


class SystemEvaluator:
    """Evaluator for end-to-end system performance"""

    def __init__(self):
        self.results: Dict[str, List[EvaluationResult]] = {}
        self.latencies: List[float] = []

    def evaluate_latency(
        self,
        latency: float,
        target_p95: float = 5.0,
    ) -> EvaluationResult:
        """Evaluate latency metric"""
        return EvaluationResult(
            metric_name="latency",
            actual_value=latency,
            expected_value=target_p95,
            passed=latency <= target_p95,
            score=max(0, 1.0 - latency / (target_p95 * 2)),
            weight=0.2,
        )

    def evaluate_throughput(
        self,
        fps: float,
        target_fps: float = 30.0,
    ) -> EvaluationResult:
        """Evaluate throughput metric"""
        return EvaluationResult(
            metric_name="throughput",
            actual_value=fps,
            expected_value=target_fps,
            passed=fps >= target_fps,
            score=min(1.0, fps / target_fps),
            weight=0.2,
        )

    def evaluate_accuracy(
        self,
        predicted: float,
        ground_truth: float,
        tolerance: float = 10.0,
    ) -> EvaluationResult:
        """Evaluate accuracy metric"""
        error = abs(predicted - ground_truth)
        passed = error <= tolerance

        return EvaluationResult(
            metric_name="accuracy",
            actual_value=error,
            expected_value=tolerance,
            passed=passed,
            score=max(0, 1.0 - error / (tolerance * 2)),
            weight=0.6,
        )

    def evaluate_end_to_end(
        self,
        result_state: Dict[str, Any],
        ground_truth: Optional[BenchmarkSample] = None,
    ) -> AggregatedMetrics:
        """Evaluate complete end-to-end performance"""
        results = []

        # Check if analysis completed
        if result_state.get("current_stage") != "completed":
            results.append(EvaluationResult(
                metric_name="completion",
                actual_value=0,
                expected_value=1,
                passed=False,
                score=0.0,
                weight=0.2,
            ))
        else:
            results.append(EvaluationResult(
                metric_name="completion",
                actual_value=1,
                expected_value=1,
                passed=True,
                score=1.0,
                weight=0.2,
            ))

        # Evaluate efficiency score accuracy if ground truth available
        if ground_truth and result_state.get("efficiency_score"):
            predicted_score = result_state["efficiency_score"].overall
            gt_score = ground_truth.efficiency_scores.get("overall", 0)

            accuracy_result = self.evaluate_accuracy(
                predicted_score, gt_score, tolerance=10.0
            )
            accuracy_result.metric_name = "efficiency_score_accuracy"
            accuracy_result.weight = 0.6
            results.append(accuracy_result)

        # Evaluate quality scores
        quality_scores = result_state.get("quality_scores", {})
        overall_quality = quality_scores.get("overall", 0.0)

        results.append(EvaluationResult(
            metric_name="quality_score",
            actual_value=overall_quality,
            expected_value=0.8,
            passed=overall_quality >= 0.8,
            score=overall_quality,
            weight=0.2,
        ))

        return AggregatedMetrics(
            total_score=sum(r.score for r in results) / len(results),
            weighted_score=sum(r.score * r.weight for r in results),
            passed_count=sum(1 for r in results if r.passed),
            failed_count=sum(1 for r in results if not r.passed),
            total_count=len(results),
            results=results,
        )

    def add_latency_sample(self, latency: float):
        """Add a latency sample for P95/P99 calculation"""
        self.latencies.append(latency)

    def get_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles"""
        if not self.latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)

        return {
            "p50": sorted_latencies[int(n * 0.5)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)],
        }


class EvalDrivenTestPipeline:
    """Automated testing pipeline for eval-driven design"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.agent_evaluators: Dict[str, AgentEvaluator] = {}
        self.system_evaluator = SystemEvaluator()
        self.results: Dict[str, Any] = {}

    def get_agent_evaluator(self, agent_name: str) -> AgentEvaluator:
        """Get or create agent evaluator"""
        if agent_name not in self.agent_evaluators:
            self.agent_evaluators[agent_name] = AgentEvaluator(agent_name)
        return self.agent_evaluators[agent_name]

    def run_agent_test(
        self,
        agent_name: str,
        test_func: Callable,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> AggregatedMetrics:
        """Run a test for a specific agent"""
        evaluator = self.get_agent_evaluator(agent_name)

        # Run the test and get results
        start_time = time.time()
        result = test_func()
        latency = time.time() - start_time

        # Add latency to system evaluator
        self.system_evaluator.add_latency_sample(latency)

        # Evaluate based on agent type
        if agent_name == "DetectionAgent":
            stats = result.get("detection_stats", {})
            evaluator.evaluate_detection(stats, ground_truth)
        elif agent_name == "AnalysisAgent":
            metrics = result.get("footwork_metrics")
            evaluator.evaluate_analysis(metrics, ground_truth)
        elif agent_name == "EvaluationAgent":
            score = result.get("efficiency_score")
            gt_score = ground_truth.get("efficiency_score") if ground_truth else None
            evaluator.evaluate_evaluation(score, gt_score)

        return evaluator.get_aggregated_results()

    def run_system_test(
        self,
        test_func: Callable,
        benchmark_sample: Optional[BenchmarkSample] = None,
    ) -> AggregatedMetrics:
        """Run end-to-end system test"""
        start_time = time.time()
        result = test_func()
        latency = time.time() - start_time

        self.system_evaluator.add_latency_sample(latency)

        return self.system_evaluator.evaluate_end_to_end(result, benchmark_sample)

    def run_benchmark(
        self,
        dataset: BenchmarkDataset,
        run_sample_func: Callable[[BenchmarkSample], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run evaluation on a benchmark dataset"""
        all_results = []
        latencies = []

        for sample in dataset.test_samples:
            start_time = time.time()

            try:
                result = run_sample_func(sample)
                latency = time.time() - start_time
                latencies.append(latency)

                # Evaluate result
                eval_result = self.system_evaluator.evaluate_end_to_end(
                    result, sample
                )
                all_results.append({
                    "sample_id": sample.sample_id,
                    "result": eval_result.to_dict(),
                    "latency": latency,
                    "success": True,
                })

            except Exception as e:
                latencies.append(time.time() - start_time)
                all_results.append({
                    "sample_id": sample.sample_id,
                    "error": str(e),
                    "success": False,
                })

        # Calculate aggregate statistics
        successful = [r for r in all_results if r.get("success")]
        failed = len(all_results) - len(successful)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_score = sum(
            r["result"]["weighted_score"] for r in successful
        ) / len(successful) if successful else 0

        # Get latency percentiles
        sorted_latencies = sorted(latencies)
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] if latencies else 0

        return {
            "dataset": dataset.name,
            "total_samples": len(dataset.test_samples),
            "successful": len(successful),
            "failed": failed,
            "success_rate": len(successful) / len(all_results) if all_results else 0,
            "average_latency": avg_latency,
            "latency_p95": p95,
            "average_score": avg_score,
            "sample_results": all_results,
        }

    def save_results(self, output_path: str):
        """Save evaluation results to file"""
        results = {
            "timestamp": time.time(),
            "config": self.config,
            "system_latencies": self.system_evaluator.get_latency_percentiles(),
            "agent_results": {
                name: eval.get_aggregated_results().to_dict()
                for name, eval in self.agent_evaluators.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations"""
        return {
            "agent_evaluations": {
                name: eval.get_aggregated_results().to_dict()
                for name, eval in self.agent_evaluators.items()
            },
            "system_latencies": self.system_evaluator.get_latency_percentiles(),
        }
