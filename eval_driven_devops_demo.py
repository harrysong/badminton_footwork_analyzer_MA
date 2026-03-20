#!/usr/bin/env python3
"""
Eval-driven DevOps 验证演示

这个脚本演示了完整的 Eval-driven DevOps 流程：
1. 定义评估指标和基线
2. 运行评估流水线
3. 检测性能回归
4. 自动/半自动优化
5. A/B 测试验证
6. 持续监控和报告

Usage:
    source venv/bin/activate
    python eval_driven_devops_demo.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation.metrics import AgentMetrics, SystemMetrics, EvaluationMetric, EvaluationResult
from evaluation.evaluators import EvalDrivenTestPipeline, AgentEvaluator, SystemEvaluator
from evaluation.benchmarks import BenchmarkDataset, BenchmarkSample, create_sample_benchmark_dataset
from evaluation.optimizer import IterativeOptimizer, ABTestRunner, ParameterTuner


# ============================================================
# 1. 评估配置和基线定义
# ============================================================

@dataclass
class EvalConfig:
    """评估配置"""
    # 质量阈值
    min_detection_rate: float = 0.80
    min_confidence: float = 0.70
    min_efficiency_score: float = 60.0
    max_latency_p95: float = 5.0  # seconds
    min_throughput: float = 15.0  # fps

    # 回归检测
    regression_threshold: float = 0.05  # 5% regression allowed
    baseline_version: str = "v1.0.0"

    # 优化配置
    optimization_trials: int = 20
    ab_test_runs: int = 5
    significance_level: float = 0.05


@dataclass
class BaselineMetrics:
    """基线指标 - 用于回归检测"""
    version: str
    timestamp: str
    metrics: Dict[str, float]

    @classmethod
    def load(cls, path: Path) -> "BaselineMetrics":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ============================================================
# 2. Mock 评估函数（用于演示）
# ============================================================

def create_mock_evaluate_fn(base_score: float = 75.0):
    """创建模拟评估函数"""
    def evaluate(config: Dict[str, Any]) -> float:
        score = base_score

        # 模型复杂度影响
        complexity = config.get("model_complexity", 1)
        score += complexity * 2

        # 平滑影响
        if config.get("enable_smoothing", True):
            score += 3

        # 置信度阈值影响
        conf = config.get("min_detection_confidence", 0.5)
        if conf > 0.6:
            score -= (conf - 0.6) * 10  # 更高阈值可能降低召回率

        # 添加一些随机噪声
        import random
        score += random.uniform(-2, 2)

        return max(0, min(100, score))

    return evaluate


# ============================================================
# 3. Eval-driven DevOps 流程
# ============================================================

class EvalDrivenDevOps:
    """Eval-driven DevOps 验证系统"""

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.pipeline = EvalDrivenTestPipeline()
        self.baseline: Optional[BaselineMetrics] = None
        self.current_metrics: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []

    def load_baseline(self, path: Optional[Path] = None) -> BaselineMetrics:
        """加载基线指标"""
        if path and path.exists():
            self.baseline = BaselineMetrics.load(path)
            print(f"  ✅ 加载基线: {self.baseline.version}")
            return self.baseline

        # 创建默认基线
        self.baseline = BaselineMetrics(
            version=self.config.baseline_version,
            timestamp=datetime.now().isoformat(),
            metrics={
                "detection_rate": 0.85,
                "avg_confidence": 0.78,
                "efficiency_score": 75.0,
                "latency_p95": 3.5,
                "throughput": 25.0,
            }
        )
        print(f"  ✅ 创建默认基线: {self.baseline.version}")
        return self.baseline

    def run_evaluation(self) -> Dict[str, Any]:
        """运行完整评估"""
        print("\n📊 运行评估流水线...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "agent_metrics": {},
            "system_metrics": {},
            "quality_checks": {},
            "passed": True,
        }

        # 1. 评估 DetectionAgent
        print("  🔍 评估 DetectionAgent...")
        detection_evaluator = AgentEvaluator("DetectionAgent")
        detection_results = detection_evaluator.evaluate_detection({
            "detection_rate": 0.87,
            "avg_confidence": 0.79,
        })
        results["agent_metrics"]["detection"] = {
            k: v.to_dict() for k, v in detection_results.items()
        }

        # 2. 评估 AnalysisAgent
        print("  📈 评估 AnalysisAgent...")
        analysis_evaluator = AgentEvaluator("AnalysisAgent")
        # Mock metrics object
        mock_metrics = type('MockMetrics', (), {
            'total_steps': 50,
            'path_efficiency': 0.75,
            'avg_speed': 2.5,
            'events': [1, 2, 3]
        })()
        analysis_results = analysis_evaluator.evaluate_analysis(mock_metrics)
        results["agent_metrics"]["analysis"] = {
            k: v.to_dict() for k, v in analysis_results.items()
        }

        # 3. 评估 EvaluationAgent
        print("  🎯 评估 EvaluationAgent...")
        eval_evaluator = AgentEvaluator("EvaluationAgent")
        mock_score = type('MockScore', (), {'overall': 78.0})()
        eval_results = eval_evaluator.evaluate_evaluation(mock_score, ground_truth_score=75.0)
        results["agent_metrics"]["evaluation"] = {
            k: v.to_dict() for k, v in eval_results.items()
        }

        # 4. 系统指标评估
        print("  ⚙️  评估系统指标...")
        system_evaluator = SystemEvaluator()
        latency_result = system_evaluator.evaluate_latency(3.2)
        throughput_result = system_evaluator.evaluate_throughput(28.0)

        results["system_metrics"] = {
            "latency_p95": latency_result.to_dict(),
            "throughput": throughput_result.to_dict(),
        }

        # 5. 质量检查
        print("  ✅ 运行质量检查...")
        quality_checks = self._run_quality_checks(results)
        results["quality_checks"] = quality_checks
        results["passed"] = all(quality_checks.values())

        # 保存当前指标
        self.current_metrics = {
            "detection_rate": 0.87,
            "avg_confidence": 0.79,
            "efficiency_score": 78.0,
            "latency_p95": 3.2,
            "throughput": 28.0,
        }

        return results

    def _run_quality_checks(self, results: Dict) -> Dict[str, bool]:
        """运行质量检查"""
        checks = {}

        # 检查检测率
        detection_rate = 0.87  # mock value
        checks["detection_rate"] = detection_rate >= self.config.min_detection_rate

        # 检查置信度
        confidence = 0.79  # mock value
        checks["confidence"] = confidence >= self.config.min_confidence

        # 检查效率分数
        efficiency_score = 78.0  # mock value
        checks["efficiency_score"] = efficiency_score >= self.config.min_efficiency_score

        # 检查延迟
        latency = 3.2  # mock value
        checks["latency"] = latency <= self.config.max_latency_p95

        # 检查吞吐量
        throughput = 28.0  # mock value
        checks["throughput"] = throughput >= self.config.min_throughput

        return checks

    def detect_regression(self) -> Dict[str, Any]:
        """检测性能回归"""
        print("\n🔍 检测性能回归...")

        if not self.baseline:
            print("  ⚠️  没有基线，无法检测回归")
            return {"regression_detected": False, "details": {}}

        regressions = {}
        baseline_metrics = self.baseline.metrics

        for metric_name, current_value in self.current_metrics.items():
            if metric_name not in baseline_metrics:
                continue

            baseline_value = baseline_metrics[metric_name]

            # 对于延迟，越低越好
            if metric_name == "latency_p95":
                change = (current_value - baseline_value) / baseline_value
                is_regression = current_value > baseline_value * (1 + self.config.regression_threshold)
            else:
                # 对于其他指标，越高越好
                change = (baseline_value - current_value) / baseline_value
                is_regression = current_value < baseline_value * (1 - self.config.regression_threshold)

            regressions[metric_name] = {
                "baseline": baseline_value,
                "current": current_value,
                "change_percent": change * 100,
                "is_regression": is_regression,
            }

            if is_regression:
                print(f"  ⚠️  回归检测: {metric_name} ({baseline_value:.2f} -> {current_value:.2f})")

        regression_detected = any(r["is_regression"] for r in regressions.values())

        if not regression_detected:
            print("  ✅ 未检测到性能回归")

        return {
            "regression_detected": regression_detected,
            "details": regressions,
        }

    def optimize(self, evaluate_fn) -> Dict[str, Any]:
        """运行优化"""
        print("\n🔧 运行自动优化...")

        tuner = ParameterTuner(evaluate_fn)

        # Grid Search
        print("  📐 Grid Search...")
        param_grid = {
            "model_complexity": [0, 1, 2],
            "enable_smoothing": [True, False],
            "min_detection_confidence": [0.4, 0.5, 0.6],
        }

        best_params, best_score = tuner.grid_search(param_grid)
        print(f"    Best params: {best_params}")
        print(f"    Best score: {best_score:.2f}")

        return {
            "method": "grid_search",
            "best_params": best_params,
            "best_score": best_score,
            "improvement": best_score - 75.0,  # vs baseline
        }

    def run_ab_test(self, evaluate_fn, variant_a: Dict, variant_b: Dict) -> Dict[str, Any]:
        """运行 A/B 测试"""
        print("\n🧪 运行 A/B 测试...")

        runner = ABTestRunner(evaluate_fn)
        result = runner.run_ab_test(
            variant_a_config=variant_a,
            variant_b_config=variant_b,
            n_runs=self.config.ab_test_runs,
        )

        print(f"  Variant A score: {result.variant_a_score:.2f}")
        print(f"  Variant B score: {result.variant_b_score:.2f}")
        print(f"  Improvement: {result.improvement:.2f}%")
        print(f"  Significant: {result.statistical_significance['significant']}")
        print(f"  Winner: {result.winner or 'None'}")

        return {
            "variant_a_score": result.variant_a_score,
            "variant_b_score": result.variant_b_score,
            "improvement_percent": result.improvement,
            "significant": result.statistical_significance["significant"],
            "winner": result.winner,
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成评估报告"""
        report = []
        report.append("\n" + "=" * 60)
        report.append("  📋 Eval-driven DevOps 评估报告")
        report.append("=" * 60)

        # 质量检查结果
        report.append("\n✅ 质量检查:")
        for check, passed in results.get("quality_checks", {}).items():
            status = "✅" if passed else "❌"
            report.append(f"  {status} {check}: {'通过' if passed else '失败'}")

        # 指标摘要
        report.append("\n📊 关键指标:")
        for metric, value in self.current_metrics.items():
            report.append(f"  • {metric}: {value:.2f}")

        # 最终状态
        overall_passed = results.get("passed", False)
        report.append(f"\n{'✅' if overall_passed else '❌'} 整体状态: {'通过' if overall_passed else '失败'}")

        return "\n".join(report)


# ============================================================
# 4. CI/CD 集成示例
# ============================================================

def ci_cd_pipeline():
    """模拟 CI/CD 流水线中的评估步骤"""
    print("\n" + "=" * 60)
    print("  🔄 CI/CD Pipeline Evaluation")
    print("=" * 60)

    config = EvalConfig()
    devops = EvalDrivenDevOps(config)

    # Step 1: 加载基线
    print("\n📋 Step 1: 加载基线指标")
    devops.load_baseline()

    # Step 2: 运行评估
    print("\n📊 Step 2: 运行评估")
    results = devops.run_evaluation()

    # Step 3: 检测回归
    print("\n🔍 Step 3: 检测性能回归")
    regression = devops.detect_regression()

    # Step 4: 决策
    print("\n🎯 Step 4: CI/CD 决策")

    if regression["regression_detected"]:
        print("  ❌ 检测到性能回归，阻止部署！")
        print("  📝 建议检查最近的代码变更")

        # 可以在这里触发优化
        evaluate_fn = create_mock_evaluate_fn()
        optimization = devops.optimize(evaluate_fn)
        print(f"  🔧 建议参数: {optimization['best_params']}")

        return False
    elif results["passed"]:
        print("  ✅ 所有检查通过，可以部署！")

        # 更新基线
        print("  📝 更新基线指标...")
        # devops.baseline.save(Path("baselines/latest.json"))

        return True
    else:
        print("  ⚠️  质量检查未通过，需要人工审核")
        return False


# ============================================================
# 5. 主函数
# ============================================================

def main():
    print("=" * 60)
    print("  🏸 Eval-driven DevOps 验证演示")
    print("  📊 Badminton Footwork Analyzer")
    print("=" * 60)

    # Demo 1: 完整评估流程
    print("\n" + "=" * 60)
    print("  Demo 1: 完整评估流程")
    print("=" * 60)

    devops = EvalDrivenDevOps()
    devops.load_baseline()
    results = devops.run_evaluation()
    regression = devops.detect_regression()
    print(devops.generate_report(results))

    # Demo 2: 自动优化
    print("\n" + "=" * 60)
    print("  Demo 2: 自动优化")
    print("=" * 60)

    evaluate_fn = create_mock_evaluate_fn(base_score=72.0)
    optimization = devops.optimize(evaluate_fn)

    # Demo 3: A/B 测试
    print("\n" + "=" * 60)
    print("  Demo 3: A/B 测试")
    print("=" * 60)

    variant_a = {"model_complexity": 1, "enable_smoothing": True}
    variant_b = {"model_complexity": 2, "enable_smoothing": True}
    ab_result = devops.run_ab_test(evaluate_fn, variant_a, variant_b)

    # Demo 4: CI/CD 集成
    print("\n" + "=" * 60)
    print("  Demo 4: CI/CD 集成示例")
    print("=" * 60)

    ci_cd_pipeline()

    # 总结
    print("\n" + "=" * 60)
    print("  📚 Eval-driven DevOps 验证完成")
    print("=" * 60)
    print("""
Eval-driven DevOps 核心要素:

1. ✅ 评估指标体系
   - Agent 级别指标 (detection_rate, accuracy, etc.)
   - 系统级别指标 (latency, throughput)
   - 质量阈值定义

2. ✅ 自动化测试流水线
   - 单元测试
   - 集成测试
   - 端到端测试
   - 质量门禁

3. ✅ 回归检测
   - 基线对比
   - 变化检测
   - 自动告警

4. ✅ 迭代优化
   - Grid Search
   - Bayesian Optimization (Optuna)
   - A/B Testing

5. ✅ CI/CD 集成
   - Pre-commit hooks
   - PR 检查
   - 部署门禁

下一步:
- 集成到 GitHub Actions / GitLab CI
- 添加 MLflow 实验追踪
- 配置 Slack/Email 告警
    """)


if __name__ == "__main__":
    main()
