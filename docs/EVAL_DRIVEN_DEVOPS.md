# Eval-driven DevOps 验证指南

## 概述

本项目实现了完整的 **Eval-driven DevOps** 模式，用于验证多智能体协同系统。

## 核心概念

### 1. 评估指标体系

```python
# 定义评估指标
from evaluation.metrics import AgentMetrics, SystemMetrics

# Agent 级别指标
detection_rate: float    # 姿态检测率
pose_accuracy: float     # 姿态���确度
event_detection_f1: float  # 事件检测 F1

# 系统级别指标
latency_p95: float       # 95 分位延迟
throughput: float        # 吞吐量 (fps)
end_to_end_accuracy: float  # 端到端准确度
```

### 2. 质量门禁 (Quality Gates)

```yaml
# configs/evaluation.yaml
agents:
  detection:
    min_detection_rate: 0.7
    min_confidence: 0.5

system:
  latency:
    target_p95_seconds: 5.0
  throughput:
    target_fps: 30.0
```

### 3. 回归检测

```python
# 对比基线指标检测性能回归
baseline = {
    "detection_rate": 0.85,
    "latency_p95": 3.5,
    "throughput": 25.0,
}

current = evaluate_system()

# 检测 5% 以上的性能下降
if current["detection_rate"] < baseline["detection_rate"] * 0.95:
    raise RegressionError("检测率下降超过阈值")
```

## 快速开始

### 本地验证

```bash
# 安装依赖
pip install -r requirements.txt

# 运行演示
python demo_multi_agent.py

# 运行 Eval-driven DevOps 验证
python eval_driven_devops_demo.py
```

### CI/CD 集成

项目已配置 GitHub Actions 工作流：

```yaml
# .github/workflows/eval_driven_ci.yml
jobs:
  evaluation:
    steps:
      - name: Run evaluation pipeline
        run: python eval_driven_devops_demo.py

      - name: Check for regression
        run: |
          if [ "$REGRESSION" = "true" ]; then
            echo "::error::检测到性能回归"
            exit 1
          fi
```

## 验证流程

### Step 1: 定义基线

```python
baseline = BaselineMetrics(
    version="v1.0.0",
    metrics={
        "detection_rate": 0.85,
        "efficiency_score": 75.0,
        "latency_p95": 3.5,
    }
)
```

### Step 2: 运行评估

```python
devops = EvalDrivenDevOps(config)
results = devops.run_evaluation()

# 检查结果
assert results["passed"], "质量检查未通过"
```

### Step 3: 检测回归

```python
regression = devops.detect_regression()
if regression["regression_detected"]:
    print("⚠️ 检测到性能回归")
    # 触发优化或阻止部署
```

### Step 4: 自动优化

```python
# Grid Search 优化
tuner = ParameterTuner(evaluate_fn)
best_params, best_score = tuner.grid_search(param_grid)

# A/B 测试验证
runner = ABTestRunner(evaluate_fn)
result = runner.run_ab_test(variant_a, variant_b)
```

## A/B 测试示例

```python
# 定义两个配置变体
variant_a = {"model_complexity": 1, "enable_smoothing": True}
variant_b = {"model_complexity": 2, "enable_smoothing": True}

# 运行 A/B 测试
result = runner.run_ab_test(variant_a, variant_b, n_runs=5)

print(f"Variant A: {result.variant_a_score:.2f}")
print(f"Variant B: {result.variant_b_score:.2f}")
print(f"Winner: {result.winner}")
print(f"Significant: {result.statistical_significance['significant']}")
```

## GitHub PR 评估报告

当 PR 创建时，自动生成评估报告：

```markdown
## 📊 Eval-driven DevOps 评估结果

### 质量检查
| 指标 | 状态 |
|------|------|
| Detection Rate | ✅ |
| Confidence | ✅ |
| Latency | ✅ |

### 系统指标
- Latency P95: 3.2s
- Throughput: 28 fps

### ✅ 所有检查通过，可以合并
```

## 持续改进循环

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Evaluate  │ ──► │   Analyze   │ ──► │   Optimize  │
│  (Metrics)  │     │ (Regression?)│     │  (Optuna)   │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                                        │
       └────────────────────────────────────────┘
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `evaluation/metrics.py` | 指标定义 |
| `evaluation/evaluators.py` | 评估器实现 |
| `evaluation/optimizer.py` | 优化器和 A/B 测试 |
| `eval_driven_devops_demo.py` | 完整演示脚本 |
| `.github/workflows/eval_driven_ci.yml` | CI/CD 配置 |

## 下一步

1. ✅ 框架已实现并验证
2. 🔜 连接真实视频数据测试
3. 🔜 配置 Slack/Email 告警
4. 🔜 添加 MLflow 实验追踪
