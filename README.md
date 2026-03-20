# 🏸 Badminton Footwork Analyzer - Multi-Agent System

基于 LangGraph 的多智能体协同系统，用于羽毛球步法分析，采用 Eval-driven DevOps 模式。

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph Multi-Agent System              │
├───────────────────────────────────────────────���─────────────┤
│                                                             │
│  ┌───────────────┐     ┌──────────────────┐                │
│  │CoordinatorAgent│◄───►│   StateGraph     │                │
│  │ (Orchestrator) │     │   (LangGraph)    │                │
│  └───────┬───────┘     └────────┬─────────┘                │
│          │                      │                           │
│          ▼                      ▼                           │
│  ┌─────────────────┐   ┌──────────────────┐                │
│  │ DetectionAgent  │◄─►│ ValidationAgent  │◄──┐            │
│  │ (Pose Tracking) │   │(Quality Control) │   │            │
│  └────────┬────────┘   └────────┬─────────┘   │            │
│           │                     │             │            │
│           ▼                     ▼             │            │
│  ┌─────────────────┐                          │            │
│  │ AnalysisAgent   │◄─────────────────────────┘            │
│  │ (Footwork/Shot) │                                       │
│  └────────┬────────┘                                       │
│           │                                                │
│           ▼                                                │
│  ┌─────────────────┐     ┌──────────────────┐              │
│  │ EvaluationAgent │     │VisualizationAgent│              │
│  │  (Efficiency)   │     │(Heatmaps/Trajs)  │              │
│  └─────────────────┘     └──────────────────┘              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## 🤖 智能体定义

| Agent | 职责 | 输入 | 输出 |
|-------|------|------|------|
| **DetectionAgent** | 姿态检测和追踪 | video_frames | poses, detection_stats |
| **AnalysisAgent** | 步法和击球分析 | poses | footwork_events, metrics |
| **EvaluationAgent** | 效率评估和对比 | footwork_metrics | efficiency_score, recommendations |
| **VisualizationAgent** | 结果可视化 | analysis results | heatmaps, trajectories |
| **ValidationAgent** | 质量控制和验证 | all results | quality_scores, requires_review |
| **CoordinatorAgent** | 任务协调和流程管理 | initial state | final results |

## 📊 Eval-driven DevOps

### 评估指标体系

**Agent 级别指标：**
| Agent | Metric | Threshold |
|-------|--------|-----------|
| DetectionAgent | detection_rate | ≥0.80 |
| DetectionAgent | pose_accuracy | ≥0.85 |
| AnalysisAgent | event_detection_f1 | ≥0.75 |
| EvaluationAgent | score_consistency | ≥0.85 |

**系统级别指标：**
| Metric | Threshold |
|--------|-----------|
| latency_p95 | ≤5.0s |
| throughput | ≥30 fps |
| end_to_end_accuracy | ≥0.80 |

### 工作流程

```
评估 → 检测回归 → 自动优化 → A/B测试 → 部署
  ▲                                    │
  └────────────────────────────────────┘
```

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/harrysong/badminton_footwork_analyzer_MA.git
cd badminton_footwork_analyzer_MA

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行演示
python demo_multi_agent.py

# 运行 Eval-driven DevOps 演示
python eval_driven_devops_demo.py
```

## 📁 项目结构

```
badminton_footwork_analyzer_MA/
├── agents/                    # 多智能体实现
│   ├── __init__.py
│   ├── base_agent.py         # 基类
│   ├── detection_agent.py    # 姿态检测
│   ├── analysis_agent.py     # 步法分析
│   ├── evaluation_agent.py   # 效率评估
│   ├── visualization_agent.py# 可视化
│   ├── validation_agent.py   # 质量验证
│   └── coordinator_agent.py  # 协调器
│
├── graph/                     # LangGraph 工作流
│   ├── __init__.py
│   ├── state.py              # 状态定义
│   └── graph_builder.py      # 图构建器
│
├── evaluation/                # Eval-driven 框架
│   ├── __init__.py
│   ├── metrics.py            # 指标定义
│   ├── benchmarks.py         # 基准数据集
│   ├── evaluators.py         # 评估器
│   └── optimizer.py          # 迭代优化
│
├── configs/                   # 配置文件
│   └── evaluation.yaml
│
├── demo_multi_agent.py        # 多智能体演示
├── eval_driven_devops_demo.py # Eval-driven DevOps 演示
└── requirements.txt
```

## 🧪 测试

```bash
# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 运行完整评估
python -m evaluation.evaluators
```

## 📚 依赖

- **langgraph** - 状态图工作流
- **langchain-core** - LangChain 核心组件
- **optuna** - 超参数优化
- **mlflow** - 实验追踪
- **mediapipe** - 姿态检测
- **opencv-python** - 视频处理
- **streamlit** - Web 界面

## 📖 参考

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [Optuna](https://optuna.readthedocs.io/)

## 📄 License

MIT License
