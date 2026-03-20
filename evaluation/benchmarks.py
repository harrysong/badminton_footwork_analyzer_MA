"""Benchmark dataset structures and management."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np


@dataclass
class BenchmarkSample:
    """Single benchmark sample with ground truth annotations"""
    sample_id: str
    video_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Ground truth annotations
    pose_annotations: List[Dict] = field(default_factory=list)  # Per-frame pose keypoints
    footwork_events: List[Dict] = field(default_factory=list)  # Annotated events with timestamps
    shot_events: List[Dict] = field(default_factory=list)  # Annotated shots with types
    efficiency_scores: Dict[str, float] = field(default_factory=dict)  # Expert-provided scores

    # Difficulty level
    difficulty: str = "medium"  # "easy", "medium", "hard"

    # Video characteristics
    resolution: Tuple[int, int] = (1920, 1080)
    duration: float = 0.0  # seconds
    fps: float = 30.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSample":
        """Create from dictionary"""
        return cls(
            sample_id=data["sample_id"],
            video_path=data["video_path"],
            metadata=data.get("metadata", {}),
            pose_annotations=data.get("pose_annotations", []),
            footwork_events=data.get("footwork_events", []),
            shot_events=data.get("shot_events", []),
            efficiency_scores=data.get("efficiency_scores", {}),
            difficulty=data.get("difficulty", "medium"),
            resolution=tuple(data.get("resolution", [1920, 1080])),
            duration=data.get("duration", 0.0),
            fps=data.get("fps", 30.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "sample_id": self.sample_id,
            "video_path": self.video_path,
            "metadata": self.metadata,
            "pose_annotations": self.pose_annotations,
            "footwork_events": self.footwork_events,
            "shot_events": self.shot_events,
            "efficiency_scores": self.efficiency_scores,
            "difficulty": self.difficulty,
            "resolution": list(self.resolution),
            "duration": self.duration,
            "fps": self.fps,
        }


@dataclass
class BenchmarkDataset:
    """Benchmark dataset for evaluation"""
    name: str
    version: str
    samples: List[BenchmarkSample] = field(default_factory=list)

    # Dataset splits
    train_sample_ids: List[str] = field(default_factory=list)
    val_sample_ids: List[str] = field(default_factory=list)
    test_sample_ids: List[str] = field(default_factory=list)

    # Metadata
    total_duration: float = 0.0
    annotation_guidelines: str = ""
    annotator_agreement: float = 0.0

    # Reference statistics
    reference_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def train_samples(self) -> List[BenchmarkSample]:
        return [s for s in self.samples if s.sample_id in self.train_sample_ids]

    @property
    def val_samples(self) -> List[BenchmarkSample]:
        return [s for s in self.samples if s.sample_id in self.val_sample_ids]

    @property
    def test_samples(self) -> List[BenchmarkSample]:
        return [s for s in self.samples if s.sample_id in self.test_sample_ids]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkDataset":
        """Create from dictionary"""
        samples = [BenchmarkSample.from_dict(s) for s in data.get("samples", [])]
        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            samples=samples,
            train_sample_ids=data.get("train_sample_ids", []),
            val_sample_ids=data.get("val_sample_ids", []),
            test_sample_ids=data.get("test_sample_ids", []),
            total_duration=data.get("total_duration", 0.0),
            annotation_guidelines=data.get("annotation_guidelines", ""),
            annotator_agreement=data.get("annotator_agreement", 0.0),
            reference_stats=data.get("reference_stats", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "samples": [s.to_dict() for s in self.samples],
            "train_sample_ids": self.train_sample_ids,
            "val_sample_ids": self.val_sample_ids,
            "test_sample_ids": self.test_sample_ids,
            "total_duration": self.total_duration,
            "annotation_guidelines": self.annotation_guidelines,
            "annotator_agreement": self.annotator_agreement,
            "reference_stats": self.reference_stats,
        }


class BenchmarkManager:
    """Manager for benchmark datasets"""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("benchmarks")
        self.datasets: Dict[str, BenchmarkDataset] = {}

    def load_dataset(self, name: str) -> BenchmarkDataset:
        """Load a benchmark dataset"""
        dataset_path = self.data_dir / name

        # Try JSON first
        json_path = dataset_path / "metadata.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                dataset = BenchmarkDataset.from_dict(data)
                self.datasets[name] = dataset
                return dataset

        raise FileNotFoundError(f"Dataset not found: {name}")

    def save_dataset(self, dataset: BenchmarkDataset) -> None:
        """Save a benchmark dataset"""
        dataset_path = self.data_dir / dataset.name
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(dataset_path / "metadata.json", "w") as f:
            json.dump(dataset.to_dict(), f, indent=2)

        self.datasets[dataset.name] = dataset

    def create_gold_standard(
        self,
        samples: List[BenchmarkSample],
        expert_annotations: Dict[str, Any],
    ) -> BenchmarkDataset:
        """Create gold standard dataset from expert annotations"""
        efficiency_scores = expert_annotations.get("efficiency_scores", {})

        # Calculate reference statistics
        reference_stats = {
            "path_efficiency": {
                "mean": np.mean([s.get("path_efficiency", 0) for s in efficiency_scores.values()]),
                "std": np.std([s.get("path_efficiency", 0) for s in efficiency_scores.values()]),
            },
            "step_frequency": {
                "mean": np.mean([s.get("step_frequency", 0) for s in efficiency_scores.values()]),
                "std": np.std([s.get("step_frequency", 0) for s in efficiency_scores.values()]),
            },
            "response_time": {
                "mean": np.mean([s.get("response_time", 0) for s in efficiency_scores.values()]),
                "std": np.std([s.get("response_time", 0) for s in efficiency_scores.values()]),
            },
        }

        dataset = BenchmarkDataset(
            name="gold_standard",
            version="1.0",
            samples=samples,
            reference_stats=reference_stats,
        )

        return dataset

    def split_dataset(
        self,
        dataset: BenchmarkDataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> BenchmarkDataset:
        """Split dataset into train/val/test"""
        import random

        sample_ids = [s.sample_id for s in dataset.samples]
        random.shuffle(sample_ids)

        n = len(sample_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_ids = sample_ids[:n_train]
        val_ids = sample_ids[n_train:n_train + n_val]
        test_ids = sample_ids[n_train + n_val:]

        dataset.train_sample_ids = train_ids
        dataset.val_sample_ids = val_ids
        dataset.test_sample_ids = test_ids

        return dataset


def create_sample_benchmark_dataset() -> BenchmarkDataset:
    """Create a sample benchmark dataset for testing"""
    samples = [
        BenchmarkSample(
            sample_id="sample_001",
            video_path="data/videos/sample_001.mp4",
            difficulty="easy",
            efficiency_scores={
                "overall": 85.0,
                "movement_efficiency": 88.0,
                "response_time": 82.0,
                "court_coverage": 80.0,
            },
            duration=30.0,
        ),
        BenchmarkSample(
            sample_id="sample_002",
            video_path="data/videos/sample_002.mp4",
            difficulty="medium",
            efficiency_scores={
                "overall": 72.0,
                "movement_efficiency": 75.0,
                "response_time": 70.0,
                "court_coverage": 71.0,
            },
            duration=45.0,
        ),
        BenchmarkSample(
            sample_id="sample_003",
            video_path="data/videos/sample_003.mp4",
            difficulty="hard",
            efficiency_scores={
                "overall": 58.0,
                "movement_efficiency": 60.0,
                "response_time": 55.0,
                "court_coverage": 59.0,
            },
            duration=60.0,
        ),
    ]

    return BenchmarkDataset(
        name="sample_benchmark",
        version="1.0",
        samples=samples,
        train_sample_ids=["sample_001"],
        val_sample_ids=["sample_002"],
        test_sample_ids=["sample_003"],
    )
