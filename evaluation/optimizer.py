"""Iterative optimizer for continuous improvement using Optuna."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional import - will work without it but with limited functionality
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Install with: pip install optuna")


@dataclass
class OptimizationResult:
    """Result of optimization iteration"""
    iteration: int
    metric_name: str
    before_value: float
    after_value: float
    improvement: float
    parameters_changed: Dict[str, Any]
    study_best_value: Optional[float] = None
    study_best_params: Optional[Dict[str, Any]] = None


@dataclass
class ABTestResult:
    """Result of A/B test"""
    variant_a_config: Dict[str, Any]
    variant_b_config: Dict[str, Any]
    variant_a_score: float
    variant_b_score: float
    improvement: float
    statistical_significance: Dict[str, Any]
    winner: Optional[str] = None


class IterativeOptimizer:
    """
    Iterative optimization loop for continuous improvement.

    Uses Bayesian optimization (Optuna) to find optimal hyperparameters.
    """

    def __init__(
        self,
        evaluate_func: Callable,
        optimizer_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize optimizer.

        Args:
            evaluate_func: Function to evaluate a configuration.
                          Should take config dict and return a score.
            optimizer_config: Configuration for the optimizer
        """
        self.evaluate_func = evaluate_func
        self.config = optimizer_config or {}

        self.history: List[OptimizationResult] = []
        self.study: Optional[object] = None
        self.current_score: Optional[float] = None

        # Optimization targets
        self.target_metric = self.config.get("target_metric", "score")
        self.direction = self.config.get("direction", "maximize")

    def run_optimization_cycle(
        self,
        n_trials: int = 50,
        n_startup_trials: int = 10,
    ) -> OptimizationResult:
        """
        Run one optimization cycle using Bayesian optimization.

        Args:
            n_trials: Number of optimization trials
            n_startup_trials: Number of random trials before optimization

        Returns:
            OptimizationResult with the best parameters found
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for optimization. Install with: pip install optuna")

        # Get baseline score
        before_score = self.evaluate_func(self.config)
        self.current_score = before_score

        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = self._sample_parameters(trial)

            # Merge with current config
            trial_config = {**self.config, **params}

            # Evaluate
            score = self.evaluate_func(trial_config)
            return score

        # Create and run study
        direction = "maximize" if self.direction == "maximize" else "minimize"
        self.study = optuna.create_study(direction=direction)
        self.study.optimize(objective, n_trials=n_trials, n_startup_trials=n_startup_trials)

        # Apply best parameters
        best_params = self.study.best_params
        after_score = self.study.best_value

        # Update config
        self.config.update(best_params)

        result = OptimizationResult(
            iteration=len(self.history),
            metric_name=self.target_metric,
            before_value=before_score,
            after_value=after_score,
            improvement=after_score - before_score,
            parameters_changed=best_params,
            study_best_value=after_score,
            study_best_params=best_params,
        )

        self.history.append(result)
        return result

    def _sample_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample parameters from trial."""
        params = {}

        # Detection parameters
        params["model_complexity"] = trial.suggest_int("model_complexity", 0, 2)
        params["min_detection_confidence"] = trial.suggest_float(
            "min_detection_confidence", 0.3, 0.7
        )

        # Smoothing parameters
        params["smoothing_window"] = trial.suggest_int("smoothing_window", 3, 15)

        # Analysis parameters
        params["jump_threshold"] = trial.suggest_float("jump_threshold", 5.0, 15.0)
        params["step_detection_sensitivity"] = trial.suggest_float(
            "step_detection_sensitivity", 0.5, 2.0
        )

        # Evaluation weights
        params["efficiency_weight_movement"] = trial.suggest_float(
            "efficiency_weight_movement", 0.1, 0.5
        )
        params["efficiency_weight_response"] = trial.suggest_float(
            "efficiency_weight_response", 0.1, 0.5
        )

        return params

    def get_best_parameters(self) -> Dict[str, Any]:
        """Get the best parameters found so far."""
        if self.study is None:
            return self.config

        return {**self.config, **self.study.best_params}

    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get the optimization history."""
        return self.history


class ABTestRunner:
    """Runner for A/B testing between configurations"""

    def __init__(
        self,
        evaluate_func: Callable,
        test_samples: Optional[List[str]] = None,
    ):
        """
        Initialize A/B test runner.

        Args:
            evaluate_func: Function to evaluate a configuration
            test_samples: Optional list of sample IDs to test
        """
        self.evaluate_func = evaluate_func
        self.test_samples = test_samples or []

    def run_ab_test(
        self,
        variant_a_config: Dict,
        variant_b_config: Dict,
        n_runs: int = 5,
    ) -> ABTestResult:
        """
        Run A/B test between two configurations.

        Args:
            variant_a_config: Configuration for variant A
            variant_b_config: Configuration for variant B
            n_runs: Number of runs for each variant

        Returns:
            ABTestResult with statistical analysis
        """
        # Run multiple evaluations for each variant
        scores_a = []
        scores_b = []

        for _ in range(n_runs):
            score_a = self.evaluate_func(variant_a_config)
            score_b = self.evaluate_func(variant_b_config)
            scores_a.append(score_a)
            scores_b.append(score_b)

        # Calculate statistics
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        std_a = np.std(scores_a)
        std_b = np.std(scores_b)

        # Statistical significance test (paired t-test)
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
            significant = p_value < 0.05
        except ImportError:
            # Fallback if scipy not available
            t_stat, p_value = 0.0, 1.0
            significant = False

        # Determine winner
        if mean_a > mean_b:
            winner = "A" if significant else None
            improvement = (mean_a - mean_b) / mean_b * 100 if mean_b > 0 else 0
        else:
            winner = "B" if significant else None
            improvement = (mean_b - mean_a) / mean_a * 100 if mean_a > 0 else 0

        return ABTestResult(
            variant_a_config=variant_a_config,
            variant_b_config=variant_b_config,
            variant_a_score=mean_a,
            variant_b_score=mean_b,
            improvement=improvement,
            statistical_significance={
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": significant,
                "std_a": float(std_a),
                "std_b": float(std_b),
            },
            winner=winner,
        )

    def run_multi_variant_test(
        self,
        variant_configs: Dict[str, Dict],
        n_runs: int = 5,
    ) -> Dict[str, ABTestResult]:
        """Run multi-variant test (A/B/C/...)."""
        results = {}
        variants = list(variant_configs.keys())

        # Compare each variant to the first one (control)
        control = variants[0]
        control_config = variant_configs[control]

        for variant in variants[1:]:
            variant_config = variant_configs[variant]
            result = self._run_pairwise_test(
                control_config, variant_config, control, variant, n_runs
            )
            results[f"{control}_vs_{variant}"] = result

        return results

    def _run_pairwise_test(
        self,
        config_a: Dict,
        config_b: Dict,
        name_a: str,
        name_b: str,
        n_runs: int,
    ) -> ABTestResult:
        """Run pairwise comparison."""
        scores_a = []
        scores_b = []

        for _ in range(n_runs):
            scores_a.append(self.evaluate_func(config_a))
            scores_b.append(self.evaluate_func(config_b))

        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)

        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
            significant = p_value < 0.05
        except ImportError:
            t_stat, p_value, significant = 0.0, 1.0, False

        return ABTestResult(
            variant_a_config=config_a,
            variant_b_config=config_b,
            variant_a_score=mean_a,
            variant_b_score=mean_b,
            improvement=(mean_b - mean_a) / mean_a * 100 if mean_a > 0 else 0,
            statistical_significance={
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": significant,
            },
            winner=name_b if significant and mean_b > mean_a else (name_a if significant else None),
        )


class ParameterTuner:
    """Simple parameter tuner without Optuna for basic use cases"""

    def __init__(self, evaluate_func: Callable):
        self.evaluate_func = evaluate_func
        self.best_score: float = float('-inf')
        self.best_params: Dict[str, Any] = {}

    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        base_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run grid search over parameter grid.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values to try
            base_config: Base configuration to start from

        Returns:
            Tuple of (best_params, best_score)
        """
        import itertools

        base_config = base_config or {}
        best_score = float('-inf')
        best_params = base_config.copy()

        # Generate all combinations
        keys = list(param_grid.keys())
        for values in itertools.product(*param_grid.values()):
            config = {**base_config, **dict(zip(keys, values))}
            score = self.evaluate_func(config)

            if score > best_score:
                best_score = score
                best_params = config.copy()

        self.best_score = best_score
        self.best_params = best_params
        return best_params, best_score

    def random_search(
        self,
        param_distributions: Dict[str, Any],
        n_iter: int = 20,
        base_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run random search over parameter distributions.

        Args:
            param_distributions: Dictionary mapping parameter names to (min, max) tuples
            n_iter: Number of random configurations to try
            base_config: Base configuration to start from

        Returns:
            Tuple of (best_params, best_score)
        """
        import random

        base_config = base_config or {}
        best_score = float('-inf')
        best_params = base_config.copy()

        for _ in range(n_iter):
            # Sample random values
            config = base_config.copy()
            for param, (min_val, max_val) in param_distributions.items():
                if isinstance(min_val, int):
                    config[param] = random.randint(min_val, max_val)
                else:
                    config[param] = random.uniform(min_val, max_val)

            score = self.evaluate_func(config)

            if score > best_score:
                best_score = score
                best_params = config.copy()

        self.best_score = best_score
        self.best_params = best_params
        return best_params, best_score
