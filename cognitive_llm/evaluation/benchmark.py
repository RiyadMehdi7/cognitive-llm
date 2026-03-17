from __future__ import annotations

"""
Benchmark evaluation runner for GSM8K, ARC-Challenge, MATH, and HellaSwag.

Uses lm-evaluation-harness for standardized evaluation and provides
a programmatic interface for integration with the training loop.
"""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass

from cognitive_llm.training.device import resolve_device


@dataclass
class BenchmarkResult:
    """Stores benchmark evaluation results."""

    task: str
    metric: str
    score: float
    stderr: float | None = None
    n_samples: int | None = None


# Target benchmarks and metrics from spec
BENCHMARK_TASKS = {
    "gsm8k": {"metric": "exact_match,strict-match", "target_delta": "+5-10%"},
    "arc_challenge": {"metric": "acc_norm", "target_delta": "+3-8%"},
    "hellaswag": {"metric": "acc_norm", "target_delta": "maintained"},
    "mathqa": {"metric": "acc", "target_delta": "+3-6%"},
}


class BenchmarkRunner:
    """
    Runs standardized benchmarks using lm-evaluation-harness.

    Args:
        model_path: Path to saved model checkpoint or HuggingFace ID.
        output_dir: Directory to save results.
        device: Device to run on (default: 'auto').
        batch_size: Evaluation batch size (default: 8).
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./results",
        device: str = "auto",
        batch_size: int = 8,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = str(resolve_device(device))
        self.batch_size = batch_size

    def run_all(self, tasks: list[str] | None = None) -> list[BenchmarkResult]:
        """
        Run all benchmark tasks.

        Args:
            tasks: List of task names. Defaults to all defined tasks.

        Returns:
            List of BenchmarkResult objects.
        """
        if tasks is None:
            tasks = list(BENCHMARK_TASKS.keys())

        results = []
        for task in tasks:
            result = self.run_task(task)
            if result:
                results.extend(result)

        return results

    def run_task(self, task: str) -> list[BenchmarkResult]:
        """
        Run a single benchmark task using lm-eval CLI.

        Args:
            task: Task name (e.g., 'gsm8k', 'arc_challenge').

        Returns:
            List of BenchmarkResult objects.
        """
        output_path = self.output_dir / task
        output_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={self.model_path}",
            "--tasks", task,
            "--device", self.device,
            "--batch_size", str(self.batch_size),
            "--output_path", str(output_path),
            "--log_samples",
        ]

        print(f"Running benchmark: {task}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )
            if result.returncode != 0:
                print(f"  Error running {task}: {result.stderr}")
                return []
        except subprocess.TimeoutExpired:
            print(f"  Timeout running {task}")
            return []
        except FileNotFoundError:
            print("  lm_eval CLI not found. Install with: pip install lm-eval")
            return []

        return self._parse_results(task, output_path)

    def _parse_results(self, task: str, output_path: Path) -> list[BenchmarkResult]:
        """Parse lm-eval output JSON files."""
        results = []
        json_files = list(output_path.glob("**/*.json"))

        for jf in json_files:
            try:
                data = json.loads(jf.read_text())
                if "results" in data:
                    for task_name, metrics in data["results"].items():
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                results.append(
                                    BenchmarkResult(
                                        task=task_name,
                                        metric=metric_name,
                                        score=float(value),
                                    )
                                )
            except (json.JSONDecodeError, KeyError):
                continue

        return results

    def print_results(self, results: list[BenchmarkResult]) -> None:
        """Print results in a formatted table."""
        print("\n" + "=" * 60)
        print(f"{'Task':<20} {'Metric':<20} {'Score':<10}")
        print("=" * 60)
        for r in results:
            print(f"{r.task:<20} {r.metric:<20} {r.score:<10.4f}")
        print("=" * 60)

    @staticmethod
    def get_lm_eval_command(
        model_path: str,
        tasks: str = "gsm8k,arc_challenge,hellaswag,mathqa",
        device: str = "auto",
        batch_size: int = 8,
    ) -> str:
        """Generate the lm-eval CLI command string."""
        return (
            f"lm_eval --model hf "
            f"--model_args pretrained={model_path} "
            f"--tasks {tasks} "
            f"--device {resolve_device(device)} "
            f"--batch_size {batch_size} "
            f"--output_path ./results/"
        )
