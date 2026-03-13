"""Regression tests for CPU/CUDA/XLA runtime helpers."""

import torch
import torch.nn as nn

from cognitive_llm.evaluation.benchmark import BenchmarkRunner
from cognitive_llm.training import device as device_utils


class FakeXM:
    def xla_device(self):
        return torch.device("xla:0")

    def is_master_ordinal(self):
        return True


class QuantizedLikeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def to(self, *args, **kwargs):
        raise ValueError("`.to` is not supported for 4-bit quantized models.")


def test_resolve_device_auto_prefers_xla_when_available(monkeypatch):
    monkeypatch.setattr(device_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_utils, "xm", FakeXM())

    device = device_utils.resolve_device("auto")

    assert str(device) == "xla:0"


def test_resolve_device_accepts_tpu_alias(monkeypatch):
    monkeypatch.setattr(device_utils, "xm", FakeXM())

    device = device_utils.resolve_device("tpu")

    assert str(device) == "xla:0"


def test_move_model_to_device_keeps_quantized_models_in_place():
    model = QuantizedLikeModule()

    moved = device_utils.move_model_to_device(model, "cuda")

    assert moved is model


def test_move_batch_to_device_handles_nested_structures():
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "extra": [torch.tensor([3]), {"labels": torch.tensor([4])}],
    }

    moved = device_utils.move_batch_to_device(batch, "cpu")

    assert moved["input_ids"].device.type == "cpu"
    assert moved["extra"][0].device.type == "cpu"
    assert moved["extra"][1]["labels"].device.type == "cpu"


def test_benchmark_command_uses_requested_device():
    command = BenchmarkRunner.get_lm_eval_command(
        "repo/model",
        tasks="gsm8k",
        device="cpu",
        batch_size=4,
    )

    assert "--device cpu" in command
    assert "--batch_size 4" in command
    assert "--device cuda" not in command
