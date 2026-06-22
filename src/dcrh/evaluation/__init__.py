"""Dataset adapters and answer verification helpers."""

from ..config import DataConfig
from .benchmarks import BenchmarkExample, build_benchmark_example
from .data import Example, iter_examples
from .verifier import verify

__all__ = [
    "BenchmarkExample",
    "DataConfig",
    "Example",
    "build_benchmark_example",
    "iter_examples",
    "verify",
]
