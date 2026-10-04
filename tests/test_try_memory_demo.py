"""Regression coverage for the runnable demo's checks and error cleanup."""
from pathlib import Path
import runpy

import nsn
import pytest


def demo():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/try_memory.py"))


def test_offline_demo_checks_real_memory_and_closes_runtime():
    module = demo()
    result = module["run_demo"]()
    assert result["recall"] == result["after_restart"] == module["TOKEN"]
    assert len(result["memory_checks"]) == 5
    assert nsn.get_default_runtime() is None


def test_demo_propagates_backend_errors_and_closes_runtime():
    def broken_model(prompt):
        raise ValueError("backend unavailable")
    with pytest.raises(ValueError, match="backend unavailable"):
        demo()["run_demo"](broken_model)
    assert nsn.get_default_runtime() is None
