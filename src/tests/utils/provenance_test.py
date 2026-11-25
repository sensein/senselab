"""Tests for the execution-environment record and per-step provenance."""

from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from senselab.utils import provenance
from senselab.utils.provenance import (
    ENVIRONMENT_KEYS,
    ENVIRONMENTS_KEY,
    PROCESSING_STEPS_KEY,
    cpu_info,
    environment,
    environment_fingerprint,
    record_step,
)

X86_CPUINFO = """processor\t: 0
vendor_id\t: AuthenticAMD
model name\t: AMD EPYC 7713   64-Core Processor
processor\t: 1
vendor_id\t: AuthenticAMD
model name\t: AMD EPYC 7713   64-Core Processor
"""
ARM_CPUINFO = """processor\t: 0
BogoMIPS\t: 50.00
CPU implementer\t: 0x41
CPU part\t: 0xd0c
"""


@pytest.fixture
def fake_cpuinfo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """Point `cpu_info` at a file the test writes, and clear its per-process caches around it."""

    def write(text: str) -> None:
        path = tmp_path / "cpuinfo"
        path.write_text(text)
        monkeypatch.setattr(provenance, "CPUINFO", str(path))
        provenance.cpu_info.cache_clear()
        provenance._static_environment.cache_clear()

    yield write
    provenance.cpu_info.cache_clear()
    provenance._static_environment.cache_clear()


def test_cpu_info_reads_the_x86_vendor_and_collapses_model_spacing(fake_cpuinfo: Any) -> None:  # noqa: ANN401
    """x86 reports ``vendor_id`` and a single-spaced model name."""
    fake_cpuinfo(X86_CPUINFO)
    assert cpu_info() == {"cpu_vendor": "AuthenticAMD", "cpu_model": "AMD EPYC 7713 64-Core Processor"}


def test_cpu_info_reads_the_arm_implementer(fake_cpuinfo: Any) -> None:  # noqa: ANN401
    """ARM reports its implementer code as the vendor; it has no model name line."""
    fake_cpuinfo(ARM_CPUINFO)
    assert cpu_info() == {"cpu_vendor": "0x41", "cpu_model": None}


def test_cpu_info_without_the_file_is_none(fake_cpuinfo: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN401
    """An unreadable cpuinfo (e.g. off Linux) gives None for both."""
    fake_cpuinfo("")
    monkeypatch.setattr(provenance, "CPUINFO", "/nonexistent/cpuinfo")
    provenance.cpu_info.cache_clear()
    assert cpu_info() == {"cpu_vendor": None, "cpu_model": None}


def test_environment_has_exactly_the_documented_keys(fake_cpuinfo: Any) -> None:  # noqa: ANN401
    """The record's keys are `ENVIRONMENT_KEYS`, in that order, and the CPU comes from cpuinfo."""
    fake_cpuinfo(X86_CPUINFO)
    record = environment()
    assert tuple(record) == ENVIRONMENT_KEYS
    assert record["cpu_vendor"] == "AuthenticAMD"
    assert record["torch_version"] is not None
    assert record["torch_cpu_capability"] == torch.backends.cpu.get_cpu_capability()


def test_environment_reads_the_thread_count_on_every_call() -> None:
    """A thread-count change between calls is reflected, not cached."""
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        assert environment()["torch_num_threads"] == 1
        torch.set_num_threads(2)
        assert environment()["torch_num_threads"] == 2
    finally:
        torch.set_num_threads(before)


def test_environment_does_not_initialize_cuda() -> None:
    """Reading the environment never initializes CUDA; GPU fields stay None until something else does."""
    if torch.cuda.is_initialized():
        pytest.skip("CUDA already initialized in this process")
    record = environment()
    assert not torch.cuda.is_initialized()
    assert record["gpu_name"] is None and record["cudnn_version"] is None


def test_fingerprint_is_stable_and_order_independent() -> None:
    """Same content, any key order, gives the same 16-hex fingerprint; a changed value changes it."""
    a: Dict[str, Any] = {"cpu_vendor": "GenuineIntel", "torch_version": "2.11.0"}
    b = {"torch_version": "2.11.0", "cpu_vendor": "GenuineIntel"}
    assert environment_fingerprint(a) == environment_fingerprint(b)
    assert len(environment_fingerprint(a)) == 16
    assert environment_fingerprint(a) != environment_fingerprint({**a, "cpu_vendor": "AuthenticAMD"})


def test_record_step_appends_and_keeps_each_environment_once() -> None:
    """Two steps in one environment give two entries and one environment record."""
    env = {"cpu_vendor": "GenuineIntel"}
    metadata: Dict[str, Any] = {}
    first = record_step(metadata, "a", {"x": 1}, env)
    second = record_step(metadata, "b", {"y": 2}, env)
    assert metadata[PROCESSING_STEPS_KEY] == [first, second]
    assert first == {"step": "a", "parameters": {"x": 1}, "environment": environment_fingerprint(env)}
    assert metadata[ENVIRONMENTS_KEY] == {environment_fingerprint(env): env}


def test_record_step_does_not_change_a_shallow_copied_source() -> None:
    """Stamping a shallow copy leaves the source's step list and environments as they were."""
    source: Dict[str, Any] = {}
    record_step(source, "a", {}, {"cpu_vendor": "GenuineIntel"})
    copy = dict(source)
    record_step(copy, "b", {}, {"cpu_vendor": "AuthenticAMD"})
    assert [s["step"] for s in source[PROCESSING_STEPS_KEY]] == ["a"]
    assert len(source[ENVIRONMENTS_KEY]) == 1
    assert [s["step"] for s in copy[PROCESSING_STEPS_KEY]] == ["a", "b"]
    assert len(copy[ENVIRONMENTS_KEY]) == 2


def test_record_step_defaults_to_the_current_environment() -> None:
    """Without an explicit record, the step names the current process's environment."""
    metadata: Dict[str, Any] = {}
    entry = record_step(metadata, "a", {})
    assert set(metadata[ENVIRONMENTS_KEY][entry["environment"]]) == set(ENVIRONMENT_KEYS)
