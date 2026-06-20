"""GPU utilization sampling during COLMAP / training runs (nvidia-smi)."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GpuSample:
    utilization_gpu: float | None
    utilization_mem: float | None
    memory_used_mb: float | None
    memory_total_mb: float | None
    temperature_c: float | None


@dataclass
class GpuRunStats:
    gpu_available: bool
    device_requested: str
    gpu_name: str | None = None
    monitor_duration_sec: float = 0.0
    gpu_active_duration_sec: float = 0.0
    gpu_active_ratio: float = 0.0
    utilization_gpu_avg: float | None = None
    utilization_gpu_max: float | None = None
    utilization_mem_avg: float | None = None
    utilization_mem_max: float | None = None
    memory_used_mb_avg: float | None = None
    memory_used_mb_max: float | None = None
    memory_total_mb: float | None = None
    temperature_c_max: float | None = None
    samples_count: int = 0
    sample_interval_sec: float = 1.0
    active_util_threshold: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _parse_nvidia_smi_line(line: str) -> tuple[str, GpuSample] | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6:
        return None

    def _float(value: str) -> float | None:
        value = value.strip()
        if not value or value in {"N/A", "[N/A]"}:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    name = parts[0]
    sample = GpuSample(
        utilization_gpu=_float(parts[1]),
        utilization_mem=_float(parts[2]),
        memory_used_mb=_float(parts[3]),
        memory_total_mb=_float(parts[4]),
        temperature_c=_float(parts[5]),
    )
    return name, sample


def query_gpu_sample() -> tuple[str, GpuSample] | None:
    if not nvidia_smi_available():
        return None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=gpu_name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return _parse_nvidia_smi_line(proc.stdout.strip().splitlines()[0])


def summarize_gpu_samples(
    samples: list[GpuSample],
    *,
    gpu_name: str | None,
    device_requested: str,
    monitor_duration_sec: float,
    sample_interval_sec: float,
    active_util_threshold: float,
) -> GpuRunStats:
    if not samples:
        return GpuRunStats(
            gpu_available=nvidia_smi_available(),
            device_requested=device_requested,
            gpu_name=gpu_name,
            monitor_duration_sec=round(monitor_duration_sec, 3),
            sample_interval_sec=sample_interval_sec,
            active_util_threshold=active_util_threshold,
        )

    def _avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 2)

    utils = [s.utilization_gpu for s in samples if s.utilization_gpu is not None]
    mem_utils = [s.utilization_mem for s in samples if s.utilization_mem is not None]
    mem_used = [s.memory_used_mb for s in samples if s.memory_used_mb is not None]
    mem_total = next((s.memory_total_mb for s in reversed(samples) if s.memory_total_mb is not None), None)
    temps = [s.temperature_c for s in samples if s.temperature_c is not None]

    active_samples = sum(1 for s in samples if (s.utilization_gpu or 0) >= active_util_threshold)
    active_duration = active_samples * sample_interval_sec
    duration = monitor_duration_sec if monitor_duration_sec > 0 else len(samples) * sample_interval_sec

    return GpuRunStats(
        gpu_available=True,
        device_requested=device_requested,
        gpu_name=gpu_name,
        monitor_duration_sec=round(duration, 3),
        gpu_active_duration_sec=round(active_duration, 3),
        gpu_active_ratio=round(active_duration / duration, 4) if duration > 0 else 0.0,
        utilization_gpu_avg=_avg(utils) if utils else None,
        utilization_gpu_max=round(max(utils), 2) if utils else None,
        utilization_mem_avg=_avg(mem_utils) if mem_utils else None,
        utilization_mem_max=round(max(mem_utils), 2) if mem_utils else None,
        memory_used_mb_avg=_avg(mem_used) if mem_used else None,
        memory_used_mb_max=round(max(mem_used), 2) if mem_used else None,
        memory_total_mb=mem_total,
        temperature_c_max=round(max(temps), 2) if temps else None,
        samples_count=len(samples),
        sample_interval_sec=sample_interval_sec,
        active_util_threshold=active_util_threshold,
    )


class GpuMonitor:
    """Background GPU sampler; use as context manager."""

    def __init__(
        self,
        *,
        device_requested: str = "auto",
        sample_interval_sec: float = 1.0,
        active_util_threshold: float = 5.0,
        enabled: bool = True,
    ) -> None:
        self.device_requested = device_requested
        self.sample_interval_sec = sample_interval_sec
        self.active_util_threshold = active_util_threshold
        self.enabled = enabled and nvidia_smi_available()
        self._samples: list[GpuSample] = []
        self._gpu_name: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def __enter__(self) -> GpuMonitor:
        if not self.enabled:
            return self
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=self.sample_interval_sec * 3)

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = query_gpu_sample()
            if sample:
                self._gpu_name, gpu_sample = sample
                self._samples.append(gpu_sample)
            self._stop.wait(self.sample_interval_sec)

    def collect_stats(self) -> GpuRunStats:
        duration = time.perf_counter() - self._started_at if self._started_at else 0.0
        if not self.enabled:
            return GpuRunStats(
                gpu_available=nvidia_smi_available(),
                device_requested=self.device_requested,
                monitor_duration_sec=round(duration, 3),
                sample_interval_sec=self.sample_interval_sec,
                active_util_threshold=self.active_util_threshold,
            )
        return summarize_gpu_samples(
            self._samples,
            gpu_name=self._gpu_name,
            device_requested=self.device_requested,
            monitor_duration_sec=duration,
            sample_interval_sec=self.sample_interval_sec,
            active_util_threshold=self.active_util_threshold,
        )
