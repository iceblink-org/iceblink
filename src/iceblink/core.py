"""Versioned identities and bounded runtime configuration."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Mode(IntEnum):
    a = 0
    b = 1
    c = 2
    d = 3

    @classmethod
    def parse(cls, value: str | int | Mode) -> Mode:
        return cls[value] if isinstance(value, str) else cls(value)


@dataclass(frozen=True)
class Compatibility:
    checkpoint: str
    revision: str
    precision: str
    tokenizer: str
    template: str
    sensor: str
    capture_version: str = "0.1.0"

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in asdict(self).values()):
            raise ValueError("Compatibility fields must be nonempty strings")

    def require(self, actual: Compatibility):
        if self != actual:
            fields = [k for k in asdict(self) if getattr(self, k) != getattr(actual, k)]
            raise ValueError("Detector compatibility mismatch: " + ", ".join(fields))


@dataclass(frozen=True)
class Limits:
    queue_count: int = 128
    queue_bytes: int = 16 * 1024 * 1024
    observation_bytes: int = 128 * 1024
    active_runs: int = 64
    active_requests: int = 128
    retained_bytes: int = 32 * 1024 * 1024
    context_tokens: int = 8192
    text_bytes: int = 512 * 1024
    heartbeat_seconds: float = 2.0
    processing_seconds: float = 10.0
    drain_seconds: float = 2.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("All limits must be positive and finite")


@dataclass(frozen=True)
class Rule:
    detector: str
    threshold: float
    mode: str

    def __post_init__(self):
        Mode.parse(self.mode)
        if not math.isfinite(self.threshold):
            raise ValueError("Nonfinite threshold")


@dataclass(frozen=True)
class Policy:
    name: str
    rules: tuple[Rule, ...]
    required: tuple[str, ...]
    readiness: str = "input_prefill"
    scope: str = "request"

    def __post_init__(self):
        if self.scope != "request" or self.readiness != "input_prefill":
            raise ValueError("Built-in policies operate on one input-prefill request")
        if not self.name or not self.required:
            raise ValueError("Policy must name required detectors")


@dataclass(frozen=True)
class Score:
    detector: str
    version: str
    target: str = "instruction_injection_attempt"
    value: float | None = None
    status: str = "unavailable"
    evidence: tuple[str, ...] = ()
    meaning: str = "uncalibrated_margin"
    reason: str | None = None

    def __post_init__(self):
        if self.status not in ("evaluated", "unavailable", "error"):
            raise ValueError("Invalid evaluation status")
        if self.status == "evaluated" and (self.value is None or not math.isfinite(self.value)):
            raise ValueError("Evaluated scores require finite values")
        if self.status != "evaluated" and self.value is not None:
            raise ValueError("Unavailable is not a risk score")
