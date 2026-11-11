"""Bounded worker-to-companion datagrams; no detector RPC on the inference hook.

Only trusted, installed instrumentation may call this interface. The socket must
live in an owner-only directory. A failed send is coverage loss, never a negative
score. Adapters register the required work before submitting to the model, so a
lost datagram is also caught by the independent progress deadline.
"""
from __future__ import annotations

import base64
import json
import socket
import time

import numpy as np

MAX_DATAGRAM = 128 * 1024


def encode_vector(vector) -> dict:
    value = np.array(vector, dtype="<f4", copy=True).reshape(-1)
    if value.nbytes > 64 * 1024 or not np.isfinite(value).all():
        raise ValueError("Invalid or oversized selected vector")
    return {"dtype": "float32", "width": value.size,
            "data": base64.b64encode(value.tobytes()).decode("ascii")}


def decode_vector(payload: dict) -> np.ndarray:
    width = payload["width"]
    if payload["dtype"] != "float32" or not isinstance(width, int) or not 0 < width <= 16384:
        raise ValueError("Invalid vector header")
    if len(payload["data"]) > 90_000:
        raise ValueError("Oversized vector")
    data = base64.b64decode(payload["data"], validate=True)
    if len(data) != width * 4:
        raise ValueError("Vector length mismatch")
    value = np.frombuffer(data, dtype="<f4").copy()
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite observation")
    return value


def emit_vector(path: str, attribution: dict, sensor: str, vector, capture_ns=None) -> bool:
    capture_ns = capture_ns if capture_ns is not None else time.monotonic_ns()
    encoded = encode_vector(vector)
    payload = json.dumps({"schema": 1, "attribution": attribution, "sensor": sensor,
                          "capture_ns": capture_ns, "worker_ns": time.monotonic_ns()-capture_ns, "vector": encoded},
                         separators=(",", ":"), allow_nan=False).encode()
    if len(payload) > MAX_DATAGRAM:
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.setblocking(False)
            return sock.sendto(payload, path) == len(payload)
    except OSError:
        return False


def emit_tensor(path: str, attribution: dict, sensor: str, tensor, capture_ns=None) -> bool:
    # Reduction precedes the host copy. Only a bounded selected vector survives
    # the next engine step; never retain a live engine tensor or full trace.
    if tensor.ndim != 1 or tensor.numel() > 16384:
        return False
    capture_ns = capture_ns if capture_ns is not None else time.monotonic_ns()
    return emit_vector(path, attribution, sensor, tensor.detach().float().cpu().numpy(), capture_ns)
