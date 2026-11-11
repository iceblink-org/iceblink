"""Explicit opt-in, bounded selected-feature recording for local replay."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid

from .live import encode_vector, decode_vector


class Recorder:
    def __init__(self, directory, *, max_count=256, max_bytes=64*1024*1024, retention_seconds=86400):
        self.path = Path(directory).resolve()
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        state = self.path.stat()
        if state.st_uid != os.getuid() or state.st_mode & 0o077:
            raise ValueError("Recording directory must be owner-only")
        if not 1 <= max_count <= 4096 or max_bytes <= 0 or retention_seconds <= 0:
            raise ValueError("Invalid recording limits")
        self.max_count, self.max_bytes, self.retention_seconds = max_count, max_bytes, retention_seconds

    def write(self, request, inputs):
        record = {"schema": 1, "request": request, "text": inputs.get("text"),
                  "features": {name: encode_vector(v) for name, v in inputs.items() if name != "text"}}
        raw = json.dumps(record, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        if len(raw) > self.max_bytes:
            return None
        files = sorted((p.stat().st_mtime, p.stat().st_size, p) for p in self.path.glob("record-*.json") if p.is_file())
        total = sum(size for _, size, _ in files)
        while files and (len(files) >= self.max_count or total+len(raw) > self.max_bytes
                         or files[0][0] < time.time()-self.retention_seconds):
            _, size, path = files.pop(0)
            path.unlink()
            total -= size
        rid = uuid.uuid4().hex
        tmp = self.path / ("." + rid + ".tmp")
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(raw)
        tmp.replace(self.path / f"record-{rid}.json")
        return rid


def read_record(path, max_bytes=1024*1024):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("Replay record exceeds byte limit")
    record = json.loads(path.read_text())
    if record["schema"] != 1:
        raise ValueError("Unsupported replay record")
    return record["request"], {"text": record["text"], **{k: decode_vector(v) for k, v in record["features"].items()}}
