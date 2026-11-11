"""Trusted detector interface and shared, dependency-light prediction code.

Manifests select implementations from a fixed registry, never Python modules.
The lab imports this exact implementation for validation and exported inference.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Protocol

import numpy as np
from safetensors.numpy import load_file, save_file

from .core import Compatibility, Score, identity


class Detector(Protocol):
    name: str
    version: str
    sensors: tuple[str, ...]
    readiness: str
    scope: str

    def evaluate(self, inputs: dict, evidence: tuple[str, ...]) -> Score: ...


def text_features(text: str, width: int = 1024) -> np.ndarray:
    """Fixed signed hashing of word unigrams/bigrams, L2 normalized.

    No labels, filenames, example IDs or provenance flags enter this function.
    Quoted attack examples can trigger false positives; evaluation reports them.
    """
    words = re.findall(r"[a-z0-9_]+", text.lower())
    terms = words + [a + " " + b for a, b in zip(words, words[1:])]
    values = np.zeros(width, dtype=np.float32)
    for term in terms:
        hashed = hashlib.blake2b(term.encode(), digest_size=8).digest()
        index = int.from_bytes(hashed[:4], "little") % width
        values[index] += 1 if hashed[4] & 1 else -1
    norm = np.linalg.norm(values)
    return values / max(float(norm), 1e-12)


def predict_linear(vector, coefficient, bias, mean, scale) -> float:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != coefficient.shape or not np.isfinite(vector).all():
        raise ValueError("Feature dimensions or values incompatible")
    return float(np.dot((vector - mean) / scale, coefficient) + bias.reshape(-1)[0])


class LinearDetector:
    readiness = "input_prefill"
    scope = "request"

    def __init__(self, manifest, tensors):
        self.manifest = manifest
        self.name = manifest["name"]
        self.version = manifest["artifact_id"]
        self.sensors = tuple(manifest["sensors"])
        self.compatibility = Compatibility(**manifest["compatibility"])
        self.threshold = float(manifest["threshold"])
        self.tensors = tensors

    def evaluate(self, inputs, evidence=()):
        if any(sensor not in inputs or inputs[sensor] is None for sensor in self.sensors):
            return Score(self.name, self.version, evidence=evidence, reason="required_observation_missing")
        try:
            value = inputs[self.sensors[0]]
            if self.manifest["preprocessing"]["kind"] == "text_hash_v1":
                value = text_features(value, self.tensors["coefficient"].size)
            score = predict_linear(value, **self.tensors)
            return Score(self.name, self.version, value=score, status="evaluated", evidence=evidence)
        except (ValueError, TypeError, FloatingPointError):
            return Score(self.name, self.version, status="error", evidence=evidence, reason="invalid_features")


class KeywordBaseline:
    """Untrained comparison available before a lab artifact exists."""
    name = "text-keywords"
    version = "1"
    sensors = ("text",)
    readiness = "input_prefill"
    scope = "request"
    patterns = (r"ignore\s+(all\s+)?(previous|prior|system)\s+(instructions|rules)",
                r"(reveal|print|send|output)\s+(the\s+)?(secret|password|system prompt)",
                r"(new|override)\s+(system|developer)\s+(message|instruction)",
                r"disregard\s+(the\s+)?(task|rules|instructions)")

    def evaluate(self, inputs, evidence=()):
        if inputs.get("text") is None:
            return Score(self.name, self.version, evidence=evidence, reason="text_missing")
        value = float(sum(bool(re.search(pattern, inputs["text"], re.I)) for pattern in self.patterns))
        return Score(self.name, self.version, value=value, status="evaluated", evidence=evidence,
                     meaning="keyword_pattern_count")


def validate_manifest(manifest):
    required = {"schema", "name", "implementation", "sensors", "preprocessing", "compatibility",
                "threshold", "score_meaning", "weights_sha256", "artifact_id", "training"}
    if set(manifest) != required or manifest["schema"] != 1:
        raise ValueError("Unsupported detector manifest schema")
    # The rename does not change the predictor. Preserve compatibility with
    # recorded bundles without rewriting their manifests or artifact IDs.
    if manifest["implementation"] not in {"iceblink.linear.v1", "scintilla.linear.v1"}:
        raise ValueError("Unregistered prediction implementation")
    if manifest["score_meaning"] != "uncalibrated_margin":
        raise ValueError("Unsupported score meaning")
    if not isinstance(manifest["sensors"], list) or len(manifest["sensors"]) != 1:
        raise ValueError("Baseline linear detector requires one sensor")
    if manifest["preprocessing"] not in ({"kind": "standardize_v1"}, {"kind": "text_hash_v1"}):
        raise ValueError("Unknown preprocessing")
    if not np.isfinite(manifest["threshold"]):
        raise ValueError("Nonfinite threshold")
    Compatibility(**manifest["compatibility"])
    if identity({k: v for k, v in manifest.items() if k != "artifact_id"}) != manifest["artifact_id"]:
        raise ValueError("Artifact manifest identity mismatch")


def load_detector(directory: Path | str, actual: Compatibility | None = None):
    directory = Path(directory)
    manifest_path, weights_path = directory / "manifest.json", directory / "weights.safetensors"
    if manifest_path.stat().st_size > 128 * 1024 or weights_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Artifact exceeds size budget")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    if hashlib.sha256(weights_path.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Weight checksum mismatch")
    tensors = load_file(str(weights_path))
    if set(tensors) != {"coefficient", "bias", "mean", "scale"}:
        raise ValueError("Invalid tensor set")
    width = tensors["coefficient"].size
    if not 0 < width <= 16384 or tensors["bias"].shape != (1,):
        raise ValueError("Invalid coefficient width or bias")
    for name, array in tensors.items():
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("Invalid tensor dtype or values")
        if name != "bias" and array.shape != (width,):
            raise ValueError("Invalid tensor shape")
    if np.any(tensors["scale"] <= 0):
        raise ValueError("Invalid standardization scale")
    detector = LinearDetector(manifest, tensors)
    if actual is not None:
        detector.compatibility.require(actual)
    return detector


def export_detector(directory, *, name, sensor, compatibility, coefficient, bias, mean, scale,
                    threshold, training, text=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    weights = {k: np.asarray(v, dtype=np.float32).reshape(-1) for k, v in
               dict(coefficient=coefficient, bias=bias, mean=mean, scale=scale).items()}
    save_file(weights, str(directory / "weights.safetensors"))
    manifest = {"schema": 1, "name": name, "implementation": "iceblink.linear.v1", "sensors": [sensor],
                "preprocessing": {"kind": "text_hash_v1" if text else "standardize_v1"},
                "compatibility": asdict(compatibility), "threshold": float(threshold),
                "score_meaning": "uncalibrated_margin", "training": training,
                "weights_sha256": hashlib.sha256((directory / "weights.safetensors").read_bytes()).hexdigest()}
    manifest["artifact_id"] = identity(manifest)
    validate_manifest(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return load_detector(directory, compatibility)
