"""Request-scoped sensor DAG. Shared dependencies compute once per evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from .core import identity


class Unavailable(Exception):
    """A requested observation does not exist or cannot be attributed."""


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    role: str
    tool_call: str | None = None


@dataclass
class InputContext:
    token_ids: tuple[int, ...]
    spans: tuple[Span, ...] = ()
    text: str | None = None
    layers: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        for s in self.spans:
            if not 0 <= s.start < s.end <= len(self.token_ids):
                raise ValueError("Span lies outside effective input")


@dataclass(frozen=True)
class Sensor:
    name: str
    operation: str
    dependencies: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    version: str = "1"

    @property
    def fingerprint(self):
        return identity(asdict(self))


class SensorGraph:
    def __init__(self, sensors: list[Sensor], operations: dict[str, Callable] | None = None):
        self.nodes = {s.name: s for s in sensors}
        if len(self.nodes) != len(sensors):
            raise ValueError("Duplicate sensor")
        self.operations = {**BUILTINS, **(operations or {})}
        visiting, visited = set(), set()

        def visit(name):
            if name in visiting:
                raise ValueError("Cyclic sensor dependency")
            if name in visited:
                return
            if name not in self.nodes:
                raise ValueError("Unknown dependency")
            visiting.add(name)
            node = self.nodes[name]
            if node.operation not in self.operations:
                raise ValueError("Unregistered sensor operation")
            for dep in node.dependencies:
                visit(dep)
            visiting.remove(name)
            visited.add(name)

        for name in self.nodes:
            visit(name)

    def evaluate(self, context: InputContext, requested: tuple[str, ...]):
        cache = {}

        def get(name):
            if name not in cache:
                if name not in self.nodes:
                    raise Unavailable("sensor_not_registered")
                node = self.nodes[name]
                deps = [get(dep) for dep in node.dependencies]
                cache[name] = self.operations[node.operation](context, deps, node.parameters)
            return cache[name]

        return {name: get(name) for name in requested}


def _tokens(ctx, deps, params):
    if not ctx.token_ids:
        raise Unavailable("empty_input")
    return ctx.token_ids


def _spans(ctx, deps, params):
    spans = tuple(s for s in ctx.spans if s.role == params.get("role", "tool"))
    if not spans:
        raise Unavailable("message_provenance_unavailable")
    return spans


def _last(ctx, deps, params):
    tokens = deps[0] if deps else ctx.token_ids
    if not tokens:
        raise Unavailable("empty_input")
    return len(tokens) - 1


def _layer(ctx, deps, params):
    value = ctx.layers.get(params["layer"])
    if value is None or value.ndim != 2 or value.shape[0] != len(ctx.token_ids):
        raise Unavailable("layer_missing_or_incomplete")
    return value


def _select(ctx, deps, params):
    layer, position = deps
    return np.array(layer[position], dtype=np.float32, copy=True)


def _mean(ctx, deps, params):
    layer, spans = deps
    positions = sorted({i for span in spans for i in range(span.start, span.end)})
    if not positions:
        raise Unavailable("empty_span")
    return np.array(layer[positions].mean(axis=0), dtype=np.float32, copy=True)


def _input_mean(ctx, deps, params):
    return np.array(deps[0].mean(axis=0, dtype=np.float32), dtype=np.float32, copy=True)


def _text(ctx, deps, params):
    if ctx.text is None:
        raise Unavailable("text_missing")
    return ctx.text


BUILTINS = {"tokens": _tokens, "spans": _spans, "last_input": _last, "layer": _layer,
            "select": _select, "span_mean": _mean, "input_mean": _input_mean, "text": _text}


def last_input_sensor(layer: int):
    return Sensor(f"residual.{layer}.last_input", "select", (f"layer.{layer}", "last_input"))


def prefill_graph(layers: tuple[int, ...]):
    nodes = [Sensor("tokens", "tokens"), Sensor("last_input", "last_input", ("tokens",)),
             Sensor("tool_spans", "spans", ("tokens",)), Sensor("text", "text")]
    for layer in layers:
        nodes += [Sensor(f"layer.{layer}", "layer", parameters={"layer": layer}), last_input_sensor(layer),
                  Sensor(f"residual.{layer}.input_mean", "input_mean", (f"layer.{layer}",)),
                  Sensor(f"residual.{layer}.tool_mean", "span_mean", (f"layer.{layer}", "tool_spans"))]
    return SensorGraph(nodes)


def capture_plan(context: InputContext, sensors: tuple[str, ...], graph: SensorGraph):
    """Resolve a graph into selections in the worker's request-local prefill rows."""
    supported = {"last_input", "input_mean", "tool_mean"}
    need_spans = any(name.endswith(".tool_mean") for name in sensors)
    names = ("last_input", "tool_spans") if need_spans and any(s.role == "tool" for s in context.spans) else ("last_input",)
    metadata = graph.evaluate(context, names)
    positions = tuple(sorted({i for span in metadata.get("tool_spans", ()) for i in range(span.start, span.end)}))
    layers = {}
    for name in sensors:
        if name == "text":
            continue
        if name not in graph.nodes:
            raise Unavailable("sensor_not_registered")
        _, layer, selector = name.split(".")
        if selector not in supported:
            raise Unavailable("selector_not_supported")
        if selector == "tool_mean" and not positions:
            continue  # Required missing observations are accounted for by coverage.
        selection = (metadata["last_input"],) if selector == "last_input" else positions if selector == "tool_mean" else None
        layers.setdefault(int(layer), []).append((name, selection, selector != "last_input"))
    return tuple(sorted((layer, tuple(sites)) for layer, sites in layers.items()))


def select_prefill_tensor(pair, input_tokens, positions, mean):
    """Worker operation for a vLLM branch/residual site; no retained full trace."""
    if not isinstance(pair, tuple) or len(pair) != 2 or pair[1] is None:
        raise Unavailable("unsupported_residual_representation")
    hidden, residual = pair
    if hidden.ndim != 2 or hidden.shape != residual.shape or hidden.shape[0] != input_tokens:
        raise Unavailable("partial_or_unattributed_prefill")
    if not mean:
        pos = positions[0]
        return (hidden[pos] + residual[pos]).float()
    if positions is None:
        return (hidden + residual).float().mean(dim=0)
    return (hidden[list(positions)] + residual[list(positions)]).float().mean(dim=0)
