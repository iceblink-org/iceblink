"""Trusted text-only vLLM/NNsight integration. No client-supplied trace code."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
import secrets
import sqlite3
import threading
import time
import uuid

import numpy as np

from ..adapters import OwnedExecutor, RunGuard, RunStopped
from ..core import Compatibility, Limits, Mode, identity
from ..live import emit_tensor, emit_vector
from ..sensors import InputContext, Span, Unavailable, capture_plan, prefill_graph, select_prefill_tensor
from ..service import Client, ServiceError
from .mappings import MAPPINGS


@dataclass(frozen=True)
class EngineConfig:
    checkpoint: str
    revision: str
    model_path: str
    mapping: str = "qwen3_5_moe.vllm.residual.v1"
    precision: str = "bfloat16"
    layers: tuple[int, ...] = (11, 19, 27)
    max_context: int = 2048
    max_concurrency: int = 2
    gpu_memory_utilization: float = 0.85
    enable_thinking: bool = False
    language_model_only: bool = True

    def __post_init__(self):
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("Pin an immutable checkpoint revision")
        if self.mapping not in MAPPINGS or self.precision != "bfloat16":
            raise ValueError("Unsupported mapping or precision")
        if not 1 <= len(self.layers) <= 8 or len(set(self.layers)) != len(self.layers) or min(self.layers) < 0:
            raise ValueError("Invalid selected layers")
        if not 64 <= self.max_context <= 8192 or not 1 <= self.max_concurrency <= 16:
            raise ValueError("Unsupported serving budget")

    @classmethod
    def from_json(cls, path):
        with open(path) as stream:
            return cls(**json.load(stream))


def render_input(tokenizer, messages, tools=None, *, enable_thinking=False):
    """Use the exact rendered tokens, and verify optional message provenance.

    Marker rendering is a side calculation. A span is accepted only if removing
    the markers reproduces the original prompt byte-for-byte. Markers are never
    sent to the model. Token indices are obtained from the effective encoding.
    """
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": enable_thinking}
    if tools:
        kwargs["tools"] = tools
    clean = [{k: v for k, v in message.items() if k in {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"}}
             for message in messages]
    rendered = tokenizer.apply_chat_template(clean, **kwargs)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = tuple(encoded["input_ids"])
    marked, markers = [dict(m) for m in clean], []
    prefix = uuid.uuid4().hex
    for index, message in enumerate(marked):
        role, text = message["role"], message.get("content")
        if role not in {"user", "tool"} or not isinstance(text, str) or not text.strip():
            continue
        a, b = f"ICEBLINKBEGIN{prefix}{index}", f"ICEBLINKEND{prefix}{index}"
        markers.append((a, b, role, message.get("tool_call_id")))
        message["content"] = a + text + b
    spans = []
    if markers:
        modified = tokenizer.apply_chat_template(marked, **kwargs)
        ranges = []
        for a, b, role, tool in markers:
            start = modified.find(a)
            end = modified.find(b, start+len(a))
            if start < 0 or end < 0:
                ranges = []
                break
            modified = modified[:start] + modified[start+len(a):end] + modified[end+len(b):]
            ranges.append((start, end-len(a), role, tool))
        if modified == rendered:
            for start, end, role, tool in ranges:
                positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start]
                if positions:
                    spans.append(Span(positions[0], positions[-1]+1, role, tool))
    return InputContext(ids, tuple(spans), rendered)


def emit_prefill(pair, input_tokens, sites, destination, attribution):
    """Executes inside the NNsight worker, with only this request's rows."""
    for name, positions, mean in sites:
        try:
            started_ns = time.monotonic_ns()
            value = select_prefill_tensor(pair, input_tokens, positions, mean)
            emit_tensor(destination, attribution, name, value, started_ns)
        except Unavailable:
            return


class CaptureEngine:
    """Single copy of the actor. Lab and serving share the same worker operation."""
    def __init__(self, config: EngineConfig):
        from nnsight.modeling.vllm import VLLM

        self.config = config
        self.mapping = MAPPINGS[config.mapping]
        self.graph = prefill_graph(tuple(config.layers))
        self.model = VLLM(config.model_path, dispatch=True, mode="async", revision=config.revision,
                         dtype=config.precision, max_model_len=config.max_context,
                         max_num_batched_tokens=config.max_context*config.max_concurrency,
                         max_num_seqs=config.max_concurrency, gpu_memory_utilization=config.gpu_memory_utilization,
                         enable_prefix_caching=False, enable_chunked_prefill=False, disable_log_stats=True,
                         language_model_only=config.language_model_only)
        self.engine = self.model.vllm_entrypoint
        actual_arch = self.engine.model_config.hf_config.architectures
        if self.mapping.architecture not in actual_arch:
            raise ValueError("Configured sensor mapping does not match loaded architecture")
        self.layers = self.mapping.layers(self.model)
        if max(config.layers) >= len(self.layers):
            raise ValueError("Requested layer outside mapped actor")
        self.tokenizer = self.model.tokenizer
        backend = self.tokenizer.backend_tokenizer.to_str()
        self.tokenizer_id = hashlib.sha256(backend.encode()).hexdigest()
        self.template_id = identity({"template": self.tokenizer.chat_template, "enable_thinking": config.enable_thinking})
        self.available = {"text"} | {f"residual.{l}.{s}" for l in config.layers for s in ("last_input", "input_mean", "tool_mean")}
        self.slots = asyncio.Semaphore(config.max_concurrency)

    def compatibility(self, sensor):
        if sensor not in self.available:
            raise ValueError("Sensor not supported by engine")
        spec = self.graph.nodes[sensor]
        fingerprint = identity({"sensor": asdict(spec), "mapping": self.mapping.identity if sensor != "text" else "rendered_text.v1",
                                "language_model_only": self.config.language_model_only})
        return Compatibility(self.config.checkpoint, self.config.revision, self.config.precision,
                             self.tokenizer_id, self.template_id, fingerprint)

    def render(self, messages, tools=None):
        context = render_input(self.tokenizer, messages, tools, enable_thinking=self.config.enable_thinking)
        if len(context.token_ids) >= self.config.max_context:
            raise ValueError("Rendered context exceeds actor budget")
        return context

    def trace(self, context, plan, *, destination=None, attribution=None, max_tokens=256, stop=None, minimum=0):
        # Static configuration chooses trusted sites. Trace globals are installed
        # by the provider, and no prompt can pick an Envoy or execute trace code.
        if len(context.token_ids)+max_tokens > self.config.max_context:
            raise ValueError("Input plus generation exceeds context budget")
        taps = tuple((self.layers[layer], sites) for layer, sites in plan)
        input_tokens = len(context.token_ids)
        token_ids = list(context.token_ids)
        options = dict(max_tokens=max_tokens, temperature=0.0, min_tokens=minimum)
        if stop:
            options["stop"] = stop
        if destination is None:
            # Explicit research extraction: selected values only, finished output
            # is acceptable for the lab, never the runtime's detection path.
            with self.model.trace(token_ids, **options) as tracer:
                from iceblink.sensors import select_prefill_tensor
                features = dict().save()
                for layer, sites in taps:
                    pair = layer.output
                    for name, positions, mean in sites:
                        features[name] = select_prefill_tensor(pair, input_tokens, positions, mean)
        else:
            readiness_layer = self.layers[self.config.layers[0]]
            with self.model.trace(token_ids, **options) as tracer:
                from iceblink.integrations.engine import emit_prefill
                from iceblink.live import emit_vector
                for layer, sites in taps:
                    pair = layer.output
                    emit_prefill(pair, input_tokens, sites, destination, attribution)
                if not taps:
                    # Read an actual prefill location so the text comparison is
                    # scheduled at prefill, before the first model token is emitted.
                    pair = readiness_layer.output
                if pair[0].shape[0] == input_tokens:
                    emit_vector(destination, attribution, "__prefill_ready__", [0.0])
        return tracer.backend

    async def extract(self, context, sensors):
        plan = capture_plan(context, tuple(sensors), self.graph)
        async with self.slots:
            backend = self.trace(context, plan, max_tokens=1)
            final = await backend
        if final.prompt_token_ids != list(context.token_ids):
            raise RuntimeError("Engine changed effective input token IDs")
        return {key: value.float().cpu().numpy().copy() for key, value in final.saves["features"].items()}

    def close(self):
        self.engine.shutdown()


class MonitoredRun:
    def __init__(self, engine, client, journal, run_id, capability, sensors, executor, limits):
        self.engine, self.client, self.journal = engine, client, journal
        self.run_id, self.capability, self.sensors = run_id, capability, tuple(sensors)
        self.executor = executor
        self.active = {}
        self.lock = threading.RLock()
        self.loop = asyncio.get_running_loop()
        self.warning_gate = asyncio.Lock()
        self.guard = RunGuard(journal, run_id, abort_model=self.abort_model, executor=executor, limits=limits)
        self.last_request_id = None

    @classmethod
    async def create(cls, engine, state_dir, journal, policy, mode, *, owned_tools=False, limits=Limits()):
        client = Client(state_dir)
        run_id, capability = await asyncio.to_thread(client.register, policy, mode,
            {"model_abort": True, "model_gate": True, "tool_cancel": owned_tools, "sensors": sorted(engine.available),
             "compatibility": {s: asdict(engine.compatibility(s)) for s in engine.available}})
        requirements = await asyncio.to_thread(client.call, "policy", run_id=run_id, capability=capability)
        executor = OwnedExecutor(journal, run_id) if owned_tools else None
        return cls(engine, client, journal, run_id, capability, requirements["sensors"], executor, limits)

    def abort_model(self):
        with self.lock:
            for state in self.active.values():
                if not state["cancel_requested"]:
                    state["cancel_requested"] = True
                    self.loop.call_soon_threadsafe(state["task"].cancel)
            return not self.active

    async def generate(self, messages, tools=None, *, max_tokens=256, stop=None, minimum=0):
        kwargs = dict(max_tokens=max_tokens, stop=stop, minimum=minimum)
        if self.guard.mode >= Mode.b:
            # One call at a time per warning-capable run makes the next-call
            # handoff idempotent. Independent runs still batch together.
            async with self.warning_gate:
                return await self._generate(messages, tools, **kwargs)
        return await self._generate(messages, tools, **kwargs)

    async def _generate(self, messages, tools, *, max_tokens, stop, minimum):
        async with self.engine.slots:
            self.guard.ensure_model_allowed()
            try:
                warnings = self.guard.pending_warnings()
            except (sqlite3.Error, OSError):
                if self.guard.mode >= Mode.c:
                    raise RunStopped("Guarded warning state unavailable") from None
                warnings = []
            messages = [dict(m) for m in messages]
            if warnings:
                warning_text = "\n".join(text for _, text in warnings)
                if messages and messages[0]["role"] == "system":
                    messages[0]["content"] += "\n\n" + warning_text
                else:
                    messages.insert(0, {"role": "system", "content": warning_text})
            context = self.engine.render(messages, tools)
            plan = capture_plan(context, tuple(s for s in self.sensors if s in self.engine.available), self.engine.graph)
            execution_id, capability = uuid.uuid4().hex, secrets.token_hex(32)
            attribution = {"execution_id": execution_id, "capability": capability}
            backend = self.engine.trace(context, plan, destination=self.client.observations, attribution=attribution,
                                        max_tokens=max_tokens, stop=stop, minimum=minimum)
            request_id = backend._request_id  # NNsight 0.8.2 adapter boundary.
            self.last_request_id = request_id
            registered = False
            stream = None
            start = time.monotonic_ns()
            result = None
            latest_tool = next((s.tool_call for s in reversed(context.spans) if s.tool_call), None)
            with self.lock:
                self.guard.ensure_model_allowed()
                self.active[request_id] = {"task": asyncio.current_task(), "cancel_requested": False}
            try:
                try:
                    await asyncio.to_thread(self.client.call, "admit", run_id=self.run_id, capability=self.capability,
                         request_id=request_id, execution_id=execution_id, text=context.text, token_ids=context.token_ids,
                         observation_capability=capability, tool_call_id=latest_tool,
                         spans=[asdict(s) for s in context.spans],
                         compatibility={s: asdict(self.engine.compatibility(s)) for s in self.sensors if s in self.engine.available})
                    registered = True
                except (OSError, ServiceError, ValueError):
                    self.event("coverage_lost", run_id=self.run_id, request_id=request_id,
                                       component="engine", status="unavailable", payload={"reason": "admission_unavailable"})
                    if self.guard.mode >= Mode.c:
                        self.journal.stop(self.run_id, self.guard.mode, source="engine_health", request_id=request_id)
                        raise RunStopped("Guarded model admission unavailable") from None
                self.guard.ensure_model_allowed()
                stream = backend.__aiter__()
                first = None
                async for output in stream:
                    if output.request_id != request_id or output.prompt_token_ids != list(context.token_ids):
                        raise RuntimeError("Effective token or request attribution mismatch")
                    if first is None:
                        first = time.monotonic_ns()
                        if warnings:
                            try:
                                self.guard.delivered([wid for wid, _ in warnings], request_id)
                            except (sqlite3.Error, OSError):
                                if self.guard.mode >= Mode.c:
                                    raise RunStopped("Guarded delivery state unavailable") from None
                    result = output
                if result is None or not result.finished:
                    raise RunStopped("Generation aborted")
                self.event("generation_finished", run_id=self.run_id, request_id=request_id, tool_call_id=latest_tool,
                    component="engine", payload={"ttft_ns": first-start, "generation_ns": time.monotonic_ns()-start,
                    "input_tokens": len(context.token_ids), "output_tokens": len(result.outputs[0].token_ids), "monitored": registered})
                return result.outputs[0].text
            except asyncio.CancelledError:
                # Cancel the vLLM request and close the NNsight stream before
                # the watchdog can acknowledge that this model call stopped.
                await self.engine.engine.abort(request_id)
                self.event("generation_aborted", run_id=self.run_id, request_id=request_id,
                           tool_call_id=latest_tool, component="engine")
                raise RunStopped("Model generation stopped") from None
            finally:
                if stream is not None:
                    await stream.aclose()
                with self.lock:
                    self.active.pop(request_id, None)

    async def finish(self):
        try:
            self.guard.flush_warning_receipts()
            await asyncio.to_thread(self.client.call, "end", run_id=self.run_id, capability=self.capability)
            # Keep the watchdog alive through the bounded drain/cancellation.
            deadline = time.monotonic()+self.guard.limits.drain_seconds+0.2
            while time.monotonic() < deadline:
                rows = self.journal.rows("SELECT kind FROM events WHERE run_id=? AND kind='run_finished' LIMIT 1", (self.run_id,))
                pending = self.journal.rows("SELECT id FROM actions WHERE run_id=? AND status='pending' LIMIT 1", (self.run_id,))
                if rows and not pending:
                    break
                await asyncio.sleep(0.03)
        except (OSError, sqlite3.Error, ValueError, ServiceError):
            self.guard.monitoring_lost_at_finish()
            self.event("run_finish_unavailable", run_id=self.run_id, component="engine", status="incomplete")
        finally:
            self.guard.close()

    def event(self, *args, **kwargs):
        try:
            return self.journal.event(*args, **kwargs)
        except (OSError, sqlite3.Error):
            if self.guard.mode >= Mode.c:
                self.guard.local_stop = True
                raise RunStopped("Guarded journal unavailable") from None
