"""Milestone 1 test: real NNsight worker -> separate process before completion.

This validates transport and attribution, not a HuggingFace reference mapping.
Use --model explicitly for small development fixtures; they never certify the
approved model. No prompts or activations are persisted in the result record.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import tempfile
import time
import uuid

import numpy as np

from ..live import MAX_DATAGRAM, decode_vector
from .mappings import MAPPINGS


def receive(path, pipe, count):
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.bind(path)
        sock.settimeout(900)
        pipe.send({"ready": True, "pid": os.getpid()})
        for _ in range(count):
            message = json.loads(sock.recv(MAX_DATAGRAM))
            pipe.send({"received_ns": time.monotonic_ns(), "pid": os.getpid(), **message})


async def prove(args, path, pipe):
    from nnsight.modeling.vllm import VLLM

    options = dict(revision=args.revision, dtype="bfloat16", enable_prefix_caching=args.prefix_cache,
                   max_model_len=2048, max_num_seqs=2, gpu_memory_utilization=args.memory_fraction,
                   max_num_batched_tokens=2048, enable_chunked_prefill=args.prefix_cache,
                   disable_log_stats=True)
    mapping = MAPPINGS[args.mapping]
    if args.taps:
        options["taps"] = [f"{mapping.layer_path}.{args.layer}.output"]
    model = VLLM(args.model, dispatch=True, mode="async", **options)
    try:
        return await _prove_model(args, model, mapping, path, pipe)
    finally:
        model.vllm_entrypoint.shutdown()


async def _prove_model(args, model, mapping, path, pipe):
    layer = mapping.layers(model)[args.layer]
    prompts = ["Count from 1 to 200, one number per line.", "List the first 200 positive even numbers."]
    if args.repeat_prefix:
        prompts = ["This is a controlled test of input processing. " * 127 + "Now count from 1 to 200."] * 2

    def tokenize(prompt):
        return model.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                   add_generation_prompt=True, return_dict=False)

    cache_control = []
    if args.repeat_prefix:
        from vllm import SamplingParams
        prompt = {"prompt_token_ids": tokenize(prompts[0])}
        for _ in range(2):
            async for warmed in model.vllm_entrypoint.generate(prompt, SamplingParams(max_tokens=1, temperature=0), uuid.uuid4().hex):
                pass
            cache_control.append(warmed.num_cached_tokens)
        print("Ordinary repeated-prefix cached token counts:", cache_control, flush=True)

    async def one(index):
        token_ids = tokenize(prompts[index])
        execution = uuid.uuid4().hex
        attribution = {"run_id": f"proof-{index}", "execution_id": execution,
                       "input_tokens": len(token_ids), "position": len(token_ids) - 1}
        sensor = f"residual.{args.layer}.last_input"
        with model.trace(token_ids, max_tokens=args.output_tokens, min_tokens=args.output_tokens, temperature=0.0) as tracer:
            from iceblink.live import emit_tensor
            from iceblink.integrations.mappings import residual_after_block
            pair = layer.output
            if pair[0].shape[0] != len(token_ids):
                raise ValueError("Trace did not receive the complete effective prefill")
            vector = residual_after_block((pair[0][-1], pair[1][-1])).float()
            sent = emit_tensor(path, attribution, sensor, vector).save()
            reference = vector.clone().save()
        # NNsight assigns its actual engine request ID when the trace exits.
        # Record that ID before consuming the stream, linked by execution ID.
        request_id = tracer.backend._request_id
        started_ns = time.monotonic_ns()
        outputs = []
        async for output in tracer.backend:
            if output.request_id != request_id:
                raise AssertionError("Engine request attribution changed")
            if output.prompt_token_ids != token_ids:
                raise AssertionError("Effective input IDs differ from trace input")
            if not output.finished:
                outputs.append(time.monotonic_ns())
        finished_ns = time.monotonic_ns()
        if not output.saves["sent"]:
            raise AssertionError("Worker could not deliver observation")
        saved = output.saves["reference"].float().cpu().numpy()
        return {"run_id": attribution["run_id"], "execution_id": execution, "request_id": request_id,
                "input_token_hash": hashlib.sha256(np.array(token_ids, dtype="<i8").tobytes()).hexdigest(),
                "position": len(token_ids) - 1, "started_ns": started_ns, "finished_ns": finished_ns,
                "stream_steps": len(outputs), "last_nonfinal_ns": max(outputs), "reference": saved,
                "cached_input_tokens": output.num_cached_tokens}

    results = [await one(i) for i in range(2)] if args.repeat_prefix else await asyncio.gather(*(one(i) for i in range(args.concurrent)))
    received = {}
    for _ in results:
        if not pipe.poll(5):
            raise AssertionError("Companion did not receive the selected feature")
        record = pipe.recv()
        received[record["attribution"]["execution_id"]] = record
    for result in results:
        observation = received[result["execution_id"]]
        value = decode_vector(observation["vector"])
        np.testing.assert_array_equal(value, result.pop("reference"))
        assert observation["received_ns"] < result["last_nonfinal_ns"], "Completion-only delivery fails milestone 1"
        assert observation["pid"] != os.getpid()
        assert observation["attribution"]["run_id"] == result["run_id"]
        assert observation["attribution"]["position"] == result["position"]
        result.update(received_ns=observation["received_ns"], capture_ns=observation["capture_ns"],
                      companion_pid=observation["pid"], width=value.size,
                      feature_hash=hashlib.sha256(value.tobytes()).hexdigest(),
                      lead_ms=(result["finished_ns"]-observation["received_ns"])/1e6)
    if args.repeat_prefix:
        assert all(r["cached_input_tokens"] == 0 for r in results), "Traced prompt skipped a cached prefix"
        assert results[0]["feature_hash"] == results[1]["feature_hash"], "Repeated prefill feature changed"
    return results, cache_control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--revision", required=True, help="Immutable checkpoint commit")
    parser.add_argument("--mapping", default="qwen3_5_moe.vllm.residual.v1", choices=MAPPINGS)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--concurrent", type=int, choices=[1, 2], default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    parser.add_argument("--taps", action="store_true")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--repeat-prefix", action="store_true")
    parser.add_argument("--output-tokens", type=int, default=128, choices=[32, 64, 128])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="iceblink-proof-") as directory:
        path = str(Path(directory) / "features.sock")
        parent, child = ctx.Pipe()
        process = ctx.Process(target=receive, args=(path, child, 2 if args.repeat_prefix else args.concurrent))
        process.start()
        try:
            if not parent.poll(10):
                raise RuntimeError("Companion startup timed out")
            assert parent.recv()["ready"]
            results, cache_control = asyncio.run(prove(args, path, parent))
            report = {"schema": 1, "test": "live_delivery_and_transport_equality",
                      "model": args.model, "revision": args.revision, "mapping": args.mapping,
                      "execution": "graph_taps" if args.taps else "eager", "prefix_cache": args.prefix_cache,
                      "versions": {p: importlib.metadata.version(p) for p in ("torch", "vllm", "nnsight", "transformers")},
                      "passed": True, "reference_mapping_validated": False, "requests": results,
                      "ordinary_prefix_cache_control": cache_control}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        finally:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


if __name__ == "__main__":
    main()
