"""Local serving comparison using identical text, lengths and token budgets."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import time
import uuid

from .engine import EngineConfig, MonitoredRun, render_input

CASES = ((512, 1), (512, 2), (1536, 1), (1536, 2))
OUTPUT_TOKENS = 32


def messages_for(length, index):
    body = "The park office will lend rakes, gloves, and bags to volunteers on Saturday. "
    return [{"role": "system", "content": "Summarize the notes accurately."},
            {"role": "user", "content": f"Notes for case {index}. " + body * (length//16) + " Write a summary."}]


def gpu_memory_mib():
    try:
        import pynvml
        import torch

        pynvml.nvmlInit()
        props = torch.cuda.get_device_properties(0)
        try:
            handle = pynvml.nvmlDeviceGetHandleByUUID(str(props.uuid))
        except Exception:
            if pynvml.nvmlDeviceGetCount() != 1:
                return None
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
    except Exception:
        return None


async def plain_call(engine, context):
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, min_tokens=OUTPUT_TOKENS)
    start = time.monotonic_ns()
    first = None
    async for output in engine.generate({"prompt_token_ids": list(context.token_ids)}, params, uuid.uuid4().hex):
        if first is None:
            first = time.monotonic_ns()
    return {"input_tokens": len(context.token_ids), "output_tokens": len(output.outputs[0].token_ids),
            "ttft_ms": (first-start)/1e6, "latency_ms": (time.monotonic_ns()-start)/1e6,
            "cached_input_tokens": output.num_cached_tokens}


def summarize(trials):
    rows = [row for trial in trials for row in trial["requests"]]
    return {"input_tokens": [r["input_tokens"] for r in rows], "requests": len(rows),
            "output_tokens_per_second": sum(r["output_tokens"] for r in rows)/sum(t["wall_seconds"] for t in trials),
            "median_ttft_ms": statistics.median(r["ttft_ms"] for r in rows),
            "median_latency_ms": statistics.median(r["latency_ms"] for r in rows),
            "cached_input_tokens": [r.get("cached_input_tokens") for r in rows], "trials": trials}


async def benchmark_monitored(engine, directory, journal, policy, limits):
    before = journal.events(limit=10000)
    cursor = before[-1]["sequence"] if before else 0
    report = {"checkpoint": engine.config.checkpoint, "revision": engine.config.revision, "precision": engine.config.precision,
              "mode": "monitored eager versus matched eager with capture off", "gpu_memory_mib": [], "cases": []}
    run = await MonitoredRun.create(engine, directory, journal, policy, "a", limits=limits)
    try:
        for nominal, concurrency in CASES:
            messages = [messages_for(nominal, i) for i in range(concurrency)]
            contexts = [engine.render(m) for m in messages]
            await asyncio.gather(*(plain_call(engine.engine, c) for c in contexts))  # warmup
            await asyncio.gather(*(run.generate(m, max_tokens=OUTPUT_TOKENS, minimum=OUTPUT_TOKENS) for m in messages))
            case = {"nominal_prompt_tokens": nominal, "concurrency": concurrency}
            for active in (False, True):
                trials = []
                for _ in range(2):
                    last = journal.events(limit=10000)
                    after = last[-1]["sequence"] if last else 0
                    start = time.monotonic()
                    if active:
                        await asyncio.gather(*(run.generate(m, max_tokens=OUTPUT_TOKENS, minimum=OUTPUT_TOKENS) for m in messages))
                        requests = [{"input_tokens": e["payload"]["input_tokens"], "output_tokens": e["payload"]["output_tokens"],
                                     "ttft_ms": e["payload"]["ttft_ns"]/1e6, "latency_ms": e["payload"]["generation_ns"]/1e6}
                                    for e in journal.events(after=after, run_id=run.run_id) if e["kind"] == "generation_finished"]
                    else:
                        requests = await asyncio.gather(*(plain_call(engine.engine, c) for c in contexts))
                    trials.append({"wall_seconds": time.monotonic()-start, "requests": requests})
                    report["gpu_memory_mib"].append(gpu_memory_mib())
                case["monitored" if active else "matched"] = summarize(trials)
            case["relative_throughput_cost"] = 1-case["monitored"]["output_tokens_per_second"]/case["matched"]["output_tokens_per_second"]
            report["cases"].append(case)
    finally:
        await run.finish()
    events = journal.events(after=cursor)
    observations = [e["payload"] for e in events if e["kind"] == "observation"]
    scores = [e["payload"]["detection_ns"] for e in events if e["kind"] == "score"]
    report["observation_metrics"] = {"count": len(observations), "feature_bytes_max": max((x["bytes"] for x in observations), default=0),
        "median_worker_ms": statistics.median(x["worker_ns"]/1e6 for x in observations) if observations else None,
        "median_transport_ms": statistics.median(x["transport_ns"]/1e6 for x in observations) if observations else None,
        "median_detection_ms": statistics.median(scores)/1e6 if scores else None,
        "queue_peak_bytes": run.client.call("status")["peak_queue_bytes"]}
    starts = {e["request_id"]: e["monotonic_ns"] for e in journal.events(limit=10000) if e["kind"] == "request_admitted"}
    report["alert_delay_ms"] = [(e["monotonic_ns"]-starts[e["request_id"]])/1e6 for e in journal.events(limit=10000)
                                if e["kind"] == "finding" and e["request_id"] in starts]
    return report


async def benchmark_optimized(config):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=config.model_path, revision=config.revision,
        dtype=config.precision, max_model_len=config.max_context, max_num_seqs=config.max_concurrency,
        max_num_batched_tokens=config.max_context*config.max_concurrency, gpu_memory_utilization=config.gpu_memory_utilization,
        language_model_only=config.language_model_only, enforce_eager=False, enable_prefix_caching=True, disable_log_stats=True))
    tokenizer = engine.get_tokenizer()
    report = {"checkpoint": config.checkpoint, "revision": config.revision, "precision": config.precision,
              "mode": "vanilla optimized vLLM, graphs and prefix caching enabled", "gpu_memory_mib": [], "cases": []}
    try:
        for nominal, concurrency in CASES:
            contexts = [render_input(tokenizer, messages_for(nominal, i), enable_thinking=config.enable_thinking) for i in range(concurrency)]
            await asyncio.gather(*(plain_call(engine, c) for c in contexts))
            trials = []
            for _ in range(2):
                start = time.monotonic()
                requests = await asyncio.gather(*(plain_call(engine, c) for c in contexts))
                trials.append({"wall_seconds": time.monotonic()-start, "requests": requests})
                report["gpu_memory_mib"].append(gpu_memory_mib())
            report["cases"].append({"nominal_prompt_tokens": nominal, "concurrency": concurrency, "optimized": summarize(trials)})
    finally:
        engine.shutdown()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = asyncio.run(benchmark_optimized(EngineConfig.from_json(args.engine)))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("OPTIMIZED_BENCHMARK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
