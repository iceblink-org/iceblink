"""Independent HF reference for post-block residual mapping on text prompts."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from .engine import CaptureEngine, EngineConfig

REFERENCE_MESSAGES = [
    [{"role": "user", "content": "Summarize: The lake trail opens again on Thursday."}],
    [{"role": "system", "content": "Summarize faithfully."}, {"role": "user", "content": "Use this note. " +
      "The community garden ordered new soil and tools. " * 12}],
]


async def record_vllm(engine, path):
    names = tuple(f"residual.{layer}.last_input" for layer in engine.config.layers)
    contents = {}
    for i, messages in enumerate(REFERENCE_MESSAGES):
        context = engine.render(messages)
        observed = await engine.extract(context, names)
        contents[f"input.{i}"] = np.array(context.token_ids, dtype=np.int64)
        for name, vector in observed.items():
            contents[f"{name}.{i}"] = vector
    with Path(path).open("xb") as stream:
        np.savez(stream, **contents)


def check_hf(config, path):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    # This is an independent eager HF forward, with no vLLM branch/residual
    # reconstruction or NNsight interception. Run after freeing the vLLM actor.
    conditional = config.mapping == "qwen3_5_moe.vllm.residual.v1"
    cls = AutoModelForImageTextToText if conditional else AutoModelForCausalLM
    model = cls.from_pretrained(config.model_path, dtype=torch.bfloat16, local_files_only=True,
                               attn_implementation="eager").eval().to("cuda")
    layers = model.model.language_model.layers if conditional else model.model.layers
    features = {}
    hooks = [layers[layer].register_forward_hook(lambda m, inputs, out, layer=layer:
              features.__setitem__(layer, out[0, -1].detach().float().cpu().numpy().copy())) for layer in config.layers]
    checks = []
    try:
        with np.load(path, allow_pickle=False) as expected, torch.inference_mode():
            for i in range(len(REFERENCE_MESSAGES)):
                ids = torch.tensor(expected[f"input.{i}"], device="cuda")[None, :]
                model(input_ids=ids, use_cache=False)
                for layer in config.layers:
                    actual = features[layer]
                    target = expected[f"residual.{layer}.last_input.{i}"]
                    l2 = float(np.linalg.norm(actual-target) / np.linalg.norm(actual))
                    cosine = float(np.dot(actual, target)/(np.linalg.norm(actual)*np.linalg.norm(target)))
                    # Independent BF16 implementations use different fused kernels;
                    # require a close direction AND scale, not bitwise equality.
                    checks.append({"case": i, "layer": layer, "input_tokens": ids.shape[-1], "width": actual.size,
                                   "relative_l2": l2, "cosine": cosine, "pass": l2 <= 0.25 and cosine >= 0.97})
    finally:
        for hook in hooks:
            hook.remove()
    return {"checkpoint": config.checkpoint, "revision": config.revision, "mapping": config.mapping,
            "reference": "HuggingFace eager BF16 decoder block output", "checks": checks,
            "passed": all(c["pass"] for c in checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check_hf(EngineConfig.from_json(args.engine), args.features)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
