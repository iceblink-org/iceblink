# Compatibility evidence

The approved configuration uses Qwen/Qwen3.6-35B-A3B at revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`, BF16, the checkpoint tokenizer and chat template, non-thinking text inputs, and a single NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB). Driver 610.57.04 exposes CUDA 13.3; the tested PyTorch 2.13.0 wheel was built for CUDA 13.0. Python 3.12.9, NNsight 0.8.2, vLLM 0.27.1, Transformers 5.22.2 and smolagents 1.26.0 were used. Pin a full environment outside this package when reproducing the measurements.

The vLLM decoder block returns the MLP branch and the accumulated residual separately. Their sum at the consumed token position is the post-block residual. The wrapper's `language_model.model.layers` provides that site. Independent Hugging Face eager BF16 comparison on two rendered inputs (24 and 134 tokens) covered layers 11, 18, 19 and 27, including both full attention and DeltaNet. The eight cosine similarities were 0.99598–0.99991; relative L2 error was 0.01338–0.08985. BF16 kernels are not bitwise equivalent. Shape alone is not a compatibility test.

In the live proof, two concurrent model calls each sent a 2,048-element selected vector to a different process while generation was still streaming. Their last-input positions and effective input IDs were checked. The wire vectors matched the selected worker copies exactly; both arrived about 35.7 seconds before completion in the long proof generation. This lead time demonstrates ordering, not typical alert latency.

The serving example uses eager execution, text-only initialization, prefix caching off, speculative decoding off, maximum context 2,048, concurrency two, and selected-vector capture only. Text-only model loading reported 64.69 GiB of weight memory. Each FP32 feature is 8,192 bytes. Guarded mode must reject unsupported sensor or abort/cancellation capabilities. No other model, precision, quantization or distributed execution mode is implied compatible. The L40S 46,068 MiB development fixture was a separate small Qwen checkpoint.

The GPU proof is explicit and separate from CPU tests:

```sh
iceblink-gpu-proof --model "$CHECKPOINT_PATH" \
  --revision 345d467b117d1fd5eda9b879a5988de9ae5f47a7 \
  --output "$WORK/approved-live.json"
```

Graph taps with prefix caching off also passed a two-request test. Effective input IDs and the selected feature hashes matched eager exactly on those inputs, and the companion received both vectors about 329 ms before completion. The hybrid prefill remains eager on this graph configuration. This is a compatibility check, not an instruction to switch the default deployment.

The prefix-cache compatibility check rejected cache-on with chunked prefill disabled: vLLM requires aligned chunking for the hybrid model. With the required chunking enabled, NNsight rejected a 1,164-token prompt split at 1,056 tokens, so it did not produce a misleading last-input observation. Keep both settings off for the monitored deployment. See the external validation records for the vanilla optimized serving comparison and its limits.

Warmed serving measurements at effective input lengths 579 and 1,667, concurrency one and two, and 32 generated tokens showed 0.8–5.1% throughput loss relative to matched eager serving without capture. Relative to vanilla optimized vLLM with graphs and prefix caching, loss was 67.6–73.7%. This integration's main cost is the lost engine optimizations. There were two measured repetitions per case; these are configuration-specific observations rather than a production performance guarantee. The longer optimized requests reused 1,056 prefix tokens.

Sources: [Qwen model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [NNsight async serving](https://nnsight.net/vllm/serving/), [NNsight performance](https://nnsight.net/vllm/performance/), [SQLite WAL](https://sqlite.org/wal.html).
