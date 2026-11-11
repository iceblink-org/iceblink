# Iceblink

Iceblink monitors model calls routed through its vLLM adapter. NNsight reads selected input activations, then a separate companion process scores them and applies a named policy. The model, prompts, features and records stay in the provider's environment. The target is an **attempt to inject instructions**, independent of the actor's response.

Requires Python 3.12 or later and Linux for Unix sockets and owned process groups. CPU service and tests need no GPU.

The Python package is `iceblink`; the distribution is `iceblink` and the main command is `iceblink`. New detector exports use `iceblink.linear.v1`. Historical `scintilla.linear.v1` bundles remain loadable with the same fitted parameters, checksums and IDs.

From an environment outside the source tree:

```sh
python -m pip install -e '.[test]'
python -m pytest -q
```

Install `.[gpu,harness]` on the machine that serves the model. The adapter targets NNsight 0.8.2 and vLLM 0.27.1. Obtain the pinned checkpoint locally or point `model_path` to its immutable snapshot. Use `VLLM_NO_USAGE_STATS=1 HF_HUB_DISABLE_TELEMETRY=1`; offline runs can set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Supply any deployment-specific CUDA/compiler/Slurm settings from outside this repository. See `COMPATIBILITY.md` for tested settings and evidence.

## Run the local example

Create a private directory on the engine's host, on local storage. Start the service in its own process. The example control policies use a keyword signal to exercise the adapters independently of scientific probe quality.

```sh
ICEBLINK_STATE=$(mktemp -d)
iceblink --state-dir "$ICEBLINK_STATE" init
iceblink --state-dir "$ICEBLINK_STATE" serve --config examples/monitor.json &
iceblink --state-dir "$ICEBLINK_STATE" status
iceblink-demo --engine examples/engine.json --state-dir "$ICEBLINK_STATE" \
  --policy control-d --mode d --carrier tool --processing-seconds 45
```

Run `control-a`, `control-b`, or `control-c` with the corresponding mode. `--carrier user`, `tool`, and `clean` exercise the two carriers and a legitimate task. The warning case makes a follow-up model call. Modes c and d start a harmless owned job before the finding; the demo reports whether it is still alive before normal teardown. The example uses `ToolCallingAgent`, vLLM's async engine and its checkpoint-specific parser. It does not expose NNsight trace execution to clients or introduce a general-purpose model server.

Use the lab's exported artifacts by copying `examples/monitor-probe.json` to your external experiment workspace and pointing its `path` fields to the artifact directories. Omitted rule thresholds use the exported threshold. In the demo pass `--policy learned-a --mode a`. A score is an uncalibrated margin. A missing observation has no score. Neither actor refusal nor an alert alone measures detector accuracy.

The Python interface is `CaptureEngine(EngineConfig(...))` constructed on an asyncio loop, then `await MonitoredRun.create(engine, state_dir, journal, policy, mode, owned_tools=True)`. Use `await run.generate(messages, tools)` for all requests associated with that run and `await run.finish()` at the end. Separate runs can batch. Calls within a warning-capable run are serialized to deliver a warning once before its next call. The provider owns the engine, local socket, run registration and executor. Callers should never expose those controls as model tools.

## Monitor, actions and recovery

| Mode | On a finding | On monitoring loss past its deadline |
|---|---|---|
| a | Record and alert | Continue, record lost coverage |
| b | Queue a fixed trusted warning for the next call | Continue, record lost coverage |
| c | Abort this run's generation and deny later calls | Same stop; existing tools continue |
| d | Do c, deny tool starts and verify owned process group cancellation | Same model and tool stop |

Rules compare individual scores with thresholds. For one prefill evaluation, all findings are recorded and the strongest allowed action wins. The watchdog runs in the adapter, checks heartbeat and outstanding required work, and reads persisted stop state independently of the companion. Guarded runs check capabilities and compatibility before work. Pending actions remain pending until the adapter acknowledges them. A stopped run cannot retry with a new request ID. An operator can clear an acknowledged stop; doing so does not restore an interrupted generation or restart an ended run. Supported tools must remain in their owned process group. Completed effects and external jobs are outside this cancellation contract.

If a warning reaches the actor while its receipt cannot be written, the adapter retains a bounded receipt and retries the write without injecting it again. Closing a run during a monitoring outage still enforces c/d. An adapter keeps reconciling an uncommitted stop or pending acknowledgement rather than declaring it complete when its caller closes.

```sh
iceblink --state-dir "$ICEBLINK_STATE" events --follow
iceblink --state-dir "$ICEBLINK_STATE" export --output /your/external/events.jsonl
iceblink --state-dir "$ICEBLINK_STATE" backup --output /your/external/journal.sqlite
iceblink --state-dir "$ICEBLINK_STATE" clear-stop RUN_ID
iceblink --state-dir "$ICEBLINK_STATE" stop-service
```

Closing a viewer has no effect on monitoring. `Journal.events(after=..., run_id=...)` offers the same ordered records to Python integrations. SQLite WAL requires same-host local storage. The backup command uses SQLite's online backup API and can write to network storage. On a cluster, export or back up before allocation teardown; node-local storage is ephemeral. Restore a surviving backup to local storage before starting the service. `init` is explicit so a missing database cannot silently stand in for surviving guarded state. Records that never committed may be lost.

Default retention is 30 days for event history. Stop and action state is retained independently. The default journal stores references, metrics, scores, coverage and control outcomes; it omits text, token IDs and vectors. Optional `recording` configuration (`directory`, `max_count`, `max_bytes`, `retention_seconds`) writes private selected-feature replay files; it defaults off and has independent limits. `read_record` loads those selected inputs. This does not recover lost activations or securely erase GPU memory. Apply deletion and backup policies to that external recording directory as needed.

## Extension boundaries

`SensorGraph` validates dependencies and caches shared computations per request. The engine resolves the graph's token, span and site selections against effective input IDs; the worker reduces only selected tensors. The built-in selectors cover the last rendered input token, the whole input mean and a tool span mean when provenance is available. `integrations/mappings.py` owns model-specific structure. Text-based and activation detectors implement the same `Detector` contract; trusted Python code can register further detectors or sensor operations. Artifacts select a fixed, non-executable predictor and use a validated manifest and SafeTensors. The first policies evaluate one prefill request. A custom later readiness phase or run-history detector should use explicitly bounded inputs.

The tested deployment uses eager execution with prefix caching and speculative decoding disabled. Disabling prefix caching preserves prefill coverage for repeated context; no full traces are retained by default. The worker's private nonblocking datagrams are bounded; queue loss becomes unavailable input and invokes policy. Only model calls routed through this adapter and text actually supplied to those calls are monitored. The system does not inspect external tool internals or unrelated requests.
