"""Same-host companion service with bounded, private Unix socket transport."""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import time
import uuid

from .core import Limits, Policy, Rule
from .detectors import KeywordBaseline, load_detector
from .journal import Journal
from .live import MAX_DATAGRAM, decode_vector
from .monitor import Monitor
from .recording import Recorder

MAX_RPC = 1024 * 1024
RPC_CONNECTIONS = 32


def private_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    stat = path.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise PermissionError("Control directory must be owned by the service user with mode 0700")
    return path


class ServiceError(RuntimeError):
    pass


class Client:
    def __init__(self, directory, timeout=2):
        self.directory = Path(directory).resolve()
        self.timeout = timeout

    @property
    def observations(self):
        return str(self.directory / "observations.sock")

    def call(self, op, **parameters):
        message = json.dumps({"op": op, **parameters}, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        if len(message) > MAX_RPC:
            raise ValueError("Control message exceeds byte limit")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect(str(self.directory / "control.sock"))
            sock.sendall(message)
            with sock.makefile("rb") as stream:
                reply = json.loads(stream.readline(MAX_RPC))
        if not reply["ok"]:
            raise ServiceError(reply["code"])
        return reply.get("result")

    def register(self, policy, maximum, capabilities, run_id=None):
        run_id = run_id or uuid.uuid4().hex
        capability = self.call("register", run_id=run_id, policy_name=policy, maximum=maximum,
                               capabilities=capabilities)
        return run_id, capability


def read_config(path):
    path = Path(path)
    if path.stat().st_size > MAX_RPC:
        raise ValueError("Configuration exceeds limit")
    config = json.loads(path.read_text())
    if set(config) - {"detectors", "policies", "limits", "retention_days", "recording"}:
        raise ValueError("Unexpected monitor configuration field")
    detectors = []
    for item in config["detectors"]:
        if item["type"] == "keywords":
            detectors.append(KeywordBaseline())
        elif item["type"] == "artifact":
            detectors.append(load_detector(path.parent / item["path"]))
        else:
            raise ValueError("Detector type must be trusted and configured")
    thresholds = {d.name: d.threshold for d in detectors if hasattr(d, "threshold")}
    policies = []
    for p in config["policies"]:
        rules = tuple(Rule(r["detector"], r["threshold"] if "threshold" in r else thresholds[r["detector"]], r["mode"])
                      for r in p["rules"])
        policies.append(Policy(p["name"], rules, tuple(p["required"])))
    retention_days = float(config.get("retention_days", 30))
    if not 0 < retention_days < 36500:
        raise ValueError("Invalid retention period")
    recording = Recorder(**config["recording"]) if config.get("recording") else None
    return detectors, policies, Limits(**config.get("limits", {})), retention_days, recording


async def serve(directory, journal, detectors, policies, limits=Limits(), retention_days=30, recorder=None):
    directory = private_directory(directory)
    lock = (directory / "service.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    control_path, observation_path = directory / "control.sock", directory / "observations.sock"
    for path in (control_path, observation_path):
        path.unlink(missing_ok=True)
    monitor = Monitor(journal, detectors, policies, limits, recorder)
    loop = asyncio.get_running_loop()
    terminate = asyncio.Event()
    connections = 0
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, terminate.set)

    def dispatch(request):
        op = request.pop("op")
        if op == "register":
            return monitor.register_run(**request)
        if op == "admit":
            return monitor.admit(**request)
        if op == "ready":
            return monitor.ready(**request)
        if op == "end":
            return monitor.end_run(**request)
        if op == "unavailable":
            run = monitor.authorize(request["run_id"], request["capability"])
            req = journal.rows("SELECT id FROM requests WHERE id=? AND run_id=?", (request["request_id"], run["id"]))
            if not req:
                raise ValueError("Request ownership does not match")
            monitor.unavailable(run["id"], request["request_id"], "engine_reported_missing_prefill")
            return None
        if op == "policy":
            run = monitor.authorize(request["run_id"], request["capability"])
            policy = monitor.policies[run["policy"]]
            names = set(policy.required) | {r.detector for r in policy.rules}
            return {"sensors": sorted({s for name in names for s in monitor.detectors[name].sensors})}
        if op == "stop_service":
            terminate.set()
            return None
        if op == "status":
            return {"pid": os.getpid(), "queue_count": monitor.queue.count, "queue_bytes": monitor.queue.bytes,
                    "peak_queue_bytes": monitor.queue.peak_bytes, "retained_bytes": monitor.retained_bytes,
                    "policies": sorted(monitor.policies)}
        raise ValueError("Unknown control operation")

    async def handle(reader, writer):
        nonlocal connections
        if connections >= RPC_CONNECTIONS:
            writer.close()
            return
        connections += 1
        try:
            try:
                raw = await asyncio.wait_for(reader.readline(), 2)
                if not raw or len(raw) > MAX_RPC:
                    raise ValueError("Invalid control frame")
                request = json.loads(raw)
                result = dispatch(request)
                reply = {"ok": True, "result": result}
            except Exception as error:
                # Reason classes are safe to send to trusted clients. Do not log
                # parser exceptions, input fragments, tokens or detector errors.
                reply = {"ok": False, "code": type(error).__name__}
            writer.write(json.dumps(reply, separators=(",", ":")).encode() + b"\n")
            await asyncio.wait_for(writer.drain(), 2)
        except (OSError, TimeoutError):
            pass
        finally:
            connections -= 1
            writer.close()

    def receive():
        # Limit per-turn work so observation flood cannot starve control/health.
        for _ in range(16):
            try:
                raw = observations.recv(min(limits.observation_bytes, MAX_DATAGRAM) + 1)
            except BlockingIOError:
                return
            if len(raw) > min(limits.observation_bytes, MAX_DATAGRAM):
                continue
            try:
                packet = json.loads(raw)
                if packet["schema"] != 1:
                    continue
                attribution = packet["attribution"]
                if packet["sensor"] == "__prefill_ready__":
                    monitor.ready(attribution["execution_id"], attribution["capability"])
                else:
                    monitor.observation(attribution["execution_id"], attribution["capability"], packet["sensor"],
                                        decode_vector(packet["vector"]), int(packet["capture_ns"]), int(packet.get("worker_ns", 0)))
            except Exception:
                # The guard's outstanding-work deadline covers corrupted, missing
                # or unauthenticated observations, independently of heartbeats.
                continue

    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as observations:
        observations.bind(str(observation_path))
        observations.setblocking(False)
        os.chmod(observation_path, 0o600)
        loop.add_reader(observations, receive)
        server = await asyncio.start_unix_server(handle, str(control_path), limit=MAX_RPC)
        os.chmod(control_path, 0o600)
        next_prune = 0.0
        try:
            async with server:
                while not terminate.is_set():
                    monitor.tick()
                    if time.monotonic() >= next_prune:
                        journal.prune(time.time_ns()-int(retention_days*86400*1e9))
                        next_prune = time.monotonic()+3600
                    try:
                        await asyncio.wait_for(terminate.wait(), timeout=min(0.1, limits.heartbeat_seconds/4))
                    except TimeoutError:
                        pass
        finally:
            loop.remove_reader(observations)
            monitor.close()
            for path in (control_path, observation_path):
                path.unlink(missing_ok=True)
            lock.close()
