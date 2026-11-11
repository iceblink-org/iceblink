"""Companion-process monitoring core; scores never directly execute tools."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from queue import Empty
import secrets
import threading
import time
import uuid

import numpy as np

from .bounded import BudgetQueue
from .core import Compatibility, Limits, Mode, Policy, Score, identity
from .journal import Journal


@dataclass
class Pending:
    run_id: str
    request_id: str
    execution_id: str
    inputs: dict
    size: int
    capability: str
    sensors: tuple[str, ...]
    queued: bool = False
    done: bool = False
    evidence: dict[str, str] = field(default_factory=dict)
    replay: dict | None = None
    tool_call_id: str | None = None


class Monitor:
    def __init__(self, journal: Journal, detectors, policies: list[Policy], limits=Limits(), recorder=None):
        self.journal, self.limits = journal, limits
        self.recorder = recorder
        self.detectors = {detector.name: detector for detector in detectors}
        self.policies = {policy.name: policy for policy in policies}
        if len(self.detectors) != len(detectors) or len(self.policies) != len(policies):
            raise ValueError("Duplicate detector or policy")
        for policy in policies:
            names = set(policy.required) | {rule.detector for rule in policy.rules}
            if not names <= self.detectors.keys():
                raise ValueError("Policy uses an unknown detector")
            for name in names:
                detector = self.detectors[name]
                if detector.scope != policy.scope or detector.readiness != policy.readiness:
                    raise ValueError("Detector readiness/scope does not match policy")
        self.pending = {}
        self.retained_bytes = 0
        self.queue = BudgetQueue(limits.queue_count, limits.queue_bytes)
        self.lock = threading.RLock()
        self.shutdown = threading.Event()
        self.ending = {}
        self.worker = threading.Thread(target=self._work, name="iceblink-detectors", daemon=True)
        self._recover()
        self.worker.start()

    def _recover(self):
        # Committed scores/stops survive. Unsaved inputs cannot be reconstructed.
        for request in self.journal.rows("SELECT * FROM requests WHERE status='pending'"):
            self.unavailable(request["run_id"], request["id"], "monitor_restart_lost_observation")
        self.journal.event("service_started", payload={"reconciliation": "pending_inputs_unavailable"})
        self.journal.heartbeat()

    def register_run(self, run_id, policy_name, maximum, capabilities):
        maximum = Mode.parse(maximum)
        policy = self.policies[policy_name]
        if any(Mode.parse(rule.mode) > maximum for rule in policy.rules):
            raise ValueError("Policy exceeds preauthorized response")
        if maximum >= Mode.c:
            if not capabilities.get("model_abort") or not capabilities.get("model_gate"):
                raise ValueError("Guarded mode requires engine abort and admission control")
            if maximum == Mode.d and not capabilities.get("tool_cancel"):
                raise ValueError("Mode d requires an owned cancellable executor")
            required = {s for name in policy.required for s in self.detectors[name].sensors}
            if not required <= set(capabilities.get("sensors", [])):
                raise ValueError("Required sensors unsupported")
            for name in policy.required:
                detector = self.detectors[name]
                if hasattr(detector, "compatibility"):
                    detector.compatibility.require(Compatibility(**capabilities["compatibility"][detector.sensors[0]]))
        with self.lock, self.journal.transaction() as db:
            if db.execute("SELECT 1 FROM runs WHERE id=?", (run_id,)).fetchone():
                raise ValueError("Run ID already registered; use its existing capability")
            active = db.execute("SELECT COUNT(*) FROM runs WHERE ended=0").fetchone()[0]
            if active + len(self.ending) >= self.limits.active_runs:
                raise ValueError("Active run budget exhausted")
            capability = secrets.token_hex(32)
            db.execute("INSERT INTO runs(id,policy,maximum,capability) VALUES(?,?,?,?)",
                       (run_id, policy_name, int(maximum), capability))
            self.journal.event("run_registered", run_id=run_id,
                               payload={"policy": policy_name, "policy_id": identity(asdict(policy)), "maximum": maximum.name})
            return capability

    def authorize(self, run_id, capability):
        run = self.journal.run(run_id)
        if not secrets.compare_digest(run["capability"], capability):
            raise PermissionError("Invalid run capability")
        return run

    def admit(self, run_id, capability, request_id, execution_id, *, text, token_ids, compatibility,
              observation_capability=None, tool_call_id=None, spans=()):
        with self.lock:
            run = self.authorize(run_id, capability)
            if run["model_stop"] or run["ended"]:
                raise PermissionError("Run stopped or ended")
            if len(token_ids) > self.limits.context_tokens or not token_ids:
                raise ValueError("Effective token budget exceeded or empty input")
            if not isinstance(text, str) or len(text.encode()) > self.limits.text_bytes:
                raise ValueError("Text budget exceeded")
            policy = self.policies[run["policy"]]
            names = set(policy.required) | {rule.detector for rule in policy.rules}
            for name in names:
                detector = self.detectors[name]
                if hasattr(detector, "compatibility"):
                    detector.compatibility.require(Compatibility(**compatibility[detector.sensors[0]]))
            sensors = tuple(sorted({s for name in names for s in self.detectors[name].sensors}))
            size = len(text.encode()) + len(token_ids) * 32 + len(json.dumps(spans).encode()) + 1024
            if len(self.pending) >= self.limits.active_requests or self.retained_bytes + size > self.limits.retained_bytes:
                self.unavailable(run_id, request_id, "request_state_budget_exhausted")
                raise ValueError("Request state budget exhausted")
            request_capability = observation_capability or secrets.token_hex(32)
            if len(request_capability) != 64 or any(c not in "0123456789abcdef" for c in request_capability):
                raise ValueError("Invalid observation capability format")
            with self.journal.transaction() as db:
                db.execute("INSERT INTO requests VALUES(?,?,?,?,NULL,'pending',?,?)",
                           (request_id, execution_id, run_id, time.monotonic_ns(), json.dumps(policy.required), request_capability))
                evidence_id = self.journal.event("request_admitted", run_id=run_id, request_id=request_id, tool_call_id=tool_call_id,
                                   payload={"execution_id": execution_id, "input_tokens": len(token_ids),
                                            "last_input_position": len(token_ids)-1, "sensors": sensors,
                                            "provenance_spans": spans})
            # Effective IDs are kept temporarily by engine/sensors; history saves
            # references and positions only unless explicit content recording is used.
            pending = Pending(run_id, request_id, execution_id, {"text": text}, size, request_capability, sensors,
                              evidence={"text": evidence_id}, tool_call_id=tool_call_id)
            if self.recorder is not None:
                pending.replay = {"run_id": run_id, "request_id": request_id, "execution_id": execution_id,
                                  "token_ids": token_ids, "spans": spans, "compatibility": compatibility}
            self.pending[execution_id] = pending
            self.retained_bytes += size
            return {"observation_capability": request_capability, "sensors": sensors}

    def observation(self, execution_id, capability, sensor, value, capture_ns, worker_ns=0):
        with self.lock:
            pending = self.pending.get(execution_id)
            if pending is None:
                rows = self.journal.rows("SELECT id,run_id,capability FROM requests WHERE execution=?", (execution_id,))
                if rows and secrets.compare_digest(rows[0]["capability"], capability):
                    self.journal.event("late_observation", run_id=rows[0]["run_id"], request_id=rows[0]["id"],
                                       status="unavailable", payload={"sensor": sensor})
                return
            if not secrets.compare_digest(pending.capability, capability):
                raise PermissionError("Invalid observation capability")
            if sensor not in pending.sensors or pending.queued or sensor in pending.evidence:
                return
            size = int(value.nbytes)
            if self.retained_bytes + size > self.limits.retained_bytes:
                self.unavailable(pending.run_id, pending.request_id, "retained_byte_budget_exhausted")
                self._release(pending)
                return
            pending.inputs[sensor] = value
            pending.size += size
            self.retained_bytes += size
            eid = self.journal.event("observation", run_id=pending.run_id, request_id=pending.request_id, tool_call_id=pending.tool_call_id,
                                     component="sensor", payload={"sensor": sensor, "bytes": size,
                                      "capture_ns": capture_ns, "worker_ns": worker_ns,
                                      "transport_ns": max(0, time.monotonic_ns()-capture_ns-worker_ns)})
            pending.evidence[sensor] = eid
            self._schedule(pending)

    def ready(self, execution_id, capability):
        with self.lock:
            pending = self.pending[execution_id]
            if not secrets.compare_digest(pending.capability, capability):
                raise PermissionError("Invalid observation capability")
            self._schedule(pending, final=True)

    def _schedule(self, pending, final=False):
        if pending.queued or (not final and not all(s in pending.inputs for s in pending.sensors)):
            return
        pending.queued = True
        if not self.queue.put(pending, pending.size):
            self.unavailable(pending.run_id, pending.request_id, "detector_queue_overload")
            self._release(pending)

    def _release(self, pending):
        if self.pending.pop(pending.execution_id, None) is not None:
            self.retained_bytes -= pending.size
        pending.done = True
        pending.inputs.clear()
        pending.replay = None

    def _work(self):
        while not self.shutdown.is_set():
            try:
                pending, size = self.queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if pending.done:
                    continue
                policy = self.policies[self.journal.run(pending.run_id)["policy"]]
                names = sorted(set(policy.required) | {r.detector for r in policy.rules})
                scores = []
                start = time.monotonic_ns()
                for name in names:
                    detector = self.detectors[name]
                    try:
                        evidence = tuple(pending.evidence.values())
                        scores.append(detector.evaluate(pending.inputs, evidence))
                    except Exception:
                        # Exception text can contain raw input. Persist only a code.
                        scores.append(Score(name, detector.version, status="error", reason="detector_exception"))
                self._results(pending, scores, time.monotonic_ns()-start)
                if self.recorder and pending.replay is not None:
                    try:
                        record_id = self.recorder.write(pending.replay, pending.inputs)
                        self.journal.event("feature_recorded" if record_id else "feature_recording_dropped",
                                           run_id=pending.run_id, request_id=pending.request_id,
                                           component="recorder", payload={"record_id": record_id})
                    except OSError:
                        self.journal.event("feature_recording_failed", run_id=pending.run_id, request_id=pending.request_id,
                                           component="recorder", status="failed")
            except Exception:
                # A journal failure prevents a successful completion. The adapter
                # watchdog sees outstanding work/stale heartbeat and fails by mode.
                pass
            finally:
                with self.lock:
                    self._release(pending)
                self.queue.release(size)

    def _results(self, pending, scores, detection_ns):
        with self.lock, self.journal.transaction() as db:
            run = self.journal.run(pending.run_id)
            policy = self.policies[run["policy"]]
            score_map = {score.detector: score for score in scores}
            for score in scores:
                self.journal.event("score", run_id=pending.run_id, request_id=pending.request_id, tool_call_id=pending.tool_call_id,
                                   component="detector", version=score.version, status=score.status,
                                   payload={**asdict(score), "detection_ns": detection_ns})
            missing = [name for name in policy.required if score_map[name].status != "evaluated"]
            if missing:
                self.unavailable(pending.run_id, pending.request_id, "required_detector_unavailable")
            else:
                db.execute("UPDATE requests SET status='evaluated',completed_ns=? WHERE id=? AND status='pending'",
                           (time.time_ns(), pending.request_id))
                if run["degraded"]:
                    db.execute("UPDATE runs SET degraded=0 WHERE id=?", (pending.run_id,))
                    self.journal.event("coverage_resumed", run_id=pending.run_id, request_id=pending.request_id)
            findings = []
            for rule in policy.rules:
                score = score_map[rule.detector]
                if score.status == "evaluated" and score.value >= rule.threshold:
                    eid = self.journal.event("finding", run_id=pending.run_id, request_id=pending.request_id, tool_call_id=pending.tool_call_id,
                        payload={"category": score.target, "detector": score.detector, "version": score.version,
                                 "score": score.value, "threshold": rule.threshold, "mode": rule.mode,
                                 "evidence": score.evidence})
                    findings.append((Mode.parse(rule.mode), eid))
            if findings:
                selected = max(mode for mode, _ in findings)
                evidence = tuple(eid for _, eid in findings)
                self.journal.event("decision", run_id=pending.run_id, request_id=pending.request_id, tool_call_id=pending.tool_call_id,
                                   component="policy", version=identity(asdict(policy)),
                                   payload={"mode": selected.name, "findings": evidence})
                if selected >= Mode.c:
                    self.journal.stop(pending.run_id, selected, source="monitor", request_id=pending.request_id, evidence=evidence)
                elif selected == Mode.b:
                    wid = uuid.uuid4().hex
                    state = "undelivered" if run["ended"] else "pending"
                    db.execute("INSERT INTO warnings VALUES(?,?,?,?,?,NULL)",
                               (wid, pending.run_id, pending.request_id, evidence[0], state))
                    self.journal.event("warning_queued" if state == "pending" else "warning_undelivered",
                                       run_id=pending.run_id, request_id=pending.request_id, payload={"warning_id": wid})

    def unavailable(self, run_id, request_id, reason):
        # May be called from an outer transaction; use its existing connection.
        run = self.journal.run(run_id)
        with self.journal.lock:
            self.journal.db.execute("UPDATE runs SET degraded=1 WHERE id=?", (run_id,))
            self.journal.db.execute("UPDATE requests SET status='unavailable',completed_ns=? WHERE id=?",
                                    (time.time_ns(), request_id))
        self.journal.event("coverage_lost", run_id=run_id, request_id=request_id,
                           status="unavailable", payload={"reason": reason})
        if reason != "required_detector_unavailable":
            policy = self.policies.get(run["policy"])
            if policy:
                for name in policy.required:
                    score = Score(name, self.detectors[name].version, reason=reason)
                    self.journal.event("score", run_id=run_id, request_id=request_id, component="detector",
                                       version=score.version, status=score.status, payload=asdict(score))
        maximum = Mode(run["maximum"])
        if maximum >= Mode.c and not (run["tool_stop"] if maximum == Mode.d else run["model_stop"]):
            self.journal.stop(run_id, maximum, source="monitor_health", request_id=request_id)

    def end_run(self, run_id, capability):
        with self.lock:
            if self.authorize(run_id, capability)["ended"]:
                return
            with self.journal.lock:
                self.journal.db.execute("UPDATE runs SET ended=1 WHERE id=?", (run_id,))
            self.ending[run_id] = time.monotonic() + self.limits.drain_seconds
            self.journal.event("run_draining", run_id=run_id)

    def tick(self):
        with self.lock:
            self.journal.heartbeat()
            now = time.monotonic_ns()
            expired = self.journal.rows("SELECT id,run_id FROM requests WHERE status='pending' AND created_ns<?",
                                        (now-int(self.limits.processing_seconds*1e9),))
            for request in expired:
                self.unavailable(request["run_id"], request["id"], "required_processing_deadline")
                for pending in tuple(self.pending.values()):
                    if pending.request_id == request["id"] and not pending.queued:
                        self._release(pending)
            for run_id, deadline in list(self.ending.items()):
                active = [p for p in self.pending.values() if p.run_id == run_id]
                if active and time.monotonic() < deadline:
                    continue
                for pending in active:
                    self.unavailable(run_id, pending.request_id, "drain_deadline")
                    if not pending.queued:
                        self._release(pending)
                with self.journal.transaction() as db:
                    for warning in self.journal.rows("SELECT id,request_id FROM warnings WHERE run_id=? AND status='pending'", (run_id,)):
                        self.journal.event("warning_undelivered", run_id=run_id, request_id=warning["request_id"], payload={"warning_id": warning["id"]})
                    db.execute("UPDATE warnings SET status='undelivered' WHERE run_id=? AND status='pending'", (run_id,))
                    self.journal.event("run_finished", run_id=run_id, status="incomplete" if active else "recorded")
                del self.ending[run_id]

    def close(self):
        self.shutdown.set()
        self.worker.join(timeout=self.limits.drain_seconds)
        self.journal.event("service_stopped", status="incomplete" if self.worker.is_alive() else "recorded")
