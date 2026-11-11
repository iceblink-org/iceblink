"""Exercise failure paths with real companion and process-group state."""
import json
import os
import sys
import threading
import time

import numpy as np
import pytest

from iceblink.adapters import OwnedExecutor, RunGuard, RunStopped
from iceblink.core import Limits, Policy, Rule, Score
from iceblink.detectors import KeywordBaseline
from iceblink.journal import Journal
from iceblink.monitor import Monitor
from iceblink.recording import Recorder, read_record


def wait_until(predicate, timeout=3):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out")


class Stalled:
    name, version, sensors = "stalled", "test", ("text",)
    readiness, scope = "input_prefill", "request"

    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()

    def evaluate(self, inputs, evidence):
        self.entered.set()
        self.release.wait(timeout=4)
        return Score(self.name, self.version, value=1, status="evaluated", evidence=evidence)


def test_live_heartbeat_during_detector_stall(tmp_path):
    journal = Journal(tmp_path / "journal.sqlite")
    stalled = Stalled()
    limits = Limits(heartbeat_seconds=0.3, processing_seconds=0.45, drain_seconds=0.1)
    monitor = Monitor(journal, [stalled], [Policy("guard", (Rule(stalled.name, 2, "c"),), (stalled.name,))], limits)
    cap = monitor.register_run("run", "guard", "c", {"model_abort": True, "model_gate": True, "sensors": ["text"]})
    stopped = threading.Event()
    guard = RunGuard(journal, "run", abort_model=lambda: stopped.set() or True, limits=limits, interval=0.01)
    try:
        result = monitor.admit("run", cap, "request", "execution", text="ordinary note", token_ids=[10], compatibility={})
        monitor.ready("execution", result["observation_capability"])
        assert stalled.entered.wait(1)
        start = time.monotonic()
        # Heartbeat still works while the detector is blocked. This loop
        # deliberately does not run Monitor.tick's deadline enforcement:
        # the independent watchdog must detect lack of required progress.
        while not stopped.is_set() and time.monotonic()-start < 2:
            journal.heartbeat()
            time.sleep(0.025)
        assert stopped.is_set() and not stalled.release.is_set()
        assert time.monotonic_ns()-journal.health("run")["heartbeat_ns"] < 0.3e9
        assert any(e["kind"] == "stop_requested" and e["component"] == "watchdog" for e in journal.events())
    finally:
        stalled.release.set()
        guard.close()
        monitor.close()
        journal.close()


def test_missing_journal_cannot_reopen_guarded_run(tmp_path):
    path = tmp_path / "journal.sqlite"
    journal = Journal(path)
    monitor = Monitor(journal, [KeywordBaseline()], [Policy("guard", (), ("text-keywords",))])
    monitor.register_run("run", "guard", "c", {"model_abort": True, "model_gate": True, "sensors": ["text"]})
    stopped = threading.Event()
    guard = RunGuard(journal, "run", abort_model=lambda: stopped.set() or True, interval=0.01)
    try:
        path.rename(tmp_path / "offline.sqlite")
        wait_until(stopped.is_set)
        with pytest.raises(RunStopped):
            guard.ensure_model_allowed()
        with pytest.raises(FileNotFoundError):
            Journal(path, create=False)
    finally:
        path.with_name("offline.sqlite").rename(path)
        guard.close()
        monitor.close()
        journal.close()


def test_opt_in_replay_is_bounded_and_not_in_journal(tmp_path):
    recorder = Recorder(tmp_path / "records", max_count=1, max_bytes=2048)
    a = recorder.write({"token_ids": [1, 2]}, {"text": "private fixture", "site": np.array([1.0, -2.0])})
    b = recorder.write({"token_ids": [7]}, {"text": "second fixture", "site": np.array([-1.0])})
    assert a != b and not (recorder.path / f"record-{a}.json").exists()
    path = recorder.path / f"record-{b}.json"
    assert path.stat().st_mode & 0o077 == 0
    metadata, inputs = read_record(path)
    assert metadata["token_ids"] == [7] and inputs["text"] == "second fixture"
    np.testing.assert_equal(inputs["site"], [-1.0])
    assert recorder.write({}, {"text": "x" * 2049}) is None


def test_mode_d_cancels_noncooperative_owned_descendant(tmp_path):
    journal = Journal(tmp_path / "journal.sqlite")
    monitor = Monitor(journal, [KeywordBaseline()], [Policy("guard", (), ("text-keywords",))])
    monitor.register_run("run", "guard", "d", {"model_abort": True, "model_gate": True, "tool_cancel": True, "sensors": ["text"]})
    executor = OwnedExecutor(journal, "run", grace=0.05)
    guard = RunGuard(journal, "run", abort_model=lambda: True, executor=executor, interval=0.02)
    ready = tmp_path / "child-ready"
    code = ("import os,pathlib,signal,sys,time\n"
            "if os.fork()==0:\n"
            " signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            " pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
            " time.sleep(30)\n"
            "else: time.sleep(30)\n")
    _, child = executor.start([sys.executable, "-c", code, str(ready)])
    try:
        wait_until(ready.exists)
        assert child.poll() is None and executor.group_alive(child.pid)
        action = journal.stop("run", "d", source="test")
        wait_until(lambda: journal.rows("SELECT status FROM actions WHERE id=?", (action,))[0]["status"] == "success")
        assert child.poll() is not None and not executor.group_alive(child.pid)
        with pytest.raises(RunStopped):
            executor.start([sys.executable, "-c", "pass"])
    finally:
        guard.close()
        executor.cancel_all(force=True)
        monitor.close()
        journal.close()
