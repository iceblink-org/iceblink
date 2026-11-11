from dataclasses import replace
import json

import numpy as np
import pytest

from iceblink.bounded import BudgetQueue
from iceblink.core import Compatibility, identity
from iceblink.detectors import export_detector, load_detector
from iceblink.live import decode_vector, encode_vector
from iceblink.sensors import InputContext, Sensor, SensorGraph, Span, capture_plan, prefill_graph


def test_sensor_graph_shares_dependencies_and_rejects_cycles():
    calls = []

    def source(context, deps, params):
        calls.append(context)
        return 3

    graph = SensorGraph([Sensor("shared", "source"), Sensor("a", "twice", ("shared",)),
                         Sensor("b", "twice", ("shared",))], {"source": source, "twice": lambda c, d, p: 2*d[0]})
    ctx = InputContext((10, 20, 30))
    assert graph.evaluate(ctx, ("a", "b")) == {"a": 6, "b": 6}
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Cyclic"):
        SensorGraph([Sensor("a", "text", ("b",)), Sensor("b", "text", ("a",))])


def test_effective_input_and_tool_positions_owned():
    layer = np.arange(12, dtype=np.float32).reshape(4, 3)
    ctx = InputContext((2, 4, 5, 99), (Span(1, 3, "tool", "t1"),), layers={4: layer})
    graph = prefill_graph((4,))
    sensors = ("residual.4.last_input", "residual.4.tool_mean")
    outputs = graph.evaluate(ctx, sensors)
    assert capture_plan(ctx, sensors, graph)[0][1][0][1] == (3,)
    layer[:] = 0
    np.testing.assert_equal(outputs[sensors[0]], [9, 10, 11])
    np.testing.assert_equal(outputs[sensors[1]], [4.5, 5.5, 6.5])


def test_byte_budget_includes_running_item():
    queue = BudgetQueue(2, 10)
    assert queue.put("first", 8)
    item, size = queue.get()
    assert item == "first"
    assert not queue.put("too big", 3)
    assert queue.put("second", 2)
    assert not queue.put("too many", 0)
    queue.release(size)
    assert queue.put("now fits", 8)
    assert queue.peak_bytes == 10


def test_artifact_version_compatibility_and_no_missing_zero(tmp_path):
    compat = Compatibility("fixture", "revision", "bfloat16", "tokenizer", "template", "site")
    path = tmp_path / "probe"
    detector = export_detector(path, name="probe", sensor="residual.4.last_input", compatibility=compat,
                               coefficient=[2, -1], bias=[0.5], mean=[1, 1], scale=[2, 1],
                               threshold=0.25, training={"fixture": True})
    assert detector.evaluate({}).value is None
    assert detector.evaluate({detector.sensors[0]: [3, 1]}).value == 2.5
    with pytest.raises(ValueError, match="precision"):
        load_detector(path, replace(compat, precision="float16"))
    with (path / "weights.safetensors").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_detector(path)


def test_wire_vector_owned_and_bounded():
    vector = np.array([3.5, 5.2], dtype=np.float32)
    encoded = encode_vector(vector)
    value = decode_vector(encoded)
    vector[:] = 0
    np.testing.assert_allclose(value, [3.5, 5.2])
    with pytest.raises(ValueError, match="oversized"):
        encode_vector(np.zeros(16385, dtype=np.float32))


def test_prebranding_detector_bundles_still_load_without_migration(tmp_path):
    compatibility = Compatibility("fixture", "revision", "bfloat16", "tokenizer", "template", "site")
    directory = tmp_path / "probe"
    current = export_detector(directory, name="probe", sensor="residual.4.last_input", compatibility=compatibility,
                              coefficient=[2, -1], bias=[0.5], mean=[1, 1], scale=[2, 1],
                              threshold=0.25, training={"fixture": True})
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["implementation"] == "iceblink.linear.v1"
    manifest["implementation"] = "scintilla.linear.v1"
    manifest["artifact_id"] = identity({k: v for k, v in manifest.items() if k != "artifact_id"})
    payload = json.dumps(manifest, indent=2) + "\n"
    (directory / "manifest.json").write_text(payload)
    legacy = load_detector(directory, compatibility)
    assert legacy.version == manifest["artifact_id"]
    assert legacy.evaluate({legacy.sensors[0]: [3, 1]}).value == current.evaluate({current.sensors[0]: [3, 1]}).value
    assert (directory / "manifest.json").read_text() == payload
