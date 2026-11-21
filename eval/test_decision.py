from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import decide
import decision_scenarios as ds
import decision_score

KEYS = {
    "s1": {
        "family": "interaction",
        "options": ["voice1", "voice2", "voice3", "no-change"],
        "defect": "voice2",
        "control": "no-change",
    }
}


def record(condition, repeat, decision, render="defect", scenario="s1", **cost):
    return {
        "scenario": scenario,
        "render": render,
        "condition": condition,
        "repeat": repeat,
        "decision": decision,
        "commands": [],
        "cost": {
            "caller_calls": 1,
            "listening_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "seconds": 1.0,
        }
        | cost,
    }


def test_change_table_separates_change_from_improvement():
    records = [
        record("source", 0, "voice1"),
        record("source+listening", 0, "voice2"),
        record("source", 1, "voice2"),
        record("source+listening", 1, "voice3"),
        record("source", 2, "voice1"),
        record("source+listening", 2, "voice3"),
        record("source", 3, "voice2"),
        record("source+listening", 3, "voice2"),
    ]
    report = decision_score.score(records, KEYS)
    table = report["families"]["interaction"]["change_vs_source"]["source+listening"]
    assert table["cells"] == {
        "wrong_to_right": 1,
        "right_to_wrong": 1,
        "wrong_to_wrong": 1,
        "unchanged": 1,
        "incomplete": 0,
    }
    conditions = report["families"]["interaction"]["conditions"]
    assert conditions["source"]["defect_correct"]["k"] == 2
    assert conditions["source+listening"]["defect_correct"]["k"] == 2
    assert table["difference"]["difference"] == 0.0


def test_control_false_change_is_a_valid_decision_that_departs_from_the_control_key():
    records = [
        record("source+listening", 0, "no-change", render="control"),
        record("source+listening", 1, "voice2", render="control"),
        record("source+listening", 2, None, render="control"),
    ]
    control = decision_score.score(records, KEYS)["families"]["interaction"]["conditions"]["source+listening"]
    assert (control["control_false_change"]["k"], control["control_false_change"]["n"]) == (1, 3)
    assert control["invalid"] == 1


def test_malformed_or_missing_decisions_are_counted_never_dropped():
    records = [
        record("source", 0, None),
        record("source", 1, "raise everything"),
        record("source", 2, "voice2"),
        record("source+listening", 0, "voice2"),
    ]
    family = decision_score.score(records, KEYS)["families"]["interaction"]
    assert family["conditions"]["source"]["invalid"] == 2
    assert (
        family["conditions"]["source"]["defect_correct"]["k"],
        family["conditions"]["source"]["defect_correct"]["n"],
    ) == (1, 3)
    cells = family["change_vs_source"]["source+listening"]["cells"]
    assert cells["incomplete"] == 3 and sum(cells.values()) == 3


def test_interval_edges_and_empty_groups():
    none_right = decision_score.score([record("source", r, "voice1") for r in range(4)], KEYS)
    rate = none_right["families"]["interaction"]["conditions"]["source"]["defect_correct"]
    assert rate["rate"] == 0.0 and rate["ci95"][0] == 0.0 and 0 < rate["ci95"][1] < 1
    all_right = decision_score.score([record("source", r, "voice2") for r in range(4)], KEYS)
    rate = all_right["families"]["interaction"]["conditions"]["source"]["defect_correct"]
    assert rate["rate"] == 1.0 and rate["ci95"][1] == 1.0 and 0 < rate["ci95"][0] < 1
    assert all_right["families"]["interaction"]["conditions"]["source"]["control_false_change"]["n"] == 0


def test_cost_per_decision_is_reported_per_condition():
    records = [
        record("source", 0, "voice2", seconds=2.0),
        record(
            "source+listening",
            0,
            "voice2",
            listening_calls=3,
            input_tokens=1_000_000,
            output_tokens=200_000,
            seconds=40.0,
        ),
        record(
            "source+listening", 1, "voice2", listening_calls=1, input_tokens=0, output_tokens=0, seconds=20.0
        ),
    ]
    conditions = decision_score.score(records, KEYS)["families"]["interaction"]["conditions"]
    cost = conditions["source+listening"]["cost_per_decision"]
    assert cost["listening_calls"] == 2.0
    assert cost["seconds_median"] == 30.0
    assert cost["usd"] == 1.5  # (1.0 M input * 1.50 + 0.2 M output * 7.50) / 2 decisions
    assert conditions["source"]["cost_per_decision"]["listening_calls"] == 0.0


# --- scenario construction ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(ds.SCENARIOS))
def test_scenario_declares_everything_a_run_needs(name):
    scenario = ds.SCENARIOS[name]
    assert scenario["goal"]["attribution"] == "human" and scenario["goal"]["text"]
    assert scenario["family"] and scenario["provenance"]
    assert scenario["defect"] != scenario["control"]
    key = ds.answer_keys()[name]
    assert key["defect"] in key["options"] and key["defect"] != ds.NO_CHANGE
    assert key["control"] == ds.NO_CHANGE


@pytest.mark.parametrize("name", list(ds.SCENARIOS))
def test_defect_and_control_differ_in_sound_as_declared(name):
    scenario = ds.SCENARIOS[name]
    spec = scenario["construction"]
    assert ds.band_ratio_db(scenario, "defect") <= spec["defect_max_db"]
    assert ds.band_ratio_db(scenario, "control") >= spec["control_min_db"]
    assert np.max(np.abs(ds.mix(scenario, "defect"))) < 1 and np.max(np.abs(ds.mix(scenario, "control"))) < 1


def test_masked_voice_keeps_its_own_parameters_and_level_in_both_renders():
    scenario = ds.SCENARIOS["masked-melody"]
    changed = {k for k in scenario["defect"] if scenario["defect"][k] != scenario["control"][k]}
    assert not any(k.startswith("melody") for k in changed)
    defect, control = ds.stems(scenario, "defect"), ds.stems(scenario, "control")
    assert np.array_equal(defect["melody"], control["melody"])
    for voice in scenario["voices"]:  # no voice is louder or quieter overall: only the interaction differs
        assert np.isclose(np.std(defect[voice]), np.std(control[voice]), rtol=1e-3)


def test_source_shown_to_the_caller_contains_no_answer():
    scenario = ds.SCENARIOS["masked-melody"]
    for kind in ("defect", "control"):
        text = ds.source_text(scenario, kind)
        assert "padDrive" in text and "def trio" in text
        compile(text, "source", "exec")
        for name in ("SR", "_one_pole", "_unit", "_unit_rms"):
            assert f"{name} = " in text or f"def {name}(" in text
        for leak in ("defect", "control", "raise:", "masked", "answer"):
            assert leak not in text


# --- whole pipeline, fake provider, no model call ----------------------------------------------

CALLER = """#!{python}
import json, sys
from pathlib import Path
argv = sys.argv[1:]
body = sys.stdin.read()
with Path({log!r}).open("a") as fh:
    fh.write(json.dumps({{"prompt": argv[1], "body": body}}) + "\\n")
prompt = argv[1]
if "Evidence about the current render" not in prompt:
    reply = {{"target": "bass", "direction": "raise", "rationale": "guess from source"}}
elif '"claims": [{{' in prompt or "fake summary" in prompt:
    drive = float(prompt.split('"padDrive": ')[1].split("\\n")[0].rstrip(","))
    reply = (
        {{"target": "melody", "direction": "raise", "rationale": "heard"}}
        if drive > 1
        else {{"target": "none", "direction": "none", "rationale": "heard"}}
    )
else:
    reply = {{"target": "pad", "direction": "lower", "rationale": "not an allowed form"}}
print(json.dumps({{
    "steps": [{{"type": "model_output", "content": [{{"type": "text", "text": json.dumps(reply)}}]}}],
    "usage": {{"total_input_tokens": 50, "total_output_tokens": 10, "total_thought_tokens": 5}},
}}))
"""


def test_pipeline_runs_end_to_end_with_fake_providers(tmp_path):
    cli = Path(sys.executable).parent / "agent-listening"
    log = tmp_path / "caller.jsonl"
    caller = tmp_path / "caller"
    caller.write_text(CALLER.format(python=sys.executable, log=str(log)))
    caller.chmod(0o755)
    fake = Path(__file__).parent.parent / "tests" / "fake_gemini_api.py"

    report = decide.run(tmp_path / "run", str(cli), str(fake), str(caller), repeats=2)

    records = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    assert len(records) == 1 * 2 * 3 * 2
    assert json.loads((tmp_path / "run" / "report.json").read_text()) == report
    for r in records:
        assert {"target", "direction", "rationale", "decision", "commands", "cost"} <= set(r)
    by_condition = {c: [r for r in records if r["condition"] == c] for c in decide.CONDITIONS}
    assert all(r["commands"] == [] for r in by_condition["source"])
    for r in by_condition["source+measurement"]:
        assert [(c["command"], c["dry_run"], c["calls"]) for c in r["commands"]] == [("investigate", True, 0)]
    for r in by_condition["source+listening"]:
        assert [c["command"] for c in r["commands"]] == ["hear", "ask"]
        assert r["cost"]["listening_calls"] == decide.LISTENING_CALL_BUDGET

    prompts = [json.loads(line)["prompt"] for line in log.read_text().splitlines()]
    assert sum("centroid_hz" in p or "rms_dbfs" in p for p in prompts) == 4  # measurement runs only
    assert all("raise:melody" not in p and "defect" not in p for p in prompts)

    family = report["families"]["interaction"]
    assert family["conditions"]["source"]["defect_correct"]["k"] == 0
    assert family["conditions"]["source+measurement"]["invalid"] == 4
    assert family["conditions"]["source+listening"]["defect_correct"]["rate"] == 1.0
    assert family["conditions"]["source+listening"]["control_false_change"]["k"] == 0
    assert family["conditions"]["source"]["control_false_change"]["k"] == 2
    assert family["change_vs_source"]["source+listening"]["cells"]["wrong_to_right"] == 2
    assert family["change_vs_source"]["source+measurement"]["cells"]["incomplete"] == 2
    assert family["conditions"]["source+listening"]["listening_commands"] == {"ask": 4, "hear": 4}


def test_caller_usage_survives_a_reply_that_is_not_a_decision(tmp_path):
    caller = tmp_path / "caller"
    caller.write_text(
        f"#!{sys.executable}\nimport json\n"
        'print(json.dumps({"steps": [{"type": "model_output", "content": '
        '[{"type": "text", "text": "not json"}]}],'
        ' "usage": {"total_input_tokens": 50, "total_output_tokens": 10}}))\n'
    )
    caller.chmod(0o755)
    reply, usage = decide.ask_caller(str(caller), "prompt", {})
    assert reply is None and usage == {"total_input_tokens": 50, "total_output_tokens": 10}


def test_decisions_already_made_are_on_disk_when_a_later_one_fails(tmp_path, monkeypatch):
    made = []

    def flaky(name, kind, condition, repeat, *rest):
        if len(made) == 2:
            raise TimeoutError
        made.append(1)
        return {"scenario": name, "render": kind, "condition": condition, "repeat": repeat}

    monkeypatch.setattr(decide, "decide", flaky)
    with pytest.raises(TimeoutError):
        decide.run(tmp_path / "run", "cli", None, "caller", repeats=1)
    assert len((tmp_path / "run" / "decisions.jsonl").read_text().splitlines()) == 2


def test_a_run_never_overwrites_recorded_decisions(tmp_path):
    (tmp_path / "decisions.jsonl").write_text("kept\n")
    with pytest.raises(FileExistsError):
        decide.run(tmp_path, "cli", None, "caller", repeats=1)
    assert (tmp_path / "decisions.jsonl").read_text() == "kept\n"
