from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import decide
import decision_registration as reg
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

NAMES = list(ds.SCENARIOS)
HIDDEN_NAMES = [n for n in NAMES if ds.SCENARIOS[n]["family"] == "hidden"]
INTERACTION_NAMES = [n for n in NAMES if ds.SCENARIOS[n]["family"] == "interaction"]


def test_each_family_has_five_paired_scenarios_of_several_kinds():
    for names in (INTERACTION_NAMES, HIDDEN_NAMES):
        assert len(names) >= 5
        assert len({ds.SCENARIOS[n]["kind"] for n in names}) >= 4
    assert {"masking", "register", "resonance", "beating"} <= {
        ds.SCENARIOS[n]["kind"] for n in INTERACTION_NAMES
    }


@pytest.mark.parametrize("name", NAMES)
def test_scenario_declares_everything_a_run_needs(name):
    scenario = ds.SCENARIOS[name]
    assert scenario["goal"]["attribution"] == "human" and scenario["goal"]["text"]
    assert scenario["family"] and "not from the 1.0 corpus or holdout" in scenario["provenance"]
    assert scenario["defect"] != scenario["control"]
    key = ds.answer_keys()[name]
    assert len(set(key["options"])) == len(key["options"]) >= 3 and key["options"][-1] == ds.NO_CHANGE
    assert ds.chance(scenario) == 1 / len(key["options"])
    assert key["defect"] in key["options"] and key["defect"] != ds.NO_CHANGE
    assert key["control"] == ds.NO_CHANGE


@pytest.mark.parametrize("name", NAMES)
def test_renders_sound_the_way_the_recipe_and_answer_key_say(name):
    scenario = ds.SCENARIOS[name]
    assert ds.construction_failures(scenario) == []
    for kind in ("defect", "control"):
        assert np.max(np.abs(ds.mix(scenario, kind))) < 1


def test_construction_check_fails_when_the_answer_key_does_not_match_the_render():
    for name in ("masked-melody", "late-entry", "drifting-line"):
        scenario = ds.SCENARIOS[name]
        assert ds.construction_failures({**scenario, "defect": scenario["control"]})
    wrong = {**ds.SCENARIOS["late-entry"], "answer": {"defect": "earlier:low", "control": ds.NO_CHANGE}}
    assert ds.construction_failures(wrong)


@pytest.mark.parametrize("name", INTERACTION_NAMES)
def test_interaction_scenarios_differ_in_one_parameter_and_keep_every_voice_level(name):
    scenario = ds.SCENARIOS[name]
    changed = [k for k in scenario["defect"] if scenario["defect"][k] != scenario["control"][k]]
    assert len(changed) == 1
    if scenario["kind"] == "resonance":  # the shared room is the interaction; dry voices are identical
        dry = {"combFeedback": 0.0}
        defect, control = ds.stems(scenario, "defect", dry), ds.stems(scenario, "control", dry)
    else:
        defect, control = ds.stems(scenario, "defect"), ds.stems(scenario, "control")
    for voice in defect:
        assert np.isclose(np.std(defect[voice]), np.std(control[voice]), rtol=1e-3)


def test_masked_voice_is_identical_in_both_renders():
    for name, voice in (("masked-melody", "melody"), ("masked-bass", "bass")):
        scenario = ds.SCENARIOS[name]
        assert np.array_equal(ds.stems(scenario, "defect")[voice], ds.stems(scenario, "control")[voice])


@pytest.mark.parametrize("name", NAMES)
def test_source_shown_to_the_caller_is_complete_and_names_no_answer(name):
    scenario = ds.SCENARIOS[name]
    for kind in ("defect", "control"):
        text = ds.source_text(scenario, kind)
        namespace = {"np": np}
        exec(compile(text, "source", "exec"), namespace)
        shown = namespace["p"]
        assert shown == {k: v for k, v in scenario[kind].items() if k not in ds.HIDDEN}
        for leak in (
            "defect",
            "control",
            "answer",
            "construction",
            name,
            *scenario["options"][:-1],
        ):
            assert leak not in text


@pytest.mark.parametrize("name", HIDDEN_NAMES)
def test_hidden_state_is_fixed_reproducible_and_never_shown(name):
    scenario = ds.SCENARIOS[name]
    texts = [ds.source_text(scenario, kind) for kind in ("defect", "control")]
    assert texts[0] == texts[1]
    visible = {k: v for k, v in scenario["defect"].items() if k != "seed"}
    assert visible == {k: v for k, v in scenario["control"].items() if k != "seed"}
    for kind in ("defect", "control"):
        seed = scenario[kind]["seed"]
        assert isinstance(seed, int) and f'"seed": {seed}' not in texts[0] and '"seed":' not in texts[0]
        assert np.array_equal(ds.mix(scenario, kind), ds.mix(scenario, kind))
    assert not np.array_equal(ds.mix(scenario, "defect"), ds.mix(scenario, "control"))
    prompt = decide.prompt(scenario, texts[0], [])
    assert f'"seed": {scenario["defect"]["seed"]}' not in prompt


# --- registration, run record, calibration gate ------------------------------------------------


def test_registered_configuration_is_what_the_code_runs():
    registered = reg.load()
    assert registered == reg.current()
    assert list(registered["scenarios"]) == NAMES
    for key in ("repeats", "listening_call_budget", "calibration_margin", "primary_comparison", "caller"):
        assert key in registered
    assert registered["primary_comparison"] == {
        "metric": "defect_correct",
        "condition": "source+listening",
        "baseline": "source",
        "renders": "defect",
        "guard": "control_false_change",
    }
    for scenario in registered["scenarios"].values():
        assert {"options", "chance", "answer", "recipe_sha256"} <= set(scenario)


def test_recipe_hash_covers_hidden_state():
    scenario = ds.SCENARIOS["late-entry"]
    moved = {**scenario, "defect": {**scenario["defect"], "seed": scenario["defect"]["seed"] + 1}}
    assert reg.recipe_hash(moved, "defect") != reg.recipe_hash(scenario, "defect")
    assert reg.config_hash(reg.current({"x": moved})) != reg.config_hash(reg.current({"x": scenario}))


def source_records(name, defect_decisions):
    return [record("source", i, d, scenario=name) for i, d in enumerate(defect_decisions)] + [
        record("source", i, "no-change", render="control", scenario=name)
        for i in range(len(defect_decisions))
    ]


def test_calibration_gate_admits_only_scenarios_near_chance_and_lists_the_rest():
    registered = reg.current({n: ds.SCENARIOS[n] for n in ("masked-melody", "beating-chime", "late-entry")})
    assert registered["calibration_margin"] == 0.15
    records = (
        source_records("masked-melody", ["raise:melody"] + ["no-change"] * 4 + ["raise:pad"] * 5)  # 0.10
        + source_records("masked-melody", ["raise:melody"] * 3)[:3]  # now 4/13 = 0.31 <= 0.25 + 0.15
        + source_records("beating-chime", ["retune:chime"] * 5 + ["no-change"] * 5)  # 0.50 > 0.33 + 0.15
        + [record("source+listening", 0, "earlier:mid", scenario="late-entry")]  # no source-only decision
    )
    gate = reg.calibrate(records, registered)
    assert [e["scenario"] for e in gate["admitted"]] == ["masked-melody"]
    assert gate["admitted"][0]["defect_correct"] == {"k": 4, "n": 13, "rate": 0.3077}
    assert {e["scenario"]: e["reason"] for e in gate["excluded"]} == {
        "beating-chime": "source-only correctness above chance + margin",
        "late-entry": "no source-only decision on the defect render",
    }


def test_calibration_gate_margin_boundary():
    registered = reg.current({"masked-melody": ds.SCENARIOS["masked-melody"]})  # chance 0.25, limit 0.40
    inside = reg.calibrate(
        source_records("masked-melody", ["raise:melody"] * 2 + ["no-change"] * 3), registered
    )
    outside = reg.calibrate(
        source_records("masked-melody", ["raise:melody"] * 9 + ["no-change"] * 11), registered
    )
    assert len(inside["admitted"]) == 1 and inside["admitted"][0]["defect_correct"]["rate"] == 0.4
    assert len(outside["excluded"]) == 1 and outside["excluded"][0]["defect_correct"]["rate"] == 0.45


# --- whole pipeline, fake provider, no model call ----------------------------------------------

CALLER = """#!{python}
import json, sys
from pathlib import Path
argv = sys.argv[1:]
body = sys.stdin.read() if "--body" in argv else ""
with Path({log!r}).open("a") as fh:
    fh.write(json.dumps({{"argv": argv, "body": body}}) + "\\n")
if argv == ["--version"]:
    print("gemini-api {version}")
    sys.exit(0)
prompt = argv[1]
options = prompt.split("exactly one of: ")[1].rstrip(".").split(", ")
if "Evidence about the current render" not in prompt:
    reply = {{"decision": options[0], "rationale": "guess from source"}}
elif '"claims": [{{' in prompt or "fake summary" in prompt:
    reply = {{"decision": "raise:melody" if '"padDrive": 6.0' in prompt else "no-change", "rationale": "heard"}}
else:
    reply = {{"decision": "lower:pad", "rationale": "not an allowed option"}}
print(json.dumps({{
    "steps": [{{"type": "model_output", "content": [{{"type": "text", "text": json.dumps(reply)}}]}}],
    "usage": {{"total_input_tokens": 50, "total_output_tokens": 10, "total_thought_tokens": 5}},
}}))
"""
CLI = str(Path(sys.executable).parent / "agent-listening")
FAKE = str(Path(__file__).parent.parent / "tests" / "fake_gemini_api.py")


@pytest.fixture
def one_scenario(monkeypatch):
    scenarios = {"masked-melody": ds.SCENARIOS["masked-melody"]}
    monkeypatch.setattr(ds, "SCENARIOS", scenarios)
    monkeypatch.setattr(reg, "load", lambda: reg.current(scenarios))
    return scenarios


@pytest.fixture
def caller(tmp_path):
    def make(version="0.4.5"):
        path = tmp_path / f"caller-{version}"
        path.write_text(
            CALLER.format(python=sys.executable, log=str(tmp_path / "caller.jsonl"), version=version)
        )
        path.chmod(0o755)
        return str(path)

    return make


def model_calls(tmp_path):
    """Every provider invocation other than --version, from the caller fake and the listening fake."""
    calls = []
    for log in (tmp_path / "caller.jsonl", tmp_path / "gemini.jsonl"):
        if log.exists():
            calls += [c for c in map(json.loads, log.read_text().splitlines()) if c["argv"] != ["--version"]]
    return calls


def test_pipeline_runs_end_to_end_with_fake_providers(tmp_path, one_scenario, caller, monkeypatch):
    monkeypatch.setenv("FAKE_GEMINI_LOG", str(tmp_path / "gemini.jsonl"))
    report = decide.run(tmp_path / "run", CLI, FAKE, caller())

    records = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    assert len(records) == 1 * 2 * 3 * 3
    assert json.loads((tmp_path / "run" / "report.json").read_text()) == report
    for r in records:
        assert {"reply", "rationale", "decision", "commands", "cost"} <= set(r)
    by_condition = {c: [r for r in records if r["condition"] == c] for c in decide.CONDITIONS}
    assert all(r["commands"] == [] for r in by_condition["source"])
    for r in by_condition["source+measurement"]:
        assert [(c["command"], c["dry_run"], c["calls"]) for c in r["commands"]] == [("investigate", True, 0)]
    for r in by_condition["source+listening"]:
        assert [c["command"] for c in r["commands"]] == ["hear", "ask"]
        assert r["cost"]["listening_calls"] == decide.LISTENING_CALL_BUDGET

    calls = [
        c
        for c in map(json.loads, (tmp_path / "caller.jsonl").read_text().splitlines())
        if "--system" in c["argv"]
    ]
    prompts = [c["argv"][1] for c in calls]
    assert len(prompts) == 18
    assert sum("centroid_hz" in p or "rms_dbfs" in p for p in prompts) == 6  # measurement runs only
    assert all("defect" not in p and "masked" not in p for p in prompts)
    fixed = {(tuple(c["argv"][2:]), c["body"]) for c in calls}
    assert len(fixed) == 1  # same model, system prompt, thinking level and schema in every condition
    argv, body = next(iter(fixed))
    assert argv[argv.index("-m") + 1] == reg.CALLER["model"]
    assert argv[argv.index("--system") + 1] == reg.CALLER["system"]
    assert json.loads(body)["generation_config"] == {"thinking_level": reg.CALLER["thinking_level"]}

    family = report["families"]["interaction"]
    assert family["conditions"]["source"]["defect_correct"]["k"] == 0
    assert family["conditions"]["source+measurement"]["invalid"] == 6
    assert family["conditions"]["source+listening"]["defect_correct"]["rate"] == 1.0
    assert family["conditions"]["source+listening"]["control_false_change"]["k"] == 0
    assert family["conditions"]["source"]["control_false_change"]["k"] == 3
    assert family["change_vs_source"]["source+listening"]["cells"]["wrong_to_right"] == 3
    assert family["change_vs_source"]["source+measurement"]["cells"]["incomplete"] == 3
    assert family["conditions"]["source+listening"]["listening_commands"] == {"ask": 6, "hear": 6}
    assert [e["scenario"] for e in report["calibration"]["admitted"]] == ["masked-melody"]


def test_run_records_what_it_ran_on(tmp_path, one_scenario, caller):
    decide.run(tmp_path / "run", CLI, FAKE, caller(), conditions=("source",))
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["configuration_sha256"] == reg.config_hash(reg.current(one_scenario))
    assert manifest["tool_version"] == "1.0.0" and manifest["provider_cli_version"] == "0.4.5"
    assert manifest["model"] == {"caller": "gemini-4-hydrogen", "listening": "gemini-4-hydrogen"}
    assert manifest["conditions"] == ["source"]
    for kind in ("defect", "control"):
        audio = tmp_path / "run" / "audio" / f"masked-melody.{kind}.wav"
        assert manifest["materials_sha256"]["masked-melody"][kind] == reg.sha256(audio.read_bytes())
    records = (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()
    assert len(records) == 6 and all(json.loads(r)["condition"] == "source" for r in records)


def refused(tmp_path, monkeypatch, match, **kwargs):
    monkeypatch.setenv("FAKE_GEMINI_LOG", str(tmp_path / "gemini.jsonl"))
    with pytest.raises(reg.Mismatch, match=match):
        decide.run(tmp_path / "run", **kwargs)
    assert model_calls(tmp_path) == []
    assert not (tmp_path / "run" / "decisions.jsonl").exists()


def test_configuration_mismatch_stops_before_any_provider_call(tmp_path, one_scenario, caller, monkeypatch):
    registered = reg.current(one_scenario)
    registered["scenarios"]["masked-melody"]["answer"] = {"defect": "raise:pad", "control": "no-change"}
    monkeypatch.setattr(reg, "load", lambda: registered)
    refused(tmp_path, monkeypatch, "configuration mismatch", cli=CLI, gemini_api=FAKE, caller_api=caller())


def test_recipe_change_stops_before_any_provider_call(tmp_path, one_scenario, caller, monkeypatch):
    registered = reg.current(one_scenario)
    changed = {
        **one_scenario["masked-melody"],
        "defect": {**one_scenario["masked-melody"]["defect"], "padDrive": 5.0},
    }
    monkeypatch.setattr(ds, "SCENARIOS", {"masked-melody": changed})
    monkeypatch.setattr(reg, "load", lambda: registered)
    refused(tmp_path, monkeypatch, "configuration mismatch", cli=CLI, gemini_api=FAKE, caller_api=caller())


def test_repeats_other_than_registered_stop_the_run(tmp_path, one_scenario, caller, monkeypatch):
    refused(tmp_path, monkeypatch, "repeats", cli=CLI, gemini_api=FAKE, caller_api=caller(), repeats=1)


def test_version_mismatch_stops_before_any_provider_call(tmp_path, one_scenario, caller, monkeypatch):
    refused(tmp_path, monkeypatch, "caller_cli 0.5.0", cli=CLI, gemini_api=FAKE, caller_api=caller("0.5.0"))
    monkeypatch.setenv("FAKE_GEMINI_VERSION", "0.3.9")
    refused(tmp_path, monkeypatch, "provider_cli 0.3.9", cli=CLI, gemini_api=FAKE, caller_api=caller())
    monkeypatch.delenv("FAKE_GEMINI_VERSION")
    old_cli = tmp_path / "old-cli"
    old_cli.write_text(f"#!{sys.executable}\nprint('agent-listening 0.2.0')\n")
    old_cli.chmod(0o755)
    refused(
        tmp_path, monkeypatch, "agent-listening 0.2.0", cli=str(old_cli), gemini_api=FAKE, caller_api=caller()
    )


def test_material_mismatch_stops_before_any_provider_call(tmp_path, one_scenario, caller, monkeypatch):
    stale = tmp_path / "run" / "audio" / "masked-melody.defect.wav"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(reg.wav_bytes(one_scenario["masked-melody"], "control"))
    refused(tmp_path, monkeypatch, "material mismatch", cli=CLI, gemini_api=FAKE, caller_api=caller())


def test_listening_call_budget_is_enforced_per_decision(tmp_path, one_scenario, monkeypatch):
    asked = []
    monkeypatch.setattr(decide, "ask_caller", lambda *a: asked.append(a) or ({}, {}))

    three = [["hear", "a.wav"], ["ask", "a.wav", "q"], ["ask", "a.wav", "q2"]]
    ran = []
    monkeypatch.setattr(decide, "listening_plan", lambda *a: three)
    monkeypatch.setattr(decide, "collect", lambda *a: ran.append(a) or ([], []))
    with pytest.raises(RuntimeError, match="plan exceeds the listening call budget"):
        decide.decide("masked-melody", "defect", "source+listening", 0, Path("a.wav"), "cli", "api", "caller")
    assert ran == [] and asked == []

    monkeypatch.setattr(decide, "listening_plan", lambda *a: three[:1])
    over = [{"command": "hear", "calls": decide.LISTENING_CALL_BUDGET + 1, "usage": {}}]
    monkeypatch.setattr(decide, "collect", lambda *a: ([{}], over))
    with pytest.raises(RuntimeError, match="listening calls exceed the budget"):
        decide.decide("masked-melody", "defect", "source+listening", 0, Path("a.wav"), "cli", "api", "caller")
    assert asked == []


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


def test_decisions_already_made_are_on_disk_when_a_later_one_fails(tmp_path, one_scenario, monkeypatch):
    made = []

    def flaky(name, kind, condition, repeat, *rest):
        if len(made) == 2:
            raise TimeoutError
        made.append(1)
        return {"scenario": name, "render": kind, "condition": condition, "repeat": repeat}

    monkeypatch.setattr(decide, "decide", flaky)
    monkeypatch.setattr(reg, "preflight", lambda *a: {})
    (tmp_path / "run").mkdir()
    with pytest.raises(TimeoutError):
        decide.run(tmp_path / "run", "cli", "api", "caller")
    assert len((tmp_path / "run" / "decisions.jsonl").read_text().splitlines()) == 2


def test_a_run_never_overwrites_recorded_decisions(tmp_path):
    (tmp_path / "decisions.jsonl").write_text("kept\n")
    with pytest.raises(FileExistsError):
        decide.run(tmp_path, "cli", "api", "caller")
    assert (tmp_path / "decisions.jsonl").read_text() == "kept\n"
