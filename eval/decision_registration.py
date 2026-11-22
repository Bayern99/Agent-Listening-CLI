# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "soundfile>=0.12,<0.14", "scipy>=1.13,<2"]
# ///
"""Registered configuration of the decision-change evaluation, the run record, and the calibration gate.

`current()` is what the code in this checkout would run. `decision_registration.json` is what was
registered. A run starts only when the two are equal and the tools, model and audio match; nothing
here calls a provider beyond `--version`.

Usage: uv run eval/decision_registration.py --write        (re-register; a reviewed commit)
       uv run eval/decision_registration.py --calibrate RUN (gate a source-only run)
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import decision_scenarios as ds

REGISTERED = Path(__file__).with_name("decision_registration.json")

CONDITIONS = ("source", "source+measurement", "source+listening")
RENDERS = ("defect", "control")
REPEATS = 3
LISTENING_CALL_BUDGET = 2
CALIBRATION_MARGIN = 0.15
CALLER = {
    "model": "gemini-4-hydrogen",
    "thinking_level": "high",
    "system": "You decide the next revision of a piece from its source. You cannot hear the audio yourself.",
}
TOOLS = {"agent_listening": "1.0.0", "provider_cli_series": "0.4", "listening_model": "gemini-4-hydrogen"}
PRIMARY = {
    "metric": "defect_correct",
    "condition": "source+listening",
    "baseline": "source",
    "renders": "defect",
    "guard": "control_false_change",
}


_VERSION = re.compile(r"\d+\.\d+\.\d+")


class Mismatch(RuntimeError):
    """The run differs from what was registered. Raised before any model call."""


def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def recipe_hash(scenario: dict, render: str) -> str:
    """Identity of a render's recipe: its full parameters, hidden ones included, and the code that renders it."""
    renderer, helpers = ds.RENDERERS[scenario["renderer"]]
    code = "\n".join(inspect.getsource(f) for f in (*helpers, renderer, ds.stems, ds.mix))
    return sha256(_canonical({"sr": ds.SR, "params": scenario[render], "code": code}))


def current(scenarios: dict | None = None) -> dict:
    scenarios = ds.SCENARIOS if scenarios is None else scenarios
    return {
        "conditions": list(CONDITIONS),
        "renders": list(RENDERS),
        "repeats": REPEATS,
        "listening_call_budget": LISTENING_CALL_BUDGET,
        "calibration_margin": CALIBRATION_MARGIN,
        "caller": CALLER,
        "tools": TOOLS,
        "primary_comparison": PRIMARY,
        "scenarios": {
            name: {
                "family": s["family"],
                "kind": s["kind"],
                "options": s["options"],
                "chance": round(ds.chance(s), 4),
                "answer": s["answer"],
                "goal": s["goal"],
                "listening_question": s["listening_question"],
                "recipe_sha256": {render: recipe_hash(s, render) for render in RENDERS},
            }
            for name, s in scenarios.items()
        },
    }


def config_hash(config: dict) -> str:
    return sha256(_canonical(config))


def load() -> dict:
    return json.loads(REGISTERED.read_text(encoding="utf-8"))


def _version(executable: str) -> str:
    proc = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=30, check=False)
    match = _VERSION.search(proc.stdout)
    if proc.returncode != 0 or match is None:
        raise Mismatch(f"{executable} --version failed")
    return match.group()


def wav_bytes(scenario: dict, render: str) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, ds.mix(scenario, render), ds.SR, subtype="PCM_24", format="WAV")
    return buffer.getvalue()


def materials(audio_dir: Path, scenarios: dict) -> dict:
    """Render every input, refusing audio already on disk that is not what the recipe renders now."""
    hashes: dict = {}
    for name, scenario in scenarios.items():
        for render in RENDERS:
            data = wav_bytes(scenario, render)
            path = audio_dir / f"{name}.{render}.wav"
            if path.exists() and path.read_bytes() != data:
                raise Mismatch(f"material mismatch: {path} is not the registered recipe's render")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            hashes.setdefault(name, {})[render] = sha256(data)
    return hashes


def preflight(
    out_dir: Path,
    cli: str,
    gemini_api: str,
    caller_api: str,
    repeats: int,
    registered: dict | None = None,
    scenarios: dict | None = None,
) -> dict:
    """Check the run against the registration and return its record. Makes no model call."""
    registered = load() if registered is None else registered
    scenarios = ds.SCENARIOS if scenarios is None else scenarios
    now = current(scenarios)
    if now != registered:
        differing = sorted(k for k in {*now, *registered} if now.get(k) != registered.get(k))
        raise Mismatch(f"configuration mismatch against {REGISTERED.name}: {', '.join(differing)}")
    if repeats != registered["repeats"]:
        raise Mismatch(f"repeats {repeats} != registered {registered['repeats']}")
    tools = registered["tools"]
    versions = {
        "agent_listening": _version(cli),
        "provider_cli": _version(gemini_api),
        "caller_cli": _version(caller_api),
    }
    if versions["agent_listening"] != tools["agent_listening"]:
        raise Mismatch(
            f"agent-listening {versions['agent_listening']} != registered {tools['agent_listening']}"
        )
    for key in ("provider_cli", "caller_cli"):
        if versions[key].split(".")[:2] != tools["provider_cli_series"].split("."):
            raise Mismatch(f"{key} {versions[key]} is not registered series {tools['provider_cli_series']}.x")
    return {
        "configuration_sha256": config_hash(registered),
        "tool_version": versions["agent_listening"],
        "provider_cli_version": versions["provider_cli"],
        "caller_cli_version": versions["caller_cli"],
        "model": {"caller": registered["caller"]["model"], "listening": tools["listening_model"]},
        "materials_sha256": materials(out_dir / "audio", scenarios),
    }


def calibrate(records: list[dict], registered: dict) -> dict:
    """Admit a scenario only when the source-only caller is not above chance on its defect render.

    One-sided on purpose: a caller that never finds the defect from source is exactly the case the
    evaluation needs, so only correctness above chance + margin excludes. A scenario with no
    valid source-only defect decision is excluded.
    """
    margin = registered["calibration_margin"]
    admitted, excluded = [], []
    for name, scenario in registered["scenarios"].items():
        rows = [r for r in records if r["scenario"] == name and r["condition"] == "source"]
        defect = [r for r in rows if r["render"] == "defect"]
        control = [r for r in rows if r["render"] == "control"]
        k = sum(r.get("decision") == scenario["answer"]["defect"] for r in defect)
        valid = sum(r.get("decision") in scenario["options"] for r in defect)
        entry = {
            "scenario": name,
            "family": scenario["family"],
            "chance": scenario["chance"],
            "limit": round(scenario["chance"] + margin, 4),
            "defect_valid": valid,
            "defect_correct": {
                "k": k,
                "n": len(defect),
                "rate": round(k / len(defect), 4) if defect else None,
            },
            "control_correct": {
                "k": sum(r.get("decision") == scenario["answer"]["control"] for r in control),
                "n": len(control),
            },
        }
        if not valid:
            excluded.append({**entry, "reason": "no valid source-only decision on the defect render"})
        elif k / len(defect) > scenario["chance"] + margin:
            excluded.append({**entry, "reason": "source-only correctness above chance + margin"})
        else:
            admitted.append(entry)
    return {"margin": margin, "admitted": admitted, "excluded": excluded}


def calibrate_run(run_dir: Path, registered: dict) -> dict:
    """Gate a saved run, only against the registration it was run under."""
    ran_under = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["configuration_sha256"]
    if ran_under != config_hash(registered):
        raise Mismatch(f"{run_dir.name} was run under configuration {ran_under[:12]}, not the registered one")
    lines = (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    return calibrate([json.loads(line) for line in lines], registered)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--calibrate", metavar="RUN")
    args = parser.parse_args()
    if args.write:
        REGISTERED.write_text(json.dumps(current(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(config_hash(current()))
        return
    from build import EVAL_DIR

    run_dir = EVAL_DIR / "runs" / args.calibrate
    print(json.dumps(calibrate_run(run_dir, load()), indent=2))


if __name__ == "__main__":
    main()
