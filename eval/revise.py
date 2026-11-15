# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "soundfile>=0.12,<0.14", "scipy>=1.13,<2"]
# ///
"""Revision-usefulness pilot.

Each scenario renders the canon with one defect against a composer goal. A text-only reviser
(gemini-4-hydrogen via gemini-api) proposes new parameters from the source alone ("code") or from
the source plus `agent-listening review` evidence of the render ("code+listening"). Success is
judged only by the pre-registered target range for the defect parameter, never by a model.

Usage: uv run eval/revise.py --run revision-1 --cli /path/to/agent-listening [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import corpus
from build import EVAL_DIR, build
from score import newcombe, wilson

SCENARIOS = {
    "buried-melody": {
        "goal": "The high bell-like melody that opens the piece must stay clearly in front of the other two voices.",
        "defect": {"v1Amp": 0.05},
        "target": {"v1Amp": [0.15, None]},
        "control": {"v1Amp": 0.3},
    },
    "detuned-bass": {
        "goal": "All three voices should sound in tune with one another.",
        "defect": {"v3Detune": 45},
        "target": {"v3Detune": [-10, 10]},
        "control": {},
    },
    "collapsed-entries": {
        "goal": "The three voices should enter one after another as a clear canon, each entry audible on its own.",
        "defect": {"v2Delay": 0.0, "v3Delay": 0.0},
        "target": {"v2Delay": [1.0, None], "v3Delay_minus_v2Delay": [1.0, None]},
        "control": {},
    },
    "struck-bell": {
        "goal": "The high voice should swell in gently, without a hard strike at the start of each note.",
        "defect": {"v1Attack": 0.002},
        "target": {"v1Attack": [0.05, None]},
        "control": {"v1Attack": 0.12},
    },
    "whistling-middle": {
        "goal": "The middle voice should sound warm and smooth, without a whistling resonance.",
        "defect": {"v2Res": 3.0},
        "target": {"v2Res": [None, 0.5]},
        "control": {},
    },
}

SYSTEM = "You revise SuperCollider compositions. You cannot hear the audio yourself."


def in_range(value: float, bounds: list[float | None]) -> bool:
    low, high = bounds
    return (low is None or value >= low) and (high is None or value <= high)


def success(params: dict, target: dict) -> bool:
    values = dict(params)
    values["v3Delay_minus_v2Delay"] = params["v3Delay"] - params["v2Delay"]
    return all(in_range(values[key], bounds) for key, bounds in target.items())


def collateral(params: dict, defect_params: dict, target: dict) -> list[str]:
    touched = {k for k in target} | ({"v2Delay", "v3Delay"} if "v3Delay_minus_v2Delay" in target else set())
    return [
        k
        for k, v in defect_params.items()
        if k not in touched and abs(params[k] - v) > max(0.1 * abs(v), 1e-6)
    ]


def compact(evidence: dict) -> dict:
    return {
        "summary": evidence.get("summary"),
        "assessments": evidence.get("assessments"),
        "claims": [
            {
                "text": c["text"],
                "aspect": c["aspect"],
                "times": [[loc["start"], loc["end"]] for loc in c["loci"]],
                "confidence": c["confidence"],
            }
            for c in evidence.get("claims", [])
        ],
    }


def revise(goal: str, source: str, evidence: dict | None, env: dict) -> dict:
    keys = list(corpus.SC_BASE)
    block = ""
    if evidence is not None:
        block = (
            "\nListening evidence about the current render from an audio-listening tool (claims are "
            "model-generated and may be wrong; times in seconds):\n```json\n"
            f"{json.dumps(compact(evidence), ensure_ascii=False)}\n```\n"
        )
    prompt = (
        f"Artistic goal from the composer: {goal}\n\nCurrent source (the parameter event ~p at the top "
        f"controls the piece):\n```supercollider\n{source}\n```\n{block}\nPropose revised values for every "
        "key in ~p so that the next render better meets the goal. Change only what the goal requires. "
        "Return every key."
    )
    schema = {
        "type": "object",
        "properties": {
            "params": {
                "type": "object",
                "properties": {k: {"type": "number"} for k in keys},
                "required": keys,
            },
            "rationale": {"type": "string"},
        },
        "required": ["params", "rationale"],
    }
    body = {
        "generation_config": {"thinking_level": "high"},
        "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
    }
    proc = subprocess.run(
        [
            "gemini-api",
            "generate",
            prompt,
            "-m",
            "gemini-4-hydrogen",
            "--system",
            SYSTEM,
            "--body",
            "@-",
            "--stream=false",
            "-o",
            "json",
            "--agent-mode",
        ],
        input=json.dumps(body),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=True,
    )
    reply = json.loads(proc.stdout)
    text = "".join(
        part["text"]
        for step in reply["steps"]
        if step.get("type") == "model_output"
        for part in step["content"]
        if part.get("type") == "text"
    )
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--cli", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    out_dir = EVAL_DIR / "runs" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["GEMINI_API_KEY"] = Path(os.environ["AGENT_LISTENING_EVAL_KEY_FILE"]).read_text().strip()

    prepared = {}
    controls: dict[str, dict] = {}
    for name, scenario in SCENARIOS.items():
        params = {**corpus.SC_BASE, **scenario["defect"]}
        wav = build({"op": "sc-canon", "params": params})
        brief = out_dir / f"{name}.brief.json"
        brief.write_text(json.dumps({"goals": [scenario["goal"]]}))
        evidence_path = out_dir / f"{name}.review.json"
        if not evidence_path.exists():
            subprocess.run(
                [args.cli, "review", str(wav), "--brief", str(brief), "--out", str(evidence_path)],
                env=env,
                check=False,
                capture_output=True,
                timeout=1800,
            )
        prepared[name] = (params, wav.with_suffix(".scd").read_text(), json.loads(evidence_path.read_text()))
        control_wav = build({"op": "sc-canon", "params": {**corpus.SC_BASE, **scenario["control"]}})
        control_path = out_dir / f"{name}.control-review.json"
        if not control_path.exists():
            subprocess.run(
                [args.cli, "review", str(control_wav), "--brief", str(brief), "--out", str(control_path)],
                env=env,
                check=False,
                capture_output=True,
                timeout=1800,
            )
        controls[name] = json.loads(control_path.read_text())

    jobs = [
        (name, condition, rep)
        for name in SCENARIOS
        for condition in ("code", "code+listening")
        for rep in range(args.repeats)
    ]

    def execute(job: tuple[str, str, int]) -> dict:
        name, condition, rep = job
        path = out_dir / f"{name}__{condition}__{rep}.json"
        if path.exists():
            return json.loads(path.read_text())
        params, source, evidence = prepared[name]
        try:
            reply = revise(SCENARIOS[name]["goal"], source, evidence if condition != "code" else None, env)
            result = {
                "scenario": name,
                "condition": condition,
                "repeat": rep,
                "params": reply["params"],
                "rationale": reply["rationale"],
                "success": success(reply["params"], SCENARIOS[name]["target"]),
                "collateral": collateral(reply["params"], params, SCENARIOS[name]["target"]),
            }
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as error:
            result = {"scenario": name, "condition": condition, "repeat": rep, "error": str(error)[:500]}
        path.write_text(json.dumps(result, indent=2))
        return result

    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(execute, jobs))

    valid = [r for r in results if "error" not in r]
    summary = {
        "scenarios": list(SCENARIOS),
        "repeats": args.repeats,
        "errors": [r for r in results if "error" in r],
        "review_status": {n: prepared[n][2]["status"] for n in SCENARIOS},
    }
    for condition in ("code", "code+listening"):
        group = [r for r in valid if r["condition"] == condition]
        summary[condition] = {
            "success": wilson(sum(r["success"] for r in group), len(group)),
            "success_without_collateral": wilson(
                sum(r["success"] and not r["collateral"] for r in group), len(group)
            ),
            "by_scenario": {n: [r["success"] for r in group if r["scenario"] == n] for n in SCENARIOS},
        }

    def verdicts(evidence: dict) -> list[str]:
        return [a["verdict"] for a in evidence.get("assessments", [])]

    summary["review_goal_verdicts"] = {
        n: {"defect": verdicts(prepared[n][2]), "control": verdicts(controls[n])} for n in SCENARIOS
    }
    flagged = [v in ("not-met", "partly") for n in SCENARIOS for v in verdicts(prepared[n][2])]
    false_alarm = [v in ("not-met", "partly") for n in SCENARIOS for v in verdicts(controls[n])]
    summary["review_defect_flagged"] = wilson(sum(flagged), len(flagged))
    summary["review_control_false_alarm"] = wilson(sum(false_alarm), len(false_alarm))
    summary["difference_listening_minus_code"] = newcombe(
        summary["code+listening"]["success"], summary["code"]["success"]
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
