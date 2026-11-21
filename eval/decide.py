# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "soundfile>=0.12,<0.14", "scipy>=1.13,<2"]
# ///
"""Decision-change evaluation: does listening evidence change, and improve, a caller's next decision?

The caller is a text agent that cannot hear. It reads a goal and the source, plus the evidence its
condition allows, and returns one closed-form decision. Evidence comes from the installed CLI only.

Usage: uv run eval/decide.py --run NAME --cli agent-listening [--gemini-api EXE] [--repeats N]
Spends model calls unless --gemini-api and --caller-api point at fakes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import decision_scenarios as ds
import decision_score
from build import EVAL_DIR

CONDITIONS = ("source", "source+measurement", "source+listening")
RENDERS = ("defect", "control")
CALLER_MODEL = "gemini-4-hydrogen"
LISTENING_CALL_BUDGET = 2
SYSTEM = "You decide the next revision of a piece from its source. You cannot hear the audio yourself."


def render(scenario_name: str, kind: str, out_dir: Path) -> Path:
    path = out_dir / f"{scenario_name}.{kind}.wav"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, ds.mix(ds.SCENARIOS[scenario_name], kind), ds.SR, subtype="PCM_24")
    return path


def listening_plan(scenario: dict, audio: Path, condition: str) -> list[list[str]]:
    """The CLI commands a condition may run. Fixed per condition so every run has the same budget."""
    span = f"0-{scenario['defect']['seconds']:g}"
    if condition == "source+measurement":
        return [["investigate", str(audio), "--focus", span, "--dry-run"]]
    if condition == "source+listening":
        return [["hear", str(audio)], ["ask", str(audio), scenario["listening_question"]]]
    return []


def collect(cli: str, plan: list[list[str]], gemini_api: str | None) -> tuple[list[dict], list[dict]]:
    """Run the plan through the installed CLI. Returns evidence documents and one log entry per command."""
    documents, log = [], []
    for command in plan:
        argv = [cli, *command, *(["--gemini-api", gemini_api] if gemini_api else [])]
        started = time.monotonic()
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=900, check=False)
        evidence = json.loads(proc.stdout)
        producer = evidence.get("producer") or {}
        documents.append(evidence)
        log.append(
            {
                "command": command[0],
                "dry_run": "--dry-run" in command,
                "status": evidence.get("status"),
                "calls": len(producer.get("interaction_ids") or []),
                "usage": producer.get("usage") or {},
                "seconds": round(time.monotonic() - started, 2),
            }
        )
    return documents, log


def compact(evidence: dict) -> dict:
    """What the caller sees of an evidence document; identity and transport detail are left out."""
    return {
        "task": evidence.get("task"),
        "status": evidence.get("status"),
        "error": evidence.get("error"),
        "summary": evidence.get("summary"),
        "answer": evidence.get("answer"),
        "claims": [
            {
                "text": c["text"],
                "aspect": c.get("aspect"),
                "times": [[loc["start"], loc["end"]] for loc in c.get("loci", [])],
                "status": c.get("status"),
                "confidence": c.get("confidence"),
            }
            for c in evidence.get("claims", [])
        ],
        "measurements": [
            {"kind": m["kind"], "result": {k: v for k, v in m["result"].items() if k != "windows"}}
            for m in evidence.get("measurements", [])
        ],
        "limitations": evidence.get("limitations"),
    }


def prompt(scenario: dict, source: str, documents: list[dict]) -> str:
    block = ""
    if documents:
        block = (
            "\nEvidence about the current render from an audio tool. Measurements are computed from the "
            "signal. Claims are model-generated and may be wrong. Times are in seconds.\n```json\n"
            f"{json.dumps([compact(d) for d in documents], ensure_ascii=False)}\n```\n"
        )
    return (
        f"Goal from the composer: {scenario['goal']['text']}\n\nCurrent source:\n```python\n{source}\n```\n"
        f"{block}\nDecide the next step. Set target to the one voice to bring forward and direction to "
        '"raise", or set both to "none" if nothing should change.'
    )


def schema(scenario: dict) -> dict:
    return {
        "type": "object",
        "properties": {
            "target": {"type": "string", "enum": [*scenario["voices"], "none"]},
            "direction": {"type": "string", "enum": ["raise", "none"]},
            "rationale": {"type": "string"},
        },
        "required": ["target", "direction", "rationale"],
    }


def decision(reply: dict | None) -> str | None:
    """Closed-form decision string, or None when the reply is not one of the allowed forms."""
    if not isinstance(reply, dict):
        return None
    target, direction = reply.get("target"), reply.get("direction")
    if target == "none" and direction == "none":
        return ds.NO_CHANGE
    if direction == "raise" and isinstance(target, str) and target != "none":
        return f"raise:{target}"
    return None


def ask_caller(caller_api: str, text: str, response_schema: dict) -> tuple[dict | None, dict]:
    body = {
        "generation_config": {"thinking_level": "high"},
        "response_format": {"type": "text", "mime_type": "application/json", "schema": response_schema},
    }
    argv = [caller_api, "generate", text, "-m", CALLER_MODEL, "--system", SYSTEM]
    argv += ["--body", "@-", "--stream=false", "-o", "json", "--agent-mode"]
    proc = subprocess.run(
        argv, input=json.dumps(body), capture_output=True, text=True, timeout=900, check=False
    )
    usage: dict = {}
    try:
        interaction = json.loads(proc.stdout)
        usage = interaction.get("usage") or {}
        output = "".join(
            part["text"]
            for step in interaction["steps"]
            if step.get("type") == "model_output"
            for part in step["content"]
            if part.get("type") == "text"
        )
        return json.loads(output), usage
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None, usage


def decide(
    name: str,
    kind: str,
    condition: str,
    repeat: int,
    audio: Path,
    cli: str,
    gemini_api: str | None,
    caller_api: str,
) -> dict:
    scenario = ds.SCENARIOS[name]
    started = time.monotonic()
    documents, commands = collect(cli, listening_plan(scenario, audio, condition), gemini_api)
    listening_calls = sum(c["calls"] for c in commands)
    if listening_calls > LISTENING_CALL_BUDGET:
        raise RuntimeError(f"{name}/{kind}/{condition}: {listening_calls} listening calls exceed the budget")
    reply, usage = ask_caller(
        caller_api, prompt(scenario, ds.source_text(scenario, kind), documents), schema(scenario)
    )
    listened = [c["usage"] for c in commands]
    return {
        "scenario": name,
        "render": kind,
        "condition": condition,
        "repeat": repeat,
        "target": reply.get("target") if isinstance(reply, dict) else None,
        "direction": reply.get("direction") if isinstance(reply, dict) else None,
        "rationale": reply.get("rationale") if isinstance(reply, dict) else None,
        "decision": decision(reply),
        "commands": commands,
        "cost": {
            "caller_calls": 1,
            "listening_calls": listening_calls,
            "input_tokens": int(usage.get("total_input_tokens") or 0)
            + sum(u.get("input_tokens", 0) for u in listened),
            "output_tokens": int(usage.get("total_output_tokens") or 0)
            + int(usage.get("total_thought_tokens") or 0)
            + sum(u.get("output_tokens", 0) + u.get("thought_tokens", 0) for u in listened),
            "seconds": round(time.monotonic() - started, 2),
        },
    }


def run(out_dir: Path, cli: str, gemini_api: str | None, caller_api: str, repeats: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (out_dir / "decisions.jsonl").open(
        "w", encoding="utf-8"
    ) as log:  # written as decided: a crash keeps them
        for name in ds.SCENARIOS:
            for kind in RENDERS:
                audio = render(name, kind, out_dir / "audio")
                for condition in CONDITIONS:
                    for repeat in range(repeats):
                        record = decide(name, kind, condition, repeat, audio, cli, gemini_api, caller_api)
                        records.append(record)
                        log.write(json.dumps(record, ensure_ascii=False) + "\n")
                        log.flush()
    report = decision_score.score(records, ds.answer_keys())
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--cli", required=True, help="installed agent-listening executable")
    parser.add_argument("--gemini-api", default=None, help="provider executable passed to the CLI")
    parser.add_argument("--caller-api", default="gemini-api", help="provider executable for the caller agent")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = run(EVAL_DIR / "runs" / args.run, args.cli, args.gemini_api, args.caller_api, args.repeats)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
