# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "soundfile>=0.12,<0.14", "scipy>=1.13,<2"]
# ///
"""Run the corpus through the installed `agent-listening` CLI.

Usage: uv run eval/run.py --split pilot --run pilot-1 --cli /path/to/agent-listening [--jobs 6] [--match REGEX]
The Gemini key is read from the file named by $AGENT_LISTENING_EVAL_KEY_FILE and passed only to the child.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import corpus
from build import EVAL_DIR, recipe_key, sha256


def commands(case: dict, condition: str, manifest: dict, cli: str, out: Path) -> list[str]:
    paths = [manifest[recipe_key(r)]["path"] for r in case["inputs"]]
    task = case["task"]
    cmd = [cli, task, *paths]
    if task == "ask":
        cmd.append(case["question"])
    if task == "investigate":
        start, end = case["focus"]
        cmd += ["--focus", f"{start}-{end}", "--question", case["question"]]
    if condition == "CB":
        cmd.append("--counterbalance")
    if condition == "H+C":
        cmd += [arg for p in paths for arg in ("--context", str(Path(p).with_suffix(".scd")))]
    if condition == "MC":
        misleading = manifest[recipe_key(case["misleading_context"])]["path"]
        cmd += [
            "--context",
            str(Path(paths[0]).with_suffix(".scd")),
            "--context",
            str(Path(misleading).with_suffix(".scd")),
        ]
    return [*cmd, "--out", str(out)]


def fingerprint(manifest: dict) -> dict[str, str]:
    files = {v["path"] for v in manifest.values()}
    files |= {str(Path(p).with_suffix(".scd")) for p in list(files) if Path(p).with_suffix(".scd").exists()}
    files |= {
        str(EVAL_DIR / "sources" / name)
        for name in json.loads((Path(__file__).parent / "source_hashes.json").read_text())
    }
    return {f: sha256(Path(f)) for f in sorted(files)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["pilot", "holdout", "all"], required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cli", required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--match", default="")
    args = parser.parse_args()

    manifest = json.loads((EVAL_DIR / "build-manifest.json").read_text())
    run_dir = EVAL_DIR / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["GEMINI_API_KEY"] = Path(os.environ["AGENT_LISTENING_EVAL_KEY_FILE"]).read_text().strip()

    jobs = []
    for case in corpus.cases():
        if args.split != "all" and case["split"] != args.split:
            continue
        if args.match and not re.search(args.match, case["id"]):
            continue
        for condition in case["conditions"]:
            out = run_dir / f"{case['id']}__{condition}.json"
            if not out.exists():
                jobs.append((case, condition, out))

    before = fingerprint(manifest)

    def execute(job: tuple[dict, str, Path]) -> str:
        case, condition, out = job
        cmd = commands(case, condition, manifest, args.cli, out)
        started = time.monotonic()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        log = {
            "case": case["id"],
            "condition": condition,
            "argv": cmd[1:],
            "exit": proc.returncode,
            "wall_s": round(time.monotonic() - started, 2),
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-4000:],
        }
        out.with_suffix(".log.json").write_text(json.dumps(log, indent=2))
        return f"{case['id']}__{condition}: exit {proc.returncode} in {log['wall_s']} s"

    with ThreadPoolExecutor(args.jobs) as pool:
        for line in pool.map(execute, jobs):
            print(line, flush=True)

    after = fingerprint(manifest)
    changed = [f for f in before if before[f] != after.get(f)]
    integrity = {
        "files_checked": len(before),
        "changed": changed,
        "fingerprint_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
    }
    (run_dir / f"integrity-{int(time.time())}.json").write_text(json.dumps(integrity, indent=2))
    print(json.dumps({k: v for k, v in integrity.items() if k != "changed"} | {"changed": len(changed)}))
    if changed:
        raise SystemExit(f"ORIGINAL OR CASE FILES CHANGED: {changed}")


if __name__ == "__main__":
    main()
