# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2,<3", "soundfile>=0.12,<0.14", "scipy>=1.13,<2", "jsonschema>=4,<5"]
# ///
"""Recover only response-free MPS OOM attempts in a separate view, with a fresh process per task."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import moss


def recoverable(record: dict) -> bool:
    return (
        record.get("status") == "failed"
        and "raw_response" not in record
        and "MPS backend out of memory" in record.get("error", "")
    )


def directory(root: Path, condition: str) -> Path:
    return root / {"F": "focused", "IM": "identical-model"}.get(condition, "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--case")
    parser.add_argument("--condition")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    args = parser.parse_args()
    source, target = (moss.EVAL_DIR / "runs" / name for name in (args.run, args.out))
    if source.resolve() == target.resolve():
        raise ValueError("recovery must preserve the original run in a separate directory")
    manifest = json.loads((moss.EVAL_DIR / "build-manifest.json").read_text())
    if args.case:
        if not args.condition:
            raise ValueError("worker needs a condition")
        case = next(c for c in moss.corpus.cases() if c["id"] == args.case)
        weights = (
            Path.home()
            / ".cache/huggingface/hub"
            / ("models--OpenMOSS-Team--MOSS-Music-8B-Instruct/snapshots")
            / moss.MODEL_REVISION
        )
        model = moss.Model(moss.EVAL_DIR / "MOSS-Music", weights, args.device)
        moss.execute(case, args.condition, manifest, model, directory(target, args.condition))
        return

    method = json.loads((source / "method.json").read_text())
    repo = Path(__file__).resolve().parents[1]
    for name, digest in method["files_sha256"].items():
        if moss.sha256(repo / name) != digest:
            raise ValueError(f"frozen method changed: {name}")
    bad = {p for p in source.rglob("raw/*.json") if recoverable(json.loads(p.read_text()))}
    jobs = sorted({tuple(p.name.split("__")[:2]) for p in bad})
    excluded = bad | {p.parents[1] / ("__".join(p.name.split("__")[:2]) + ".json") for p in bad}
    before = moss.fingerprint(manifest)
    originals = {str(p.relative_to(source)): moss.sha256(p) for p in source.rglob("*__*.json")}
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*__*.json"):
        if path in excluded:
            continue
        dest = target / path.relative_to(source)
        if dest.exists():
            if dest.read_bytes() != path.read_bytes():
                raise ValueError(f"preserved response differs: {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    record = {
        "source": str(source),
        "method": method,
        "jobs": jobs,
        "rule": "Only MPS OOM failures with no raw_response; generated replies are never retried",
        "original_response_sha256": originals,
        "workers": [],
    }
    checkpoint = target / "recovery.json"
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous["source"] != str(source) or previous["original_response_sha256"] != originals:
            raise ValueError("recovery source changed")
        record = previous
    try:
        for case, condition in jobs:
            out = directory(target, condition) / f"{case}__{condition}.json"
            if out.exists():
                continue
            log = target / "worker-logs" / f"{case}__{condition}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run",
                args.run,
                "--out",
                args.out,
                "--case",
                case,
                "--condition",
                condition,
                "--device",
                args.device,
            ]
            with log.open("w") as handle:
                result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, timeout=1800)
            record["workers"].append(
                {
                    "case": case,
                    "condition": condition,
                    "exit": result.returncode,
                    "wall_s": round(time.monotonic() - started, 3),
                }
            )
            moss.write(target / "recovery.json", record)
            if not out.exists():
                raise RuntimeError(f"worker failed before saving a response: {case} {condition}")
            print(f"{case} {condition}: {json.loads(out.read_text())['status']}", flush=True)
    finally:
        after = moss.fingerprint(manifest)
        changed = [p for p in before if before[p] != after.get(p)]
        changed_replies = [name for name, digest in originals.items() if moss.sha256(source / name) != digest]
        moss.write(
            target / "integrity-after.json",
            {
                "files_checked": len(before),
                "changed": changed,
                "original_responses_changed": changed_replies,
            },
        )
        moss.write(target / "recovery.json", record)
        if changed or changed_replies:
            raise ValueError("original audio or response files changed")


if __name__ == "__main__":
    main()
