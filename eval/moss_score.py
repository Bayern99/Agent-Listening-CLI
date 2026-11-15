# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "soundfile>=0.12,<0.14", "scipy>=1.13,<2"]
# ///
"""Supplement eval/score.py with separate focused, raw-call and matched-pair records."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import corpus
import score
from build import EVAL_DIR


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def changes(rows: list[dict]) -> dict:
    by_kind = defaultdict(list)
    for row in rows:
        by_kind[row["change"]].append(row["result"]["detected"])
    return {
        "detection": score.wilson(sum(r["result"]["detected"] for r in rows), len(rows)),
        "by_change": {k: score.wilson(sum(v), len(v)) for k, v in sorted(by_kind.items())},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--baseline", default="holdout-1")
    args = parser.parse_args()
    root = EVAL_DIR / "runs" / args.run
    baseline = EVAL_DIR / "runs" / args.baseline
    cases = [c for c in corpus.cases() if c["split"] == "holdout"]
    baseline_summary = load(baseline / "summary-holdout.json")
    expected = baseline_summary["scored"] + len(baseline_summary["missing"]) + len(baseline_summary["failed"])
    if sum(len(c["conditions"]) for c in cases) != expected:
        raise ValueError("Corpus differs from baseline; set AGENT_LISTENING_EVAL_PAID_TIER=1 for MUSE")
    focused, pairs, missing, failed, identical, cb_cases = [], [], [], [], [], []
    for case in cases:
        for condition in case["conditions"]:
            path = root / f"{case['id']}__{condition}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            current = load(path)
            if current["status"] == "failed":
                failed.append(str(path))
                continue
            old = load(baseline / path.name)
            if [i["sha256"] for i in current["inputs"]] != [i["sha256"] for i in old["inputs"]]:
                raise ValueError(f"baseline input mismatch: {path.name}")
            if condition == "H" and case["truth"]["kind"] in ("change", "choice"):
                a = score.score_case(current, case["truth"])
                b = score.score_case(old, case["truth"])
                field = "detected" if case["truth"]["kind"] == "change" else "correct"
                pairs.append(
                    {
                        "case": case["id"],
                        "kind": case["truth"]["kind"],
                        "moss": a[field],
                        "hydrogen": b[field],
                    }
                )
            if condition == "CB":
                flags = score.score_change(current, case["truth"])["order_consistent"]
                cb_cases.append(
                    {"case": case["id"], "difference_claims": len(flags), "stable_claims": sum(flags)}
                )
        if case["truth"]["kind"] == "change":
            path = root / "focused" / f"{case['id']}__F.json"
            if path.exists() and load(path)["status"] != "failed":
                focused.append(
                    {
                        "case": case["id"],
                        "change": case["truth"]["change"],
                        "result": score.score_change(load(path), case["truth"]),
                    }
                )
            else:
                failed.append(str(path))
        if case["truth"]["kind"] == "identical":
            path = root / "identical-model" / f"{case['id']}__IM.json"
            if path.exists() and load(path)["status"] != "failed":
                identical.append({"case": case["id"], "result": score.score_null(load(path), {})})
            else:
                failed.append(str(path))
    raw = [load(p) | {"file": str(p)} for p in root.rglob("raw/*.json")]
    raw_failures = [r for r in raw if r["status"] == "failed"]
    matched = {}
    for kind in ("change", "choice"):
        rows = [r for r in pairs if r["kind"] == kind]
        matched[kind] = {
            "n": len(rows),
            "moss_only_correct": [r["case"] for r in rows if r["moss"] and not r["hydrogen"]],
            "hydrogen_only_correct": [r["case"] for r in rows if r["hydrogen"] and not r["moss"]],
            "both_correct": sum(r["moss"] and r["hydrogen"] for r in rows),
            "both_wrong": sum(not r["moss"] and not r["hydrogen"] for r in rows),
        }
    summary = {
        "run": args.run,
        "baseline": args.baseline,
        "base_summary": load(root / "summary-holdout.json"),
        "missing": missing,
        "failed": failed,
        "focused": changes(focused),
        "matched": matched,
        "matched_rows": pairs,
        "counterbalance_case_coverage": cb_cases,
        "identical_model": {
            "hallucinated": score.wilson(sum(r["result"]["hallucinated"] for r in identical), len(identical)),
            "rows": identical,
        },
        "raw_calls": len(raw),
        "raw_failures": [{"file": r["file"], "error": r.get("error")} for r in raw_failures],
        "parse_failures": sum("raw_response" in r for r in raw_failures),
        "call_wall_s": score.median_ci([r["wall_s"] for r in raw]),
        "total_call_wall_s": round(sum(r["wall_s"] for r in raw), 2),
        "peak_mps_driver_bytes_sampled": max(
            (r["memory"]["mps_driver_bytes_sampled"] for r in raw), default=0
        ),
        "peak_rss_bytes_process_highwater": max(
            (r["memory"]["rss_bytes_process_highwater"] for r in raw), default=0
        ),
    }
    out = root / "comparison.json"
    out.write_text(json.dumps(summary, indent=2))
    print(out)


if __name__ == "__main__":
    main()
