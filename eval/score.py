# /// script
# requires-python = ">=3.11"
# ///
"""Score a run directory against corpus ground truth. Usage: uv run eval/score.py --run pilot-1 [--split pilot]"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import corpus
from build import EVAL_DIR

HIT_S = 3.0
PRICE_PER_M = {"input": 1.50, "output": 7.50}
LETTER = re.compile(r"^\W*([A-D])\b")


def wilson(k: int, n: int, z: float = 1.96) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": None}
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "k": k,
        "n": n,
        "rate": round(p, 3),
        "ci95": [round(max(0, centre - half), 3), round(min(1, centre + half), 3)],
    }


def newcombe(a: dict, b: dict) -> dict | None:
    if not a["n"] or not b["n"]:
        return None
    diff = a["rate"] - b["rate"]
    (la, ua), (lb, ub) = a["ci95"], b["ci95"]
    lower = diff - math.sqrt((a["rate"] - la) ** 2 + (ub - b["rate"]) ** 2)
    upper = diff + math.sqrt((ua - a["rate"]) ** 2 + (b["rate"] - lb) ** 2)
    return {"difference": round(diff, 3), "ci95": [round(lower, 3), round(upper, 3)]}


def median_ci(values: list[float], reps: int = 2000) -> dict:
    if not values:
        return {"n": 0, "median": None, "ci95": None}
    rng = random.Random(0)
    boots = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(reps))
    return {
        "n": len(values),
        "median": round(statistics.median(values), 2),
        "ci95": [round(boots[int(0.025 * reps)], 2), round(boots[int(0.975 * reps)], 2)],
    }


def overlaps(locus: dict, region: list[float], pad: float) -> bool:
    return locus["start"] <= region[1] + pad and locus["end"] >= region[0] - pad


def difference_regions(evidence: dict) -> list[list[float]] | None:
    for m in evidence.get("measurements", []):
        if m["kind"] == "signal-difference" and m["result"].get("available"):
            return [[r["start"], r["end"]] for r in m["result"]["regions"]]
    return None


def score_change(evidence: dict, truth: dict) -> dict:
    region = truth["regions"][0]
    differs = [c for c in evidence["claims"] if c.get("relation") == "differs"]
    right = [c for c in differs if c["aspect"] in truth["aspects"]]
    errors = [abs(locus["start"] - region[0]) for c in right for locus in c["loci"]]
    false_pos = [c for c in differs if not any(overlaps(locus, region, HIT_S) for locus in c["loci"])]
    regions = difference_regions(evidence)
    det_error = abs(min(r[0] for r in regions) - region[0]) if regions else None
    return {
        "detected": bool(right),
        "detected_after_checks": any(c["status"] != "contradicted" for c in right),
        "onset_error_s": min(errors) if errors else None,
        "onset_hit3": bool(errors) and min(errors) <= HIT_S,
        "overlap_hit": any(overlaps(locus, region, HIT_S) for c in right for locus in c["loci"]),
        "differs_claims": len(differs),
        "false_positive_claims": len(false_pos),
        "false_positive_after_checks": sum(c["status"] != "contradicted" for c in false_pos),
        "deterministic_onset_error_s": det_error,
        "deterministic_onset_hit3": det_error is not None and det_error <= HIT_S,
        "order_consistent": [
            any(k["name"] == "order-consistency" and k["result"] == "pass" for k in c["checks"])
            for c in differs
            if any(k["name"] == "order-consistency" for k in c["checks"])
        ],
    }


def score_null(evidence: dict, truth: dict) -> dict:
    differs = [c for c in evidence["claims"] if c.get("relation") == "differs"]
    misleading = set(truth.get("misleading_aspects", []))
    return {
        "differs_claims": len(differs),
        "hallucinated": bool(differs),
        "uncontradicted": sum(c["status"] != "contradicted" for c in differs),
        "misleading_aspect_claims": sum(c["aspect"] in misleading for c in differs),
        "context_basis_claims": sum(c["basis"] != "audio" for c in evidence["claims"]),
    }


def calls(evidence: dict) -> int:
    producer = evidence.get("producer") or {}
    return len(producer.get("interaction_ids") or [])


def score_case(evidence: dict, truth: dict) -> dict:
    kind = truth["kind"]
    if kind == "change":
        return score_change(evidence, truth)
    if kind == "null":
        return score_null(evidence, truth)
    if kind == "identical":
        return {"model_calls": calls(evidence), "claims": len(evidence["claims"])}
    if kind == "choice":
        match = LETTER.match(evidence.get("answer") or "")
        letter = match.group(1) if match else None
        return {"letter": letter, "correct": letter == truth["answer"]}
    claims = evidence["claims"]
    return {
        "claims": len(claims),
        "ungrounded": sum(c["status"] == "ungrounded" for c in claims),
        "scales": sorted({c["scale"] for c in claims}),
        "relations": sum(c.get("relation") not in (None, "differs", "unchanged") for c in claims),
        "context_basis": sum(c["basis"] != "audio" for c in claims),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="all")
    args = parser.parse_args()
    run_dir = EVAL_DIR / "runs" / args.run

    rows, missing, failed = [], [], []
    for case in corpus.cases():
        if args.split != "all" and case["split"] != args.split:
            continue
        for condition in case["conditions"]:
            path = run_dir / f"{case['id']}__{condition}.json"
            if not path.exists():
                missing.append(path.name)
                continue
            evidence = json.loads(path.read_text())
            if evidence["status"] == "failed":
                failed.append({"case": path.name, "error": evidence.get("error")})
                continue
            producer = evidence.get("producer") or {}
            rows.append(
                {
                    "case": case["id"],
                    "condition": condition,
                    "task": case["task"],
                    "split": case["split"],
                    "truth": case["truth"],
                    "result": score_case(evidence, case["truth"]),
                    "latency_s": producer.get("latency_s"),
                    "usage": producer.get("usage") or {},
                    "audio_s": sum(i["format"]["duration_s"] for i in evidence["inputs"]),
                }
            )

    summary: dict = {
        "run": args.run,
        "split": args.split,
        "scored": len(rows),
        "missing": missing,
        "failed": failed,
    }
    by = defaultdict(list)
    for row in rows:
        by[(row["truth"]["kind"], row["condition"])].append(row)

    for (kind, condition), group in sorted(by.items()):
        key = f"{kind}/{condition}"
        results = [r["result"] for r in group]
        if kind == "change":
            per_change = defaultdict(list)
            for r in group:
                per_change[r["truth"]["change"]].append(r["result"]["detected"])
            consistent = [x for r in results for x in r["order_consistent"]]
            summary[key] = {
                "detection": wilson(sum(r["detected"] for r in results), len(results)),
                "detection_after_checks": wilson(
                    sum(r["detected_after_checks"] for r in results), len(results)
                ),
                "onset_hit3": wilson(sum(r["onset_hit3"] for r in results), len(results)),
                "onset_error_s": median_ci(
                    [r["onset_error_s"] for r in results if r["onset_error_s"] is not None]
                ),
                "overlap_hit": wilson(sum(r["overlap_hit"] for r in results), len(results)),
                "false_positive_claims_per_case": round(
                    statistics.mean(r["false_positive_claims"] for r in results), 2
                ),
                "false_positive_after_checks_per_case": round(
                    statistics.mean(r["false_positive_after_checks"] for r in results), 2
                ),
                "deterministic_onset_hit3": wilson(
                    sum(r["deterministic_onset_hit3"] for r in results), len(results)
                ),
                "by_change": {c: wilson(sum(v), len(v)) for c, v in sorted(per_change.items())},
                "order_consistency": wilson(sum(consistent), len(consistent)),
            }
        elif kind == "null":
            summary[key] = {
                "cases_with_hallucinated_difference": wilson(
                    sum(r["hallucinated"] for r in results), len(results)
                ),
                "differs_claims": sum(r["differs_claims"] for r in results),
                "uncontradicted_differs_claims": sum(r["uncontradicted"] for r in results),
                "misleading_aspect_claims": sum(r["misleading_aspect_claims"] for r in results),
                "context_basis_claims": sum(r["context_basis_claims"] for r in results),
            }
        elif kind == "identical":
            summary[key] = {
                "cases": len(results),
                "model_calls": sum(r["model_calls"] for r in results),
                "claims": sum(r["claims"] for r in results),
            }
        elif kind == "choice":
            families = defaultdict(list)
            for r in group:
                families[
                    r["case"].rsplit("-ask-", 1)[-1] if "-ask-" in r["case"] else r["case"].split("-")[0]
                ].append(r)
            summary[key] = {
                "accuracy": wilson(sum(r["correct"] for r in results), len(results)),
                "mean_chance": round(statistics.mean(r["truth"]["chance"] for r in group), 3),
                "by_family": {
                    f: wilson(sum(x["result"]["correct"] for x in v), len(v))
                    for f, v in sorted(families.items())
                },
                "wrong": [
                    {"case": r["case"], "answer": r["result"]["letter"], "expected": r["truth"]["answer"]}
                    for r in group
                    if not r["result"]["correct"]
                ],
            }
        else:
            summary[key] = {"cases": [{"case": r["case"], **r["result"]} for r in group]}

    if "change/H" in summary and "change/H+C" in summary:
        sc_h = [r for r in by[("change", "H")] if r["case"].startswith("sc-")]
        h = wilson(sum(r["result"]["detected"] for r in sc_h), len(sc_h))
        summary["context_effect_sc_detection"] = {
            "H": h,
            "H+C": summary["change/H+C"]["detection"],
            "difference": newcombe(summary["change/H+C"]["detection"], h),
        }

    timed = [r for r in rows if r["latency_s"]]
    audio_min = sum(r["audio_s"] for r in timed) / 60 or 1
    tokens_in = sum(r["usage"].get("input_tokens", 0) for r in timed)
    tokens_out = sum(r["usage"].get("output_tokens", 0) + r["usage"].get("thought_tokens", 0) for r in timed)
    cost = tokens_in / 1e6 * PRICE_PER_M["input"] + tokens_out / 1e6 * PRICE_PER_M["output"]
    summary["cost"] = {
        "calls": len(timed),
        "latency_s": median_ci([r["latency_s"] for r in timed]),
        "input_tokens": tokens_in,
        "audio_tokens": sum(r["usage"].get("audio_tokens", 0) for r in timed),
        "output_plus_thought_tokens": tokens_out,
        "estimated_usd": round(cost, 3),
        "estimated_usd_per_audio_minute": round(cost / audio_min, 4),
    }
    out = run_dir / f"summary-{args.split}.json"
    out.write_text(json.dumps(summary, indent=2))
    (run_dir / f"rows-{args.split}.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
