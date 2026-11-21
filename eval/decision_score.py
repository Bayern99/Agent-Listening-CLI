"""Score decision records against recipe-registered answer keys. Pure: no I/O, no model.

A record is one caller decision: scenario, render ("defect" or "control"), condition, repeat,
decision (an option of the scenario, or anything else when the caller failed to decide), cost.
A decision outside the scenario's options is invalid: it stays in every denominator, is never
correct, and is never a false change.
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import PRICE_PER_M, newcombe, wilson

BASELINE = "source"


def _valid(record: dict, keys: dict) -> bool:
    return record.get("decision") in keys[record["scenario"]]["options"]


def _correct(record: dict, keys: dict) -> bool:
    return _valid(record, keys) and record["decision"] == keys[record["scenario"]][record["render"]]


def _cost(records: list[dict]) -> dict:
    n = len(records)
    total = {
        k: sum(r["cost"].get(k, 0) for r in records)
        for k in ("caller_calls", "listening_calls", "input_tokens", "output_tokens")
    }
    usd = (
        total["input_tokens"] * PRICE_PER_M["input"] + total["output_tokens"] * PRICE_PER_M["output"]
    ) / 1e6
    return {
        "caller_calls": round(total["caller_calls"] / n, 2),
        "listening_calls": round(total["listening_calls"] / n, 2),
        "input_tokens": round(total["input_tokens"] / n),
        "output_tokens": round(total["output_tokens"] / n),
        "usd": round(usd / n, 4),
        "seconds_median": round(statistics.median(r["cost"].get("seconds", 0.0) for r in records), 2),
    }


def _condition(records: list[dict], keys: dict) -> dict:
    defect = [r for r in records if r["render"] == "defect"]
    control = [r for r in records if r["render"] == "control"]
    commands: dict[str, int] = defaultdict(int)
    for r in records:
        for command in r.get("commands", []):
            commands[command["command"]] += 1
    return {
        "defect_correct": wilson(sum(_correct(r, keys) for r in defect), len(defect)),
        "control_false_change": wilson(
            sum(_valid(r, keys) and not _correct(r, keys) for r in control), len(control)
        ),
        "invalid": sum(not _valid(r, keys) for r in records),
        "cost_per_decision": _cost(records),
        "listening_commands": dict(sorted(commands.items())),
    }


def _change(baseline: list[dict], other: list[dict], keys: dict) -> dict:
    """Pair defect-render decisions by scenario and repeat; a change is not an improvement."""
    cells = {"wrong_to_right": 0, "right_to_wrong": 0, "wrong_to_wrong": 0, "unchanged": 0, "incomplete": 0}
    index = {(r["scenario"], r["repeat"]): r for r in other if r["render"] == "defect"}
    seen = set()
    for before in (r for r in baseline if r["render"] == "defect"):
        pair = (before["scenario"], before["repeat"])
        seen.add(pair)
        after = index.get(pair)
        if after is None or not _valid(before, keys) or not _valid(after, keys):
            cells["incomplete"] += 1
        elif before["decision"] == after["decision"]:
            cells["unchanged"] += 1
        elif _correct(after, keys):
            cells["wrong_to_right"] += 1
        elif _correct(before, keys):
            cells["right_to_wrong"] += 1
        else:
            cells["wrong_to_wrong"] += 1
    cells["incomplete"] += len(set(index) - seen)
    return cells


def _family(records: list[dict], keys: dict) -> dict:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_condition[r["condition"]].append(r)
    conditions = {name: _condition(group, keys) for name, group in sorted(by_condition.items())}
    change = {}
    for name, group in sorted(by_condition.items()):
        if name == BASELINE or BASELINE not in by_condition:
            continue
        change[name] = {
            "cells": _change(by_condition[BASELINE], group, keys),
            "difference": newcombe(
                conditions[name]["defect_correct"], conditions[BASELINE]["defect_correct"]
            ),
        }
    return {
        "scenarios": sorted({r["scenario"] for r in records}),
        "conditions": conditions,
        "change_vs_source": change,
    }


def score(records: list[dict], keys: dict) -> dict:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_family[keys[r["scenario"]]["family"]].append(r)
    return {
        "records": len(records),
        "families": {name: _family(group, keys) for name, group in sorted(by_family.items())},
    }
