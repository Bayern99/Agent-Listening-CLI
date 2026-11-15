# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "soundfile>=0.12,<0.14", "scipy>=1.13,<2"]
# ///
"""Re-apply the current deterministic checks to a finished run's stored model claims (no model calls).

Usage: uv run --with-editable . eval/recheck.py --run holdout-1 --out holdout-1-recheck
Comparison and review-with-before documents are re-checked; every other document is copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build import EVAL_DIR

from agent_listening import tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, target = EVAL_DIR / "runs" / args.run, EVAL_DIR / "runs" / args.out
    target.mkdir(parents=True, exist_ok=True)
    rechecked = 0
    for path in sorted(source.glob("*__*.json")):
        if path.name.endswith(".log.json"):
            continue
        document = json.loads(path.read_text())
        before = document.get("inputs", [])
        if (
            document.get("status") != "failed"
            and document["task"] in ("compare", "review")
            and len(before) == 2
        ):
            document = tasks.recheck_claims(document)
            rechecked += 1
        (target / path.name).write_text(json.dumps(document, indent=2, ensure_ascii=False))
    for path in source.glob("integrity-*.json"):
        shutil.copy(path, target / path.name)
    print(f"rechecked {rechecked} documents into {target}")


if __name__ == "__main__":
    main()
