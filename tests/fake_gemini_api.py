#!/usr/bin/env python3
"""Fake gemini-api executable for offline tests.

Records every invocation as a JSON line {"argv": [...], "stdin": "..."}
in the file named by env FAKE_GEMINI_LOG. Behaviour knobs:

- FAKE_GEMINI_REPLY: path to a JSON file used as the model_output text payload
- FAKE_GEMINI_EXIT: simulate a non-zero exit with an agent-mode stderr error
- FAKE_GEMINI_NO_CREDENTIAL: whoami reports both credentials unset
- FAKE_GEMINI_VERSION: version string printed by --version (default 0.4.5)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_REPLY = {"summary": "fake summary", "claims": []}


def _record(argv: list[str], stdin: str) -> None:
    log = os.environ.get("FAKE_GEMINI_LOG")
    if log:
        with Path(log).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"argv": argv, "stdin": stdin}) + "\n")


def _whoami() -> int:
    if os.environ.get("FAKE_GEMINI_NO_CREDENTIAL"):
        print("Credentials:")
        print("  --api-key                   [unset  ] (not set)")
        print("  --access-token              [unset  ] (not set)")
    else:
        source = os.environ.get("FAKE_GEMINI_CREDENTIAL") or "env"
        print("Credentials:")
        print(f"  --api-key                   [{source:<7}] AQ******zA")
        print("  --access-token              [unset  ] (not set)")
    return 0


def _generate(argv: list[str], stdin: str) -> int:
    exit_code = int(os.environ.get("FAKE_GEMINI_EXIT") or 0)
    if exit_code:
        err = {
            "error": "simulated failure",
            "error_reason": "FAKE",
            "error_type": "authentication_error" if exit_code == 3 else "runtime_error",
            "exit_code": exit_code,
            "hints": [],
            "message": "simulated failure",
        }
        print(json.dumps(err), file=sys.stderr)
        return exit_code
    reply_path = os.environ.get("FAKE_GEMINI_REPLY")
    payload = json.loads(Path(reply_path).read_text(encoding="utf-8")) if reply_path else DEFAULT_REPLY
    interaction = {
        "created": "2026-11-15T00:00:00Z",
        "id": "ix_fake",
        "model": "gemini-4-hydrogen",
        "status": "completed",
        "updated": "2026-11-15T00:00:00Z",
        "steps": [{"content": [{"text": json.dumps(payload), "type": "text"}], "type": "model_output"}],
        "usage": {
            "total_input_tokens": 100,
            "input_tokens_by_modality": [{"modality": "audio", "tokens": 64}],
            "total_output_tokens": 20,
            "total_thought_tokens": 7,
        },
    }
    print(json.dumps(interaction))
    return 0


def main() -> int:
    argv = sys.argv[1:]
    stdin_data = ""
    if "--body" in argv and argv[argv.index("--body") + 1] == "@-":
        stdin_data = sys.stdin.read()
    _record(argv, stdin_data)
    if argv == ["--version"] or argv == ["version"]:
        print(f"gemini-api {os.environ.get('FAKE_GEMINI_VERSION', '0.4.5')}")
        return 0
    if argv[:1] == ["whoami"]:
        return _whoami()
    if argv[:2] == ["files", "upload"]:
        print("files/fake-upload-id")
        return 0
    if argv[:2] == ["files", "delete"]:
        return 0
    if argv[:1] == ["generate"]:
        return _generate(argv, stdin_data)
    print(
        json.dumps({"error_type": "validation_error", "message": "unknown command", "exit_code": 2}),
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
