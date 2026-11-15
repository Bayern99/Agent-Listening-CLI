"""Argument parsing, output writing and exit codes.

Stdlib-only at module level: tasks, audio and evidence are imported lazily
inside handlers so `agent-listening --help` and `--version` load neither
numpy nor soundfile.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_listening import __version__

AUDIO_EXTS = frozenset({".wav", ".flac", ".aif", ".aiff", ".mp3", ".ogg", ".oga", ".m4a", ".opus"})
DEFAULT_CACHE = "~/.cache/agent-listening"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-listening",
        description="Listening-model evidence for composed audio.",
    )
    parser.add_argument("--version", action="version", version=f"agent-listening {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def listening(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--out", help="write evidence JSON here instead of stdout (no-clobber)")
        p.add_argument("--overwrite", action="store_true", help="replace an existing --out file")
        p.add_argument("--model", default=None, help="listening model id")
        p.add_argument(
            "--thinking",
            default=None,
            choices=("minimal", "low", "medium", "high"),
            help="model thinking level",
        )
        p.add_argument("--dry-run", action="store_true", help="identity, transport and measurements only")
        p.add_argument("--cache-dir", default=None, help="transport cache dir (env AGENT_LISTENING_CACHE)")
        p.add_argument("--gemini-api", dest="gemini_api", default=None, help="provider executable")
        p.add_argument("--context", action="append", default=[], help="UTF-8 context file (repeatable)")
        return p

    hear = listening("hear", "perceptual description across scales")
    hear.add_argument("audio")
    hear.add_argument("--focus", default=None, help="time span S-E, M:SS-M:SS or H:MM:SS-H:MM:SS")

    ask = listening("ask", "targeted question with supporting claims")
    ask.add_argument("audio")
    ask.add_argument("question")
    ask.add_argument("--focus", default=None, help="time span")

    compare = listening("compare", "differences and preserved relations between versions")
    compare.add_argument("audio_a")
    compare.add_argument("audio_b")
    compare.add_argument("--question", default=None)
    compare.add_argument("--counterbalance", action="store_true", help="second call with clip order swapped")
    compare.add_argument(
        "--follow-up",
        type=int,
        default=2,
        metavar="N",
        help="re-listen calls for uncovered measured regions (0 disables, default 2)",
    )

    investigate = listening("investigate", "zoomed listening plus deterministic measurements")
    investigate.add_argument("audio")
    investigate.add_argument("--question", default=None)
    investigate.add_argument("--focus", required=True, help="time span S-E (required)")
    investigate.add_argument(
        "--measure",
        default=None,
        help="comma list of level,spectrum,stereo,onsets (default: all)",
    )

    review = listening("review", "goal and protected-relation assessments against a brief")
    review.add_argument("audio")
    review.add_argument("--brief", required=True, help="brief JSON path")
    review.add_argument("--before", default=None, help="previous render for revision context")

    identify = sub.add_parser("identify", help="identity and format only")
    identify.add_argument("audio", nargs="+")

    check = sub.add_parser("check", help="re-hash inputs of an evidence document")
    check.add_argument("evidence")
    check.add_argument("--audio", action="append", default=[], help="override input path (in order)")

    sub.add_parser("doctor", help="provider CLI version and credential presence")
    return parser


def _exe(args: argparse.Namespace) -> str:
    return getattr(args, "gemini_api", None) or os.environ.get("AGENT_LISTENING_GEMINI_API") or "gemini-api"


def _options(args: argparse.Namespace) -> Any:
    from agent_listening import tasks

    cache = args.cache_dir or os.environ.get("AGENT_LISTENING_CACHE") or DEFAULT_CACHE
    return tasks.Options(
        model=args.model or tasks.Options.model,
        thinking=args.thinking or tasks.Options.thinking,
        cache_dir=cache,
        dry_run=args.dry_run,
        executable=_exe(args),
    )


def _focus(args: argparse.Namespace) -> tuple[float, float] | None:
    if getattr(args, "focus", None) is None:
        return None
    from agent_listening import evidence

    try:
        return evidence.parse_span(args.focus)
    except ValueError as exc:
        raise SystemExit(_usage_error(str(exc))) from exc


def _usage_error(message: str) -> int:
    print(json.dumps({"error": message}), file=sys.stderr)
    return 2


def _input_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    for attr in ("audio", "audio_a", "audio_b", "before"):
        value = getattr(args, attr, None)
        if isinstance(value, str):
            paths.append(value)
        elif value:
            paths.extend(value)
    return paths


def _check_out(p: Path, inputs: Sequence[str], overwrite: bool) -> str | None:
    """Return a usage error message, or None when the --out target is acceptable."""
    if p.suffix.lower() in AUDIO_EXTS:
        return f"refusing to write evidence over an audio file: {p}"
    resolved = p.resolve()
    for raw in inputs:
        if resolved == Path(raw).resolve():
            return f"--out equals an input path: {p}"
    if p.exists() and not overwrite:
        return f"--out exists (use --overwrite): {p}"
    return None


def _validate_out(args: argparse.Namespace) -> None:
    """Fail fast on an unusable --out target, before any provider call."""
    out = getattr(args, "out", None)
    if not out:
        return
    problem = _check_out(Path(out), _input_paths(args), args.overwrite)
    if problem is not None:
        raise SystemExit(_usage_error(problem))


def _write_evidence(out_path: str, text: str, inputs: Sequence[str], overwrite: bool) -> None:
    """The only evidence writer: no-clobber, refuses audio targets and inputs."""
    p = Path(out_path)
    problem = _check_out(p, inputs, overwrite)
    if problem is not None:
        raise SystemExit(_usage_error(problem))
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        Path(tmp_name).replace(p)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _emit(args: argparse.Namespace, result: Any, code: int) -> int:
    from agent_listening import evidence

    if hasattr(result, "schema") and result.schema == evidence.SCHEMA:
        text = evidence.dumps(result)
        status = result.status
        input_paths = [i.path for i in result.inputs]
    else:
        text = json.dumps(result, indent=2, ensure_ascii=False)
        status = result.get("status") if isinstance(result, dict) else ""
        input_paths = []
    if getattr(args, "out", None):
        _write_evidence(args.out, text + "\n", input_paths, args.overwrite)
        print(json.dumps({"evidence": args.out, "status": status}))
    else:
        print(text)
    return code


def _run_command(args: argparse.Namespace) -> tuple[Any, int]:
    from agent_listening import tasks

    if args.command == "identify":
        return tasks.identify(args.audio)
    if args.command == "check":
        return tasks.check(args.evidence, args.audio)
    options = _options(args)
    if args.command == "hear":
        return tasks.hear(args.audio, focus=_focus(args), contexts=args.context, options=options)
    if args.command == "ask":
        return tasks.ask(
            args.audio, args.question, focus=_focus(args), contexts=args.context, options=options
        )
    if args.command == "compare":
        return tasks.compare(
            args.audio_a,
            args.audio_b,
            question=args.question,
            contexts=args.context,
            counterbalance=args.counterbalance,
            follow_up=args.follow_up,
            options=options,
        )
    if args.command == "investigate":
        focus = _focus(args)
        if focus is None:
            raise tasks.UsageError("--focus is required")
        measures = tuple(m.strip() for m in (args.measure or "").split(",") if m.strip())
        return tasks.investigate(
            args.audio,
            focus=focus,
            measures=measures,
            question=args.question,
            contexts=args.context,
            options=options,
        )
    if args.command == "review":
        return tasks.review(
            args.audio, brief_path=args.brief, before=args.before, contexts=args.context, options=options
        )
    raise SystemExit(_usage_error(f"unknown command: {args.command}"))


def _doctor() -> int:
    from agent_listening import tasks

    exe = os.environ.get("AGENT_LISTENING_GEMINI_API") or "gemini-api"
    out = tasks.doctor(exe)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    ready = out["gemini_api"]["supported"] and out["credential"] == "configured"
    return 0 if ready else 3


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    from agent_listening import audio, evidence, tasks

    _validate_out(args)
    try:
        result, code = _run_command(args)
    except tasks.UsageError as exc:
        return _usage_error(str(exc))
    except (audio.IdentityError, audio.AudioError, evidence.ContractError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    except OSError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    return _emit(args, result, code)
