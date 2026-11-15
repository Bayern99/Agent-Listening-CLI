"""The only gemini-api subprocess adapter. Stdlib only; no internal imports."""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SERIES = "0.4"

_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")
_FILE_URI_RE = re.compile(r"files/\S+")
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class ProviderError(Exception):
    def __init__(self, code: str, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code


@dataclass(frozen=True)
class Reply:
    payload: dict[str, object]
    text: str
    interaction_id: str | None
    usage: dict[str, int]
    latency_s: float
    cli_version: str


def _run(
    argv: Sequence[str], *, input_text: str | None, timeout_s: float
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(argv),
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ProviderError("PROVIDER_MISSING", f"provider executable not found: {argv[0]}", 1) from exc
    except subprocess.TimeoutExpired as exc:
        raise ProviderError("PROVIDER_TIMEOUT", f"provider call timed out after {timeout_s}s", 1) from exc


def _agent_mode_error(stderr: str, fallback_code: str, exit_code: int) -> ProviderError:
    detail = stderr.strip()
    if detail:
        try:
            body = json.loads(detail)
            return ProviderError(
                str(body.get("error_type") or fallback_code),
                str(body.get("message") or body.get("error") or detail),
                int(body.get("exit_code") or exit_code),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return ProviderError(fallback_code, detail or f"provider exited {exit_code}", exit_code)


def version(executable: str) -> str:
    proc = _run([executable, "--version"], input_text=None, timeout_s=30)
    if proc.returncode != 0:
        raise _agent_mode_error(proc.stderr, "PROVIDER_ERROR", proc.returncode)
    match = _VERSION_RE.search(proc.stdout)
    if match is None:
        raise ProviderError("PROVIDER_BAD_OUTPUT", f"cannot parse version output: {proc.stdout.strip()!r}", 1)
    return match.group(1)


_CRED_SOURCE_RE = re.compile(r"\[\s*([A-Za-z_-]+)\s*\]")


def credential_status(executable: str) -> str:
    """'configured' or 'missing' per `whoami`; never touches key material."""
    try:
        proc = _run([executable, "whoami"], input_text=None, timeout_s=30)
    except ProviderError:
        return "missing"
    if proc.returncode != 0:
        return "missing"
    in_credentials = False
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Credentials:"):
            in_credentials = True
            continue
        if not in_credentials:
            continue
        match = _CRED_SOURCE_RE.search(stripped)
        if match and match.group(1).lower() != "unset":
            return "configured"
    return "missing"


def _upload(executable: str, path: Path, timeout_s: float) -> str:
    proc = _run([executable, "files", "upload", str(path), "--wait"], input_text=None, timeout_s=timeout_s)
    if proc.returncode != 0:
        raise _agent_mode_error(proc.stderr, "PROVIDER_UPLOAD_FAILED", proc.returncode)
    match = _FILE_URI_RE.search(proc.stdout)
    if match is None:
        raise ProviderError(
            "PROVIDER_BAD_OUTPUT", f"files upload returned no files/<id>: {proc.stdout.strip()!r}", 1
        )
    return match.group(0)


def _delete(executable: str, file_uri: str) -> None:
    with contextlib.suppress(ProviderError):
        _run([executable, "files", "delete", file_uri], input_text=None, timeout_s=60)


def _model_text(interaction: Mapping[str, object]) -> str:
    parts = []
    steps = interaction.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict) or step.get("type") != "model_output":
                continue
            content = step.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ):
                    parts.append(item["text"])
    return "".join(parts)


def _map_usage(raw: object) -> dict[str, int]:
    usage = raw if isinstance(raw, dict) else {}
    audio_tokens = 0
    by_modality = usage.get("input_tokens_by_modality")
    if isinstance(by_modality, list):
        for entry in by_modality:
            if isinstance(entry, dict) and entry.get("modality") == "audio":
                audio_tokens += int(entry.get("tokens", entry.get("token_count", 0)) or 0)
    return {
        "input_tokens": int(usage.get("total_input_tokens") or 0),
        "audio_tokens": audio_tokens,
        "output_tokens": int(usage.get("total_output_tokens") or 0),
        "thought_tokens": int(usage.get("total_thought_tokens") or 0),
    }


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match is not None:
        return match.group(1)
    return stripped


def listen(
    *,
    executable: str,
    model: str,
    thinking: str,
    system: str,
    prompt: str,
    audio: Sequence[Path],
    schema: Mapping[str, object],
    timeout_s: float = 900,
    inline_limit_bytes: int = 14_000_000,
) -> Reply:
    """One non-streaming structured `gemini-api generate` call."""
    cli_version = version(executable)
    if cli_version.split(".")[:2] != SUPPORTED_SERIES.split("."):
        raise ProviderError(
            "PROVIDER_UNSUPPORTED",
            f"gemini-api {cli_version} unsupported (need {SUPPORTED_SERIES}.x)",
            3,
        )
    uploads: list[str] = []
    inputs: list[str] = []
    try:
        paths = [Path(p) for p in audio]
        upload_all = sum(p.stat().st_size for p in paths) > inline_limit_bytes
        for p in paths:
            if upload_all:
                uploads.append(_upload(executable, p, timeout_s))
                inputs.append(uploads[-1])
            else:
                inputs.append(str(p))
        body = json.dumps(
            {
                "generation_config": {"thinking_level": thinking},
                "response_format": {"type": "text", "mime_type": "application/json", "schema": dict(schema)},
                "response_mime_type": "application/json",
            }
        )
        argv = [executable, "generate", prompt, "-m", model, "--system", system]
        for item in inputs:
            argv += ["-i", item]
        argv += ["--body", "@-", "--stream=false", "-o", "json", "--agent-mode"]
        started = time.monotonic()
        proc = _run(argv, input_text=body, timeout_s=timeout_s)
        latency_s = time.monotonic() - started
        if proc.returncode != 0:
            code = "PROVIDER_AUTH" if proc.returncode == 3 else "PROVIDER_ERROR"
            raise _agent_mode_error(proc.stderr, code, proc.returncode)
        try:
            interaction = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "PROVIDER_BAD_OUTPUT", f"provider stdout is not JSON: {proc.stdout[:200]!r}", 1
            ) from exc
        if not isinstance(interaction, dict):
            raise ProviderError("PROVIDER_BAD_OUTPUT", "provider stdout is not an object", 1)
        text = _model_text(interaction)
        try:
            payload = json.loads(_strip_fence(text))
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "PROVIDER_BAD_PAYLOAD", f"model reply text is not JSON: {text[:200]!r}", 1
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderError("PROVIDER_BAD_PAYLOAD", "model reply JSON is not an object", 1)
        raw_id = interaction.get("id")
        return Reply(
            payload=payload,
            text=text,
            interaction_id=str(raw_id) if raw_id is not None else None,
            usage=_map_usage(interaction.get("usage")),
            latency_s=latency_s,
            cli_version=cli_version,
        )
    finally:
        for file_uri in uploads:
            _delete(executable, file_uri)
