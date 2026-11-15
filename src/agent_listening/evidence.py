"""Evidence contract types, validation, grounding checks and serialization.

Stdlib only. Times are seconds in the original input's time base.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, dataclass, fields, is_dataclass, replace
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

SCHEMA = "agent-listening/evidence"
VERSION = 1
TASKS = ("hear", "ask", "compare", "investigate", "review", "identify")
ROLES = ("subject", "before", "after")
ASPECTS = (
    "timbre",
    "articulation",
    "dynamics",
    "pitch",
    "tuning",
    "rhythm",
    "texture",
    "voice",
    "motif",
    "form",
    "space",
    "gesture",
    "production",
    "other",
)
SCALES = ("event", "phrase", "section", "work")
RELATIONS = (
    "repeats",
    "varies",
    "imitates",
    "answers",
    "doubles",
    "accompanies",
    "contrasts",
    "transforms",
    "returns",
    "differs",
    "unchanged",
)
BASES = ("audio", "context", "audio+context")
CONFIDENCES = ("low", "medium", "high")
ORIGINS = ("model", "measurement", "human")
STATUSES = ("unchecked", "consistent", "ungrounded", "contradicted")
VERDICTS = ("met", "partly", "not-met", "unclear")
CHECK_RESULTS = ("pass", "fail", "n/a")
EVIDENCE_STATUSES = ("complete", "partial", "failed")
THINKING_LEVELS = ("minimal", "low", "medium", "high")

DIFFERENCE_TOLERANCE_S = 2.0
DIFFERS_SUPPORT_FRACTION = 0.25
DIFFERS_SUPPORT_MAX_S = 0.5
LOCUS_TOLERANCE_S = 0.5

LOCUS_CHECK = "locus-in-range"
IDENTITY_CHECK = "pcm-identity"
DIFFERENCE_CHECK = "signal-difference"
ORDER_CHECK = "order-consistency"

DETERMINISTIC_CHECKS = (IDENTITY_CHECK, DIFFERENCE_CHECK)


class ContractError(Exception):
    """Evidence contract violation."""


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ContractError(message)


def _enum(value: str, allowed: tuple[str, ...], field_name: str) -> None:
    _require(value in allowed, f"{field_name} {value!r} not in {allowed}")


def _sha256(value: str, field_name: str) -> None:
    _require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        f"{field_name} must be a lowercase sha256 hex digest",
    )


@dataclass(frozen=True)
class AudioFormat:
    container: str
    subtype: str
    sample_rate: int
    channels: int
    frames: int
    duration_s: float

    def __post_init__(self) -> None:
        _require(self.sample_rate > 0, "sample_rate must be positive")
        _require(self.channels > 0, "channels must be positive")
        _require(self.frames >= 0, "frames must be non-negative")


@dataclass(frozen=True)
class Input:
    label: str
    role: str
    path: str
    sha256: str
    pcm_sha256: str
    format: AudioFormat

    def __post_init__(self) -> None:
        _require(bool(self.label), "input label must be non-empty")
        _enum(self.role, ROLES, "role")
        _sha256(self.sha256, "sha256")
        _sha256(self.pcm_sha256, "pcm_sha256")


@dataclass(frozen=True)
class ContextFile:
    label: str
    kind: str
    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        _require(bool(self.label), "context label must be non-empty")
        _sha256(self.sha256, "sha256")
        _require(self.bytes >= 0, "bytes must be non-negative")


@dataclass(frozen=True)
class BriefLocus:
    start: float
    end: float
    voice: str | None = None

    def __post_init__(self) -> None:
        _require(self.end >= self.start >= 0, "brief locus must satisfy 0 <= start <= end")


@dataclass(frozen=True)
class Goal:
    id: str
    text: str

    def __post_init__(self) -> None:
        _require(bool(self.id), "goal id must be non-empty")


@dataclass(frozen=True)
class Protection:
    id: str
    text: str
    loci: tuple[BriefLocus, ...] = ()

    def __post_init__(self) -> None:
        _require(bool(self.id), "protection id must be non-empty")
        _require(isinstance(self.loci, tuple), "protect loci must be a tuple")


@dataclass(frozen=True)
class Brief:
    path: str
    sha256: str
    goals: tuple[Goal, ...]
    protect: tuple[Protection, ...]

    def __post_init__(self) -> None:
        _sha256(self.sha256, "sha256")
        _require(
            isinstance(self.goals, tuple) and isinstance(self.protect, tuple), "brief fields must be tuples"
        )


@dataclass(frozen=True)
class Focus:
    start: float
    end: float

    def __post_init__(self) -> None:
        _require(self.end > self.start >= 0, "focus must satisfy 0 <= start < end")


@dataclass(frozen=True)
class Request:
    question: str | None
    focus: Focus | None
    measures: tuple[str, ...]
    counterbalance: bool
    follow_up: int = 0

    def __post_init__(self) -> None:
        _require(isinstance(self.measures, tuple), "measures must be a tuple")
        _require(self.follow_up >= 0, "follow_up must be non-negative")


@dataclass(frozen=True)
class Locus:
    input: str
    start: float
    end: float
    voice: str | None = None

    def __post_init__(self) -> None:
        _require(bool(self.input), "locus input label must be non-empty")
        _require(self.end >= self.start >= 0, "locus must satisfy 0 <= start <= end")


@dataclass(frozen=True)
class Check:
    name: str
    result: str
    detail: str
    measurement: str | None = None

    def __post_init__(self) -> None:
        _require(bool(self.name), "check name must be non-empty")
        _enum(self.result, CHECK_RESULTS, "check result")


@dataclass(frozen=True)
class Claim:
    id: str
    origin: str
    text: str
    aspect: str
    scale: str
    loci: tuple[Locus, ...]
    relation: str | None
    basis: str
    confidence: str
    checks: tuple[Check, ...]
    status: str

    def __post_init__(self) -> None:
        _require(bool(self.id), "claim id must be non-empty")
        _enum(self.origin, ORIGINS, "origin")
        _enum(self.aspect, ASPECTS, "aspect")
        _enum(self.scale, SCALES, "scale")
        if self.relation is not None:
            _enum(self.relation, RELATIONS, "relation")
        _enum(self.basis, BASES, "basis")
        _enum(self.confidence, CONFIDENCES, "confidence")
        _enum(self.status, STATUSES, "status")
        _require(
            isinstance(self.loci, tuple) and isinstance(self.checks, tuple), "claim fields must be tuples"
        )
        expected = derive_status(self.loci, self.checks)
        _require(self.status == expected, f"claim status {self.status!r} != derived {expected!r}")


def derive_status(loci: Sequence[Locus], checks: Sequence[Check]) -> str:
    """Derive claim status from loci and deterministic checks; never set freely.

    locus-in-range is a precondition, never corroboration; only deterministic
    measurements (DETERMINISTIC_CHECKS) can contradict or corroborate a claim.
    Any other check name (e.g. order-consistency self-agreement) is informational.
    """
    if any(c.name in DETERMINISTIC_CHECKS and c.result == "fail" for c in checks):
        return "contradicted"
    if not loci or any(c.name == LOCUS_CHECK and c.result == "fail" for c in checks):
        return "ungrounded"
    if any(c.name in DETERMINISTIC_CHECKS and c.result == "pass" for c in checks):
        return "consistent"
    return "unchecked"


def make_claim(
    *,
    id: str,
    origin: str,
    text: str,
    aspect: str,
    scale: str,
    loci: Sequence[Locus],
    relation: str | None,
    basis: str,
    confidence: str,
    checks: Sequence[Check] = (),
) -> Claim:
    loc = tuple(loci)
    chk = tuple(checks)
    return Claim(
        id=id,
        origin=origin,
        text=text,
        aspect=aspect,
        scale=scale,
        loci=loc,
        relation=relation,
        basis=basis,
        confidence=confidence,
        checks=chk,
        status=derive_status(loc, chk),
    )


@dataclass(frozen=True)
class Transport:
    input: str
    path: str
    sha256: str
    operation: str
    offset_s: float
    duration_s: float
    sample_rate: int
    channels: int

    def __post_init__(self) -> None:
        _require(bool(self.input), "transport input label must be non-empty")
        _sha256(self.sha256, "sha256")
        _require(self.operation in ("full", "excerpt"), "operation must be 'full' or 'excerpt'")
        _require(self.offset_s >= 0 and self.duration_s > 0, "bad transport span")
        _require(self.sample_rate > 0 and self.channels > 0, "bad transport format")


@dataclass(frozen=True)
class Measurement:
    id: str
    kind: str
    inputs: tuple[str, ...]
    method: str
    params: Mapping[str, Any]
    result: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require(bool(self.id) and bool(self.kind), "measurement id/kind must be non-empty")
        _require(isinstance(self.inputs, tuple), "measurement inputs must be a tuple")


@dataclass(frozen=True)
class Assessment:
    target: str
    verdict: str
    claims: tuple[str, ...]
    note: str

    def __post_init__(self) -> None:
        _require(bool(self.target), "assessment target must be non-empty")
        _enum(self.verdict, VERDICTS, "verdict")
        _require(isinstance(self.claims, tuple), "assessment claims must be a tuple")


@dataclass(frozen=True)
class Producer:
    provider: str
    cli_version: str
    model: str
    thinking: str
    instructions_sha256: str
    interaction_ids: tuple[str, ...]
    usage: Mapping[str, int]
    latency_s: float
    attempts: int

    def __post_init__(self) -> None:
        _enum(self.thinking, THINKING_LEVELS, "thinking")
        _sha256(self.instructions_sha256, "instructions_sha256")
        _require(isinstance(self.interaction_ids, tuple), "interaction_ids must be a tuple")
        _require(self.attempts >= 0, "attempts must be non-negative")


@dataclass(frozen=True)
class Tool:
    name: str
    version: str


@dataclass(frozen=True)
class ErrorBody:
    code: str
    message: str


@dataclass(frozen=True)
class Evidence:
    schema: str
    version: int
    task: str
    status: str
    tool: Tool
    created_at: str
    inputs: tuple[Input, ...]
    context: tuple[ContextFile, ...]
    brief: Brief | None
    request: Request
    transport: tuple[Transport, ...]
    measurements: tuple[Measurement, ...]
    claims: tuple[Claim, ...]
    assessments: tuple[Assessment, ...]
    answer: str | None
    summary: str
    producer: Producer | None
    limitations: tuple[str, ...]
    error: ErrorBody | None

    def __post_init__(self) -> None:
        _require(self.schema == SCHEMA, f"schema must be {SCHEMA!r}")
        _require(self.version == VERSION, f"version must be {VERSION}")
        _enum(self.task, TASKS, "task")
        _enum(self.status, EVIDENCE_STATUSES, "status")
        for field_name in (
            "inputs",
            "context",
            "transport",
            "measurements",
            "claims",
            "assessments",
            "limitations",
        ):
            _require(isinstance(getattr(self, field_name), tuple), f"{field_name} must be a tuple")


_SPAN_RE = re.compile(
    r"^\s*(\d+(?::\d{1,2})?(?::\d{1,2})?(?:\.\d+)?)\s*-\s*(\d+(?::\d{1,2})?(?::\d{1,2})?(?:\.\d+)?)\s*$"
)


def _clock_seconds(part: str) -> float:
    pieces = part.split(":")
    if len(pieces) == 1:
        value = float(pieces[0])
    elif len(pieces) == 2:
        value = int(pieces[0]) * 60 + float(pieces[1])
    elif len(pieces) == 3:
        value = int(pieces[0]) * 3600 + int(pieces[1]) * 60 + float(pieces[2])
    else:
        raise ValueError(f"bad span part: {part!r}")
    if value < 0:
        raise ValueError(f"negative span part: {part!r}")
    return value


def parse_span(text: str) -> tuple[float, float]:
    """Parse 'S-E', 'M:SS-M:SS' or 'H:MM:SS-H:MM:SS' into (start, end) seconds."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty span")
    m = _SPAN_RE.match(text)
    if m is None:
        raise ValueError(f"cannot parse span: {text!r}")
    start, end = _clock_seconds(m.group(1)), _clock_seconds(m.group(2))
    if start >= end:
        raise ValueError(f"span start must precede end: {text!r}")
    return start, end


@dataclass(frozen=True)
class ClipInfo:
    label: str
    input: str
    offset_s: float
    duration_s: float
    input_duration_s: float


def _locus_in_range(
    clip_loci: Sequence[tuple[ClipInfo, float, float]], clip_durations: Mapping[str, float]
) -> Check:
    for clip, start, end in clip_loci:
        limit = clip_durations[clip.input] + LOCUS_TOLERANCE_S
        if not (0 <= start <= end and end <= limit):
            return Check(
                name=LOCUS_CHECK,
                result="fail",
                detail=f"locus {clip.input} {start:.3f}-{end:.3f} outside clip bounds {limit:.3f}",
            )
    return Check(name=LOCUS_CHECK, result="pass", detail="all loci within clip bounds")


def claims_from_model(
    payload: object,
    clips: Mapping[str, ClipInfo],
    *,
    id_prefix: str = "c",
) -> tuple[tuple[Claim, ...], tuple[str, ...]]:
    """Convert a model reply payload into claims, repairing or dropping bad parts."""
    problems: list[str] = []
    out: list[Claim] = []
    raw_claims = payload.get("claims") if isinstance(payload, dict) else None
    if not isinstance(raw_claims, list):
        return (), ("payload.claims missing or not a list",)
    clip_durations = {clip.input: clip.duration_s for clip in clips.values()}
    first_clip = next(iter(clips.values()), None)
    for index, item in enumerate(raw_claims):
        cid = f"{id_prefix}{index}"
        if not isinstance(item, dict):
            problems.append(f"{cid}: claim is not an object")
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            problems.append(f"{cid}: missing text")
            continue
        aspect = item.get("aspect")
        if aspect not in ASPECTS:
            problems.append(f"{cid}: unknown aspect {aspect!r} mapped to 'other'")
            aspect = "other"
        scale = item.get("scale")
        if scale not in SCALES:
            problems.append(f"{cid}: unknown scale {scale!r} mapped to 'event'")
            scale = "event"
        basis = item.get("basis")
        if basis not in BASES:
            problems.append(f"{cid}: unknown basis {basis!r} mapped to 'audio'")
            basis = "audio"
        confidence = item.get("confidence")
        if confidence not in CONFIDENCES:
            problems.append(f"{cid}: unknown confidence {confidence!r} mapped to 'low'")
            confidence = "low"
        relation = item.get("relation")
        if relation == "none":
            relation = None
        elif relation is not None and relation not in RELATIONS:
            problems.append(f"{cid}: unknown relation {relation!r} dropped")
            relation = None

        clip_loci: list[tuple[ClipInfo, float, float, str | None]] = []
        raw_loci = item.get("loci")
        if not isinstance(raw_loci, list):
            problems.append(f"{cid}: loci missing or not a list")
            raw_loci = []
        for li, raw_loc in enumerate(raw_loci):
            if not isinstance(raw_loc, dict):
                problems.append(f"{cid}: locus {li} is not an object")
                continue
            clip_label = raw_loc.get("clip")
            clip = clips.get(clip_label) if isinstance(clip_label, str) else None
            if clip is None:
                problems.append(f"{cid}: locus {li} names unknown clip {clip_label!r}")
                continue
            try:
                cstart, cend = float(raw_loc["start"]), float(raw_loc["end"])
            except (KeyError, TypeError, ValueError):
                problems.append(f"{cid}: locus {li} has non-numeric times")
                continue
            voice = raw_loc.get("voice")
            clip_loci.append((clip, cstart, cend, voice if isinstance(voice, str) else None))

        check: Check
        if scale == "work" and not clip_loci:
            if first_clip is None:
                problems.append(f"{cid}: work-scale claim has no clip to ground")
                continue
            clip_loci = [(first_clip, 0.0, first_clip.duration_s, None)]
            check = Check(name=LOCUS_CHECK, result="pass", detail="implicit whole-clip locus")
        elif not clip_loci:
            check = Check(name=LOCUS_CHECK, result="fail", detail="no locus")
        else:
            check = _locus_in_range([(c, s, e) for c, s, e, _ in clip_loci], clip_durations)

        loci = tuple(
            Locus(
                input=clip.input,
                start=max(0.0, s + clip.offset_s),
                end=max(0.0, e + clip.offset_s),
                voice=v,
            )
            for clip, s, e, v in clip_loci
            if e >= 0 and e >= s
        )
        out.append(
            make_claim(
                id=cid,
                origin="model",
                text=text,
                aspect=aspect,
                scale=scale,
                loci=loci,
                relation=relation,
                basis=basis,
                confidence=confidence,
                checks=(check,),
            )
        )
    return tuple(out), tuple(problems)


def _with_check(claim: Claim, check: Check) -> Claim:
    checks = (*claim.checks, check)
    return replace(claim, checks=checks, status=derive_status(claim.loci, checks))


def _map_claims(claims: Sequence[Claim], check_for: Callable[[Claim], Check | None]) -> tuple[Claim, ...]:
    out = []
    for claim in claims:
        extra = check_for(claim)
        out.append(_with_check(claim, extra) if extra is not None else claim)
    return tuple(out)


def apply_identity_check(
    claims: Sequence[Claim],
    *,
    identical: bool,
    measurement_id: str,
) -> tuple[Claim, ...]:
    """PCM identity: 'differs' contradicted, 'unchanged' corroborated, only when identical."""
    if not identical:
        return tuple(claims)

    def check_for(claim: Claim) -> Check | None:
        if claim.relation == "differs":
            return Check(IDENTITY_CHECK, "fail", "inputs are PCM-identical", measurement_id)
        if claim.relation == "unchanged":
            return Check(IDENTITY_CHECK, "pass", "inputs are PCM-identical", measurement_id)
        return None

    return _map_claims(claims, check_for)


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def apply_difference_check(
    claims: Sequence[Claim],
    *,
    regions: Mapping[str, Sequence[tuple[float, float]]],
    global_change: bool,
    measurement_id: str,
    spectral_regions: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    spectral_global: bool = False,
    tolerance_s: float = DIFFERENCE_TOLERANCE_S,
) -> tuple[Claim, ...]:
    """Cross-check differs/unchanged claims against measured difference regions.

    Waveform regions are phase-sensitive: they support "differs" and contradict
    "unchanged" only when the waveform change is local; a global waveform
    residual cannot contradict perceptual sameness. Spectral regions
    (phase-invariant) do both, including when global. With no spectral data
    (older measurements) only the legacy waveform rules apply."""

    def check_for(claim: Claim) -> Check | None:
        if claim.relation not in ("differs", "unchanged"):
            return None
        spectral = spectral_regions is not None
        if claim.relation == "differs":
            if global_change and (not spectral or spectral_global):
                return Check(
                    DIFFERENCE_CHECK,
                    "n/a",
                    "global change: region check not applicable",
                    measurement_id,
                )
            # a global region set covers everything and cannot support a
            # localized "differs"; global spectral still contradicts "unchanged"
            sets: tuple[tuple[Mapping[str, Sequence[tuple[float, float]]], str], ...] = ()
            if not global_change:
                sets += ((regions, "waveform"),)
            if spectral and not spectral_global:
                sets += ((spectral_regions or {}, "spectral"),)
            hit: tuple[str, float, float] | None = None
            for region_map, name in sets:
                for loc in claim.loci:
                    coverage = sum(
                        _overlaps(loc.start - tolerance_s, loc.end + tolerance_s, rs, re_)
                        for rs, re_ in region_map.get(loc.input, ())
                    )
                    required = min(DIFFERS_SUPPORT_FRACTION * (loc.end - loc.start), DIFFERS_SUPPORT_MAX_S)
                    supported = coverage > 0 if loc.end <= loc.start else coverage >= required
                    if supported:
                        hit = (name, coverage, required)
                        break
                if hit is not None:
                    break
            if hit is not None:
                name, coverage, required = hit
                return Check(
                    DIFFERENCE_CHECK,
                    "pass",
                    f"measured {name} difference covers a locus by {coverage:.2f}s "
                    f"(required {required:.2f}s, tolerance {tolerance_s}s)",
                    measurement_id,
                )
            return Check(
                DIFFERENCE_CHECK,
                "fail",
                f"measured difference covers no locus by the required "
                f"min({DIFFERS_SUPPORT_FRACTION} x locus duration, {DIFFERS_SUPPORT_MAX_S}s) "
                f"(tolerance {tolerance_s}s)",
                measurement_id,
            )
        if not spectral:
            # without spectral data the waveform set decides alone
            if global_change:
                return Check(
                    DIFFERENCE_CHECK,
                    "n/a",
                    "global change: region check not applicable",
                    measurement_id,
                )
            breach = any(
                _overlaps(loc.start, loc.end, rs, re_) > 0.5
                for loc in claim.loci
                for rs, re_ in regions.get(loc.input, ())
            )
            if breach:
                return Check(
                    DIFFERENCE_CHECK,
                    "fail",
                    "locus overlaps a measured difference region by >0.5s",
                    measurement_id,
                )
            return Check(
                DIFFERENCE_CHECK,
                "pass",
                "no measured difference region overlaps any locus",
                measurement_id,
            )
        breach = any(
            _overlaps(loc.start, loc.end, rs, re_) > 0.5
            for loc in claim.loci
            for rs, re_ in (spectral_regions or {}).get(loc.input, ())
        )
        if breach:
            return Check(
                DIFFERENCE_CHECK,
                "fail",
                "locus overlaps a measured spectral difference region by >0.5s",
                measurement_id,
            )
        if not global_change and any(
            _overlaps(loc.start, loc.end, rs, re_) > 0.5
            for loc in claim.loci
            for rs, re_ in regions.get(loc.input, ())
        ):
            return Check(
                DIFFERENCE_CHECK,
                "fail",
                "locus overlaps a measured waveform difference region by >0.5s",
                measurement_id,
            )
        return Check(
            DIFFERENCE_CHECK,
            "pass",
            "no measured spectral or local waveform difference region overlaps any locus by >0.5s",
            measurement_id,
        )

    return _map_claims(claims, check_for)


def apply_order_check(
    first: Sequence[Claim],
    second: Sequence[Claim],
    tolerance_s: float = DIFFERENCE_TOLERANCE_S,
) -> tuple[tuple[Claim, ...], tuple[Claim, ...]]:
    """Mark counterbalanced differs-claims stable when the other run agrees."""

    def stable(claim: Claim, others: Sequence[Claim]) -> Check | None:
        if claim.relation != "differs":
            return None
        for other in others:
            if other.relation != "differs" or other.aspect != claim.aspect:
                continue
            same_input_overlap = any(
                loc.input == oloc.input
                and _overlaps(loc.start, loc.end, oloc.start - tolerance_s, oloc.end + tolerance_s) > 0
                for loc in claim.loci
                for oloc in other.loci
            )
            if same_input_overlap:
                return Check(ORDER_CHECK, "pass", "matching differs-claim in counterbalanced run")
        return Check(ORDER_CHECK, "fail", "no matching differs-claim in counterbalanced run")

    return _map_claims(first, lambda c: stable(c, second)), _map_claims(second, lambda c: stable(c, first))


def _round_floats(value: object) -> object:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, Mapping):
        return {k: _round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(v) for v in value]
    return value


_OMIT_IF_NONE: tuple[tuple[type, str], ...] = (
    (Locus, "voice"),
    (BriefLocus, "voice"),
    (Check, "measurement"),
)


def _encode_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, (tuple, list)):
        return [_encode_value(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _encode_value(v) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return _encode_record(value)
    return value


def _encode_record(obj: object) -> dict[str, Any]:
    cls = type(obj)
    out: dict[str, Any] = {}
    for field in fields(cast(Any, type(obj))):
        value = getattr(obj, field.name)
        if value is None and (cls, field.name) in _OMIT_IF_NONE:
            continue
        out[field.name] = _encode_value(value)
    return out


def to_json(evidence: Evidence) -> dict[str, Any]:
    """Serialize an Evidence to a deterministic dict (key order fixed, seconds 3dp)."""
    return _encode_record(evidence)


def _need(cond: bool, message: str) -> None:
    if not cond:
        raise ContractError(message)


def _need_type(value: object, typ: type | tuple[type, ...], field_name: str) -> Any:
    name = typ.__name__ if isinstance(typ, type) else "|".join(t.__name__ for t in typ)
    _need(isinstance(value, typ), f"{field_name}: expected {name}, got {type(value).__name__}")
    return value


def _decode_record(cls: type, raw: object, fname: str) -> Any:
    d = _need_type(raw, dict, fname)
    hints = get_type_hints(cls)
    unknown = sorted(set(d) - set(hints))
    _need(not unknown, f"{fname}: unknown keys {unknown}")
    defaults = {f.name: f.default for f in fields(cls) if f.default is not MISSING}
    kwargs = {
        name: (
            defaults[name]
            if name not in d and name in defaults
            else _decode_value(tp, d.get(name), f"{fname}.{name}")
        )
        for name, tp in hints.items()
    }
    return cls(**kwargs)


def _decode_value(tp: object, value: object, fname: str) -> Any:
    if tp is Any:
        return value
    origin = get_origin(tp)
    if isinstance(tp, UnionType) or origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if value is None:
            return None
        _need(len(args) == 1, f"{fname}: unsupported union {tp}")
        return _decode_value(args[0], value, fname)
    if isinstance(tp, type) and is_dataclass(tp):
        return _decode_record(tp, value, fname)
    if tp is str:
        return _need_type(value, str, fname)
    if tp is bool:
        return _need_type(value, bool, fname)
    if tp is int:
        return int(_need_type(value, int, fname))
    if tp is float:
        return float(_need_type(value, (int, float), fname))
    if origin is tuple:
        elem = get_args(tp)[0]
        return tuple(_decode_value(elem, item, fname) for item in _need_type(value, list, fname))
    if origin in (Mapping, dict):
        d = _need_type(value, dict, fname)
        map_args = get_args(tp)
        value_type = map_args[1] if len(map_args) == 2 else Any
        return {str(k): _decode_value(value_type, v, f"{fname}.{k}") for k, v in d.items()}
    _need(False, f"{fname}: unsupported field type {tp!r}")
    return None


def from_json(data: object) -> Evidence:
    """Parse a serialized evidence dict; ContractError on any contract violation."""
    d = _need_type(data, dict, "evidence")
    _need(d.get("schema") == SCHEMA, f"schema must be {SCHEMA!r}")
    _need(d.get("version") == VERSION, f"version must be {VERSION}")
    result: Evidence = _decode_record(Evidence, d, "evidence")
    return result


def dumps(evidence: Evidence) -> str:
    return json.dumps(to_json(evidence), indent=2, ensure_ascii=False)
