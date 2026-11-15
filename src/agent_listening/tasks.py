"""Task orchestration: the five listening tasks plus identify and check."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_listening import __version__, audio, evidence, gemini, prompts
from agent_listening.evidence import (
    SCHEMA,
    VERSION,
    Assessment,
    Brief,
    BriefLocus,
    ClipInfo,
    ContextFile,
    ErrorBody,
    Evidence,
    Focus,
    Goal,
    Input,
    Measurement,
    Producer,
    Protection,
    Request,
    Tool,
)

TOOL = Tool(name="agent-listening", version=__version__)

SC_EXTS = frozenset({".scd", ".sc"})
SCORE_EXTS = frozenset({".musicxml", ".xml", ".krn", ".abc", ".mei"})
MEASURE_KINDS = ("level", "spectrum", "stereo", "onsets")

LIM_MONO = "MODEL_INPUT_MONO"
LIM_RESOLUTION = "MODEL_INPUT_REDUCED_RESOLUTION"
LIM_TIME = "MODEL_TIME_APPROXIMATE"
LIM_CONTEXT = "CONTEXT_SUPPLIED"
LIM_DIFF_UNAVAILABLE = "DIFFERENCE_CHECK_UNAVAILABLE"
LIM_DRY_RUN = "DRY_RUN_NO_MODEL_CALL"
LIM_FOLLOW_UP = "MEASUREMENT_GUIDED_FOLLOW_UP"


class UsageError(Exception):
    """Bad caller input; maps to exit code 2."""


@dataclass(frozen=True)
class Options:
    model: str = "gemini-4-hydrogen"
    thinking: str = "high"
    cache_dir: Path | str = "~/.cache/agent-listening"
    dry_run: bool = False
    executable: str = "gemini-api"

    def __post_init__(self) -> None:
        if self.thinking not in evidence.THINKING_LEVELS:
            raise UsageError(f"thinking must be one of {evidence.THINKING_LEVELS}")
        expanded = Path(os.path.expandvars(str(self.cache_dir))).expanduser()
        object.__setattr__(self, "cache_dir", expanded)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _contexts(paths: Sequence[str]) -> tuple[tuple[ContextFile, ...], tuple[prompts.ContextView, ...]]:
    files = []
    views = []
    for index, raw in enumerate(paths):
        p = Path(raw)
        data = p.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UsageError(f"context file is not UTF-8 text: {p}") from exc
        suffix = p.suffix.lower()
        kind = "sc-source" if suffix in SC_EXTS else "score" if suffix in SCORE_EXTS else "notes"
        label = f"S{index + 1}"
        files.append(
            ContextFile(
                label=label, kind=kind, path=str(p), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)
            )
        )
        views.append(prompts.ContextView(label=label, kind=kind, text=text))
    return tuple(files), tuple(views)


def _renumber(measurements: Sequence[Measurement]) -> tuple[Measurement, ...]:
    return tuple(replace(m, id=f"m{i + 1}") for i, m in enumerate(measurements))


def _clip_info(inp: Input, tr: evidence.Transport) -> ClipInfo:
    return ClipInfo(
        label=inp.label,
        input=inp.label,
        offset_s=tr.offset_s,
        duration_s=tr.duration_s,
        input_duration_s=inp.format.duration_s,
    )


def _clip_view(inp: Input, tr: evidence.Transport) -> prompts.ClipView:
    return prompts.ClipView(
        label=inp.label,
        role=inp.role,
        duration_s=tr.duration_s,
        offset_s=tr.offset_s,
        source_duration_s=inp.format.duration_s,
        channels=tr.channels,
    )


def _limitations(
    transports: Sequence[evidence.Transport],
    contexts: Sequence[ContextFile],
    inputs: Sequence[Input],
    called_model: bool,
    extra: Sequence[str],
) -> tuple[str, ...]:
    out: list[str] = []
    if any(t.channels > 1 for t in transports):
        out.append(LIM_MONO)
    for inp in inputs:
        if not inp.format.subtype.startswith("PCM"):
            out.append(f"TRANSPORT_QUANTIZED_TO_PCM24: {inp.label}")
    if called_model:
        out += [LIM_RESOLUTION, LIM_TIME]
    if contexts:
        out.append(LIM_CONTEXT)
    out.extend(extra)
    return tuple(out)


def _producer(
    opts: Options, replies: Sequence[gemini.Reply], system: str, prompt: str, schema: Mapping[str, object]
) -> Producer:
    instructions = system + "\n" + prompt + "\n" + json.dumps(schema, sort_keys=True, ensure_ascii=False)
    usage = {
        key: sum(r.usage.get(key, 0) for r in replies)
        for key in ("input_tokens", "audio_tokens", "output_tokens", "thought_tokens")
    }
    return Producer(
        provider="gemini-api",
        cli_version=replies[0].cli_version if replies else "unknown",
        model=opts.model,
        thinking=opts.thinking,
        instructions_sha256=hashlib.sha256(instructions.encode()).hexdigest(),
        interaction_ids=tuple(r.interaction_id for r in replies if r.interaction_id is not None),
        usage=usage,
        latency_s=sum(r.latency_s for r in replies),
        attempts=sum(r.attempts for r in replies),
    )


def _evidence(
    *,
    task: str,
    status: str,
    request: Request,
    inputs: Sequence[Input] = (),
    contexts: Sequence[ContextFile] = (),
    brief: Brief | None = None,
    transports: Sequence[evidence.Transport] = (),
    measurements: Sequence[Measurement] = (),
    claims: Sequence[evidence.Claim] = (),
    assessments: Sequence[Assessment] = (),
    answer: str | None = None,
    summary: str = "",
    producer: Producer | None = None,
    limitations: Sequence[str] = (),
    error: ErrorBody | None = None,
) -> Evidence:
    """The single Evidence construction point; measurements renumbered here."""
    return Evidence(
        schema=SCHEMA,
        version=VERSION,
        task=task,
        status=status,
        tool=TOOL,
        created_at=_now(),
        inputs=tuple(inputs),
        context=tuple(contexts),
        brief=brief,
        request=request,
        transport=tuple(transports),
        measurements=_renumber(measurements),
        claims=tuple(claims),
        assessments=tuple(assessments),
        answer=answer,
        summary=summary,
        producer=producer,
        limitations=tuple(limitations),
        error=error,
    )


def _failed_evidence(
    *,
    task: str,
    inputs: Sequence[Input],
    contexts: Sequence[ContextFile],
    brief: Brief | None,
    request: Request,
    transports: Sequence[evidence.Transport],
    measurements: Sequence[Measurement],
    error: gemini.ProviderError,
    limitations: Sequence[str],
) -> Evidence:
    return _evidence(
        task=task,
        status="failed",
        request=request,
        inputs=inputs,
        contexts=contexts,
        brief=brief,
        transports=transports,
        measurements=measurements,
        limitations=limitations,
        error=ErrorBody(code=error.code, message=error.message),
    )


def _problems_limitations(problems: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"MODEL_OUTPUT_ADJUSTED: {p}" for p in problems)


def _identify_inputs(specs: Sequence[tuple[str, str, str]]) -> tuple[Input, ...]:
    return tuple(audio.identify(path, label, role) for path, label, role in specs)


def _transports(
    inputs: Sequence[Input], opts: Options, focuses: Mapping[str, tuple[float, float]]
) -> tuple[evidence.Transport, ...]:
    return tuple(audio.transport(inp, opts.cache_dir, focuses.get(inp.label)) for inp in inputs)


def _listen_or_dry(
    opts: Options,
    built: prompts.ListeningPrompt,
    transports: Sequence[evidence.Transport],
) -> gemini.Reply | None:
    if opts.dry_run:
        return None
    return gemini.listen(
        executable=opts.executable,
        model=opts.model,
        thinking=opts.thinking,
        system=built.system,
        prompt=built.prompt,
        audio=[Path(t.path) for t in transports],
        schema=built.schema,
    )


def hear(
    audio_path: str,
    *,
    focus: tuple[float, float] | None,
    contexts: Sequence[str],
    options: Options,
) -> tuple[Evidence, int]:
    inputs = _identify_inputs([(audio_path, "A", "subject")])
    ctx_files, ctx_views = _contexts(contexts)
    focuses = {"A": focus} if focus is not None else {}
    transports = _transports(inputs, options, focuses)
    clips = {inp.label: _clip_info(inp, tr) for inp, tr in zip(inputs, transports, strict=True)}
    built = prompts.build(
        "hear", [_clip_view(inp, tr) for inp, tr in zip(inputs, transports, strict=True)], contexts=ctx_views
    )
    request = Request(
        question=None, focus=Focus(*focus) if focus else None, measures=(), counterbalance=False
    )
    return _run_single(
        task="hear",
        inputs=inputs,
        contexts=ctx_files,
        brief=None,
        request=request,
        transports=transports,
        measurements=(),
        clips=clips,
        built=built,
        options=options,
        post=None,
    )


def ask(
    audio_path: str,
    question: str,
    *,
    focus: tuple[float, float] | None,
    contexts: Sequence[str],
    options: Options,
) -> tuple[Evidence, int]:
    inputs = _identify_inputs([(audio_path, "A", "subject")])
    ctx_files, ctx_views = _contexts(contexts)
    focuses = {"A": focus} if focus is not None else {}
    transports = _transports(inputs, options, focuses)
    clips = {inp.label: _clip_info(inp, tr) for inp, tr in zip(inputs, transports, strict=True)}
    built = prompts.build(
        "ask",
        [_clip_view(inp, tr) for inp, tr in zip(inputs, transports, strict=True)],
        question=question,
        contexts=ctx_views,
    )
    request = Request(
        question=question, focus=Focus(*focus) if focus else None, measures=(), counterbalance=False
    )
    return _run_single(
        task="ask",
        inputs=inputs,
        contexts=ctx_files,
        brief=None,
        request=request,
        transports=transports,
        measurements=(),
        clips=clips,
        built=built,
        options=options,
        post=None,
    )


def _run_single(
    *,
    task: str,
    inputs: Sequence[Input],
    contexts: Sequence[ContextFile],
    brief: Brief | None,
    request: Request,
    transports: Sequence[evidence.Transport],
    measurements: Sequence[Measurement],
    clips: Mapping[str, ClipInfo],
    built: prompts.ListeningPrompt,
    options: Options,
    post: Callable[[tuple[evidence.Claim, ...]], tuple[evidence.Claim, ...]] | None,
    assess: Callable[
        [Mapping[str, object], tuple[evidence.Claim, ...]], tuple[tuple[Assessment, ...], tuple[str, ...]]
    ]
    | None = None,
    follow_up: Callable[
        [tuple[evidence.Claim, ...]],
        tuple[
            tuple[evidence.Claim, ...],
            tuple[evidence.Transport, ...],
            list[gemini.Reply],
            list[str],
        ],
    ]
    | None = None,
    base_limitations: Sequence[str] = (),
) -> tuple[Evidence, int]:
    extra_limitations: list[str] = list(base_limitations)
    try:
        reply = _listen_or_dry(options, built, transports)
    except gemini.ProviderError as exc:
        return (
            _failed_evidence(
                task=task,
                inputs=inputs,
                contexts=contexts,
                brief=brief,
                request=request,
                transports=transports,
                measurements=_renumber(measurements),
                error=exc,
                limitations=_limitations(transports, contexts, inputs, False, extra_limitations),
            ),
            3,
        )
    claims: tuple[evidence.Claim, ...] = ()
    assessments: tuple[Assessment, ...] = ()
    problems: tuple[str, ...] = ()
    summary = ""
    answer = None
    producer = None
    if reply is None:
        extra_limitations.append(LIM_DRY_RUN)
        status = "partial"
    else:
        claims, problems = evidence.claims_from_model(reply.payload, clips)
        if post is not None:
            claims = post(claims)
        if assess is not None:
            assessments, assess_problems = assess(reply.payload, claims)
            problems = (*problems, *assess_problems)
        extra_limitations.extend(_problems_limitations(problems))
        replies = [reply]
        if follow_up is not None:
            f_claims, f_transports, f_replies, f_limits = follow_up(claims)
            claims = (*claims, *f_claims)
            transports = (*transports, *f_transports)
            replies += f_replies
            extra_limitations.extend(f_limits)
        summary = str(reply.payload.get("summary") or "")
        raw_answer = reply.payload.get("answer")
        if isinstance(raw_answer, str):
            answer = raw_answer
        elif task in ("ask", "investigate"):
            extra_limitations.append("MODEL_OUTPUT_ADJUSTED: missing answer field")
        producer = _producer(options, replies, built.system, built.prompt, built.schema)
        status = "complete"
    limitations = _limitations(transports, contexts, inputs, reply is not None, extra_limitations)
    return (
        _evidence(
            task=task,
            status=status,
            request=request,
            inputs=inputs,
            contexts=contexts,
            brief=brief,
            transports=transports,
            measurements=measurements,
            claims=claims,
            assessments=assessments,
            answer=answer,
            summary=summary,
            producer=producer,
            limitations=limitations,
        ),
        0,
    )


def _difference_regions(
    measurement: Measurement | None,
) -> tuple[Mapping[str, Sequence[tuple[float, float]]], bool]:
    """Regions per input label in each input's own time base."""
    if measurement is None or not measurement.result.get("available"):
        return {}, False
    lag = float(measurement.result.get("lag_s", 0.0))
    a_regions = [(float(r["start"]), float(r["end"])) for r in measurement.result.get("regions", [])]
    b_regions = [(s + lag, e + lag) for s, e in a_regions]
    return {"A": a_regions, "B": b_regions}, bool(measurement.result.get("global_change"))


def compare(
    audio_a: str,
    audio_b: str,
    *,
    question: str | None,
    contexts: Sequence[str],
    counterbalance: bool,
    follow_up: int = 2,
    options: Options,
) -> tuple[Evidence, int]:
    if follow_up < 0:
        raise UsageError("--follow-up must be >= 0")
    inputs = _identify_inputs([(audio_a, "A", "before"), (audio_b, "B", "after")])
    ctx_files, ctx_views = _contexts(contexts)
    measurements: list[Measurement] = [audio.pcm_identity(inputs[0], inputs[1])]
    request = Request(
        question=question, focus=None, measures=(), counterbalance=counterbalance, follow_up=follow_up
    )
    if measurements[0].result["identical"]:
        return (
            _evidence(
                task="compare",
                status="complete",
                request=request,
                inputs=inputs,
                contexts=ctx_files,
                measurements=measurements,
                summary="Inputs are PCM-identical; no perceptible difference is possible.",
                limitations=_limitations((), ctx_files, inputs, False, ()),
            ),
            0,
        )
    diff = audio.signal_difference(inputs[0], inputs[1])
    measurements.append(diff)
    regions, global_change = _difference_regions(diff)
    extra: list[str] = [] if diff.result.get("available") else [LIM_DIFF_UNAVAILABLE]
    transports = _transports(inputs, options, {})
    clips = {inp.label: _clip_info(inp, tr) for inp, tr in zip(inputs, transports, strict=True)}
    built = prompts.build(
        "compare",
        [_clip_view(inputs[0], transports[0]), _clip_view(inputs[1], transports[1])],
        question=question,
        contexts=ctx_views,
    )

    def post(claims: tuple[evidence.Claim, ...]) -> tuple[evidence.Claim, ...]:
        numbered = _renumber(measurements)
        diff_id = numbered[1].id
        return evidence.apply_difference_check(
            claims, regions=regions, global_change=global_change, measurement_id=diff_id
        )

    def follow_ups(
        checked: tuple[evidence.Claim, ...],
    ) -> tuple[tuple[evidence.Claim, ...], tuple[evidence.Transport, ...], list[gemini.Reply], list[str]]:
        """Re-hear measured regions no consistent differs-claim covers, as matched excerpts."""
        if options.dry_run or follow_up <= 0 or not diff.result.get("available") or global_change:
            return (), (), [], []
        tolerance = evidence.DIFFERENCE_TOLERANCE_S
        uncovered = [
            region
            for region in diff.result.get("regions", [])
            if "only_in" not in region
            and not any(
                c.relation == "differs"
                and c.status == "consistent"
                and any(
                    max(
                        0.0,
                        min(loc.end, float(region["end"]) + tolerance)
                        - max(loc.start, float(region["start"]) - tolerance),
                    )
                    > 0
                    for loc in c.loci
                    if loc.input == "A"
                )
                for c in checked
            )
        ]
        uncovered.sort(key=lambda r: (-float(r.get("peak_db", 0.0)), float(r["start"])))
        lag = float(diff.result.get("lag_s", 0.0))
        diff_id = f"m{len(measurements)}"
        out_claims: list[evidence.Claim] = []
        out_transports: list[evidence.Transport] = []
        replies: list[gemini.Reply] = []
        limitations: list[str] = []
        for n, region in enumerate(uncovered[:follow_up]):
            start, end = float(region["start"]), float(region["end"])
            window_a = (max(0.0, start - 2.0), min(inputs[0].format.duration_s, end + 2.0))
            window_b = (
                max(0.0, window_a[0] + lag),
                min(inputs[1].format.duration_s, window_a[1] + lag),
            )
            tr_a = audio.transport(inputs[0], options.cache_dir, window_a)
            tr_b = audio.transport(inputs[1], options.cache_dir, window_b)
            built_f = prompts.build(
                "compare-focus",
                [_clip_view(inputs[0], tr_a), _clip_view(inputs[1], tr_b)],
                region=(start - window_a[0], end - window_a[0]),
            )
            try:
                reply = _listen_or_dry(options, built_f, [tr_a, tr_b])
            except gemini.ProviderError as exc:
                limitations.append(f"FOLLOW_UP_FAILED: {exc.code}")
                break
            if reply is None:
                break
            replies.append(reply)
            out_transports += [tr_a, tr_b]
            clip_infos = {"A": _clip_info(inputs[0], tr_a), "B": _clip_info(inputs[1], tr_b)}
            run_claims, run_problems = evidence.claims_from_model(
                reply.payload, clip_infos, id_prefix=f"f{n}_"
            )
            run_claims = evidence.apply_difference_check(
                run_claims, regions=regions, global_change=False, measurement_id=diff_id
            )
            guided = evidence.Check(
                name="measurement-guided",
                result="pass",
                detail=(f"re-heard after measured difference region {start:.2f}-{end:.2f} s was pointed out"),
                measurement=diff_id,
            )
            for claim in run_claims:
                checks = (*claim.checks, guided)
                out_claims.append(
                    replace(claim, checks=checks, status=evidence.derive_status(claim.loci, checks))
                )
            limitations.extend(_problems_limitations(run_problems))
        if replies:
            limitations.append(LIM_FOLLOW_UP)
        return tuple(out_claims), tuple(out_transports), replies, limitations

    if not counterbalance:
        return _run_single(
            task="compare",
            inputs=inputs,
            contexts=ctx_files,
            brief=None,
            request=request,
            transports=transports,
            measurements=measurements,
            clips=clips,
            built=built,
            options=options,
            post=post,
            follow_up=follow_ups,
            base_limitations=extra,
        )

    diff_id = f"m{len(measurements)}"

    def _compare_run(
        order_indices: tuple[int, int], prefix: str
    ) -> tuple[gemini.Reply | None, tuple[evidence.Claim, ...], tuple[str, ...], str]:
        """One listening call: clips and audio in the same presentation order."""
        views = [_clip_view(inputs[i], transports[i]) for i in order_indices]
        run_transport = [transports[i] for i in order_indices]
        built_run = prompts.build("compare", views, question=question, contexts=ctx_views)
        reply = _listen_or_dry(options, built_run, run_transport)
        if reply is None:
            return None, (), (), ""
        run_claims, run_problems = evidence.claims_from_model(reply.payload, clips, id_prefix=prefix)
        run_claims = evidence.apply_difference_check(
            run_claims, regions=regions, global_change=global_change, measurement_id=diff_id
        )
        return reply, run_claims, tuple(run_problems), str(reply.payload.get("summary") or "")

    extra_limitations = list(extra)
    replies: list[gemini.Reply] = []
    all_claims: list[tuple[evidence.Claim, ...]] = []
    problems: list[str] = []
    summaries: list[str] = []
    for run_index, order_indices in enumerate(((0, 1), (1, 0))):
        try:
            reply, run_claims, run_problems, run_summary = _compare_run(
                order_indices, "c" if run_index == 0 else "k"
            )
        except gemini.ProviderError as exc:
            return (
                _failed_evidence(
                    task="compare",
                    inputs=inputs,
                    contexts=ctx_files,
                    brief=None,
                    request=request,
                    transports=transports,
                    measurements=measurements,
                    error=exc,
                    limitations=_limitations(transports, ctx_files, inputs, False, extra_limitations),
                ),
                3,
            )
        if reply is None:
            continue
        replies.append(reply)
        all_claims.append(run_claims)
        problems.extend(run_problems)
        summaries.append(run_summary)
    first, second = [*all_claims, (), ()][:2]
    first, second = evidence.apply_order_check(first, second)
    claims = tuple(first) + tuple(second)
    if not replies:
        extra_limitations.append(LIM_DRY_RUN)
        status = "partial"
        summary = ""
        producer = None
    else:
        f_claims, f_transports, f_replies, f_limits = follow_ups(claims)
        claims = (*claims, *f_claims)
        transports = (*transports, *f_transports)
        replies += f_replies
        extra_limitations.extend(f_limits)
        status = "complete"
        summary = summaries[0] if summaries else ""
        producer = _producer(options, replies, built.system, built.prompt, built.schema)
    extra_limitations.extend(_problems_limitations(problems))
    limitations = _limitations(transports, ctx_files, inputs, bool(replies), extra_limitations)
    return (
        _evidence(
            task="compare",
            status=status,
            request=request,
            inputs=inputs,
            contexts=ctx_files,
            transports=transports,
            measurements=measurements,
            claims=claims,
            summary=summary,
            producer=producer,
            limitations=limitations,
        ),
        0,
    )


def investigate(
    audio_path: str,
    *,
    focus: tuple[float, float],
    measures: Sequence[str],
    question: str | None,
    contexts: Sequence[str],
    options: Options,
) -> tuple[Evidence, int]:
    kinds = tuple(measures) if measures else MEASURE_KINDS
    for kind in kinds:
        if kind not in MEASURE_KINDS:
            raise UsageError(f"unknown measure: {kind} (choices: {', '.join(MEASURE_KINDS)})")
    inputs = _identify_inputs([(audio_path, "A", "subject")])
    ctx_files, ctx_views = _contexts(contexts)
    transports = _transports(inputs, options, {"A": focus})
    clips = {inp.label: _clip_info(inp, tr) for inp, tr in zip(inputs, transports, strict=True)}
    measurements = audio.describe(inputs[0], focus, kinds)
    prompt_measurements = {m.kind: _clip_relative(m.result, focus[0]) for m in measurements}
    built = prompts.build(
        "investigate",
        [_clip_view(inp, tr) for inp, tr in zip(inputs, transports, strict=True)],
        question=question,
        contexts=ctx_views,
        measurements=prompt_measurements,
    )
    request = Request(question=question, focus=Focus(*focus), measures=kinds, counterbalance=False)
    return _run_single(
        task="investigate",
        inputs=inputs,
        contexts=ctx_files,
        brief=None,
        request=request,
        transports=transports,
        measurements=measurements,
        clips=clips,
        built=built,
        options=options,
        post=None,
    )


def _clip_relative(value: object, offset_s: float) -> object:
    if isinstance(value, dict):
        return {
            k: (
                round(v - offset_s, 3)
                if k == "t" and isinstance(v, (int, float))
                else _clip_relative(v, offset_s)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_clip_relative(v, offset_s) for v in value]
    return value


def _parse_brief(brief_path: str) -> tuple[Brief, prompts.BriefView]:
    p = Path(brief_path)
    data = p.read_bytes()
    try:
        raw = json.loads(data)
    except json.JSONDecodeError as exc:
        raise UsageError(f"brief is not valid JSON: {p}") from exc
    if not isinstance(raw, dict):
        raise UsageError(f"brief must be a JSON object: {p}")
    goals: list[Goal] = []
    for i, item in enumerate(raw.get("goals") or []):
        if isinstance(item, str):
            goals.append(Goal(id=f"g{i + 1}", text=item))
        elif isinstance(item, dict):
            goals.append(Goal(id=str(item.get("id") or f"g{i + 1}"), text=str(item.get("text") or "")))
        else:
            raise UsageError(f"brief goals[{i}] must be a string or object")
    protect: list[Protection] = []
    for i, item in enumerate(raw.get("protect") or []):
        if not isinstance(item, dict):
            raise UsageError(f"brief protect[{i}] must be an object")
        loci = tuple(
            BriefLocus(start=float(loc["start"]), end=float(loc["end"]), voice=loc.get("voice"))
            for loc in item.get("loci") or []
        )
        protect.append(
            Protection(id=str(item.get("id") or f"p{i + 1}"), text=str(item.get("text") or ""), loci=loci)
        )
    if not goals and not protect:
        raise UsageError("brief needs at least one goal or protected relationship")
    brief = Brief(
        path=str(p),
        sha256=hashlib.sha256(data).hexdigest(),
        goals=tuple(goals),
        protect=tuple(protect),
    )
    view = prompts.BriefView(
        goals=[(g.id, g.text) for g in goals],
        protect=[(pr.id, pr.text) for pr in protect],
    )
    return brief, view


def review(
    audio_path: str,
    *,
    brief_path: str,
    before: str | None,
    contexts: Sequence[str],
    options: Options,
) -> tuple[Evidence, int]:
    brief, brief_view = _parse_brief(brief_path)
    specs = [(audio_path, "A", "subject")]
    if before is not None:
        specs.append((before, "B", "before"))
    inputs = _identify_inputs(specs)
    ctx_files, ctx_views = _contexts(contexts)
    measurements: list[Measurement] = []
    regions: Mapping[str, Sequence[tuple[float, float]]] = {}
    global_change = False
    identical = False
    extra: list[str] = []
    if before is not None:
        identity = audio.pcm_identity(inputs[0], inputs[1])
        measurements.append(identity)
        identical = bool(identity.result["identical"])
        if identical:
            extra.append(LIM_DIFF_UNAVAILABLE)
        else:
            diff = audio.signal_difference(inputs[0], inputs[1])
            measurements.append(diff)
            regions, global_change = _difference_regions(diff)
            if not diff.result.get("available"):
                extra.append(LIM_DIFF_UNAVAILABLE)
    transports = _transports(inputs, options, {})
    clips = {inp.label: _clip_info(inp, tr) for inp, tr in zip(inputs, transports, strict=True)}
    built = prompts.build(
        "review",
        [_clip_view(inp, tr) for inp, tr in zip(inputs, transports, strict=True)],
        contexts=ctx_views,
        brief=brief_view,
    )
    request = Request(question=None, focus=None, measures=(), counterbalance=False)

    def post(claims: tuple[evidence.Claim, ...]) -> tuple[evidence.Claim, ...]:
        if before is None:
            return claims
        numbered = _renumber(measurements)
        out = evidence.apply_identity_check(claims, identical=identical, measurement_id=numbered[0].id)
        if len(numbered) > 1:
            out = evidence.apply_difference_check(
                out, regions=regions, global_change=global_change, measurement_id=numbered[1].id
            )
        return out

    def assess(
        payload: Mapping[str, object], claims: tuple[evidence.Claim, ...]
    ) -> tuple[tuple[Assessment, ...], tuple[str, ...]]:
        problems: list[str] = []
        claim_ids = {c.id for c in claims}
        targets = [g.id for g in brief.goals] + [p.id for p in brief.protect]
        done: dict[str, Assessment] = {}
        raw_items = payload.get("assessments")
        if not isinstance(raw_items, list):
            problems.append("assessments missing or not a list")
            raw_items = []
        for ai, item in enumerate(raw_items):
            if not isinstance(item, dict):
                problems.append(f"assessment {ai} is not an object")
                continue
            target = item.get("target")
            if target not in targets:
                problems.append(f"assessment {ai} names unknown target {target!r}")
                continue
            verdict = item.get("verdict")
            if verdict not in evidence.VERDICTS:
                problems.append(f"assessment {ai}: unknown verdict {verdict!r} mapped to 'unclear'")
                verdict = "unclear"
            cited: list[str] = []
            raw_indices = item.get("claims")
            if not isinstance(raw_indices, list):
                problems.append(f"assessment {ai}: claims is not a list")
                raw_indices = []
            for raw_index in raw_indices:
                cid = f"c{raw_index}" if isinstance(raw_index, int) else None
                if cid is None or cid not in claim_ids:
                    problems.append(f"assessment {ai}: invalid claim index {raw_index!r}")
                    continue
                cited.append(cid)
            done[str(target)] = Assessment(
                target=str(target),
                verdict=str(verdict),
                claims=tuple(cited),
                note=str(item.get("note") or ""),
            )
        for target in targets:
            if target not in done:
                done[target] = Assessment(
                    target=target, verdict="unclear", claims=(), note="no assessment returned"
                )
        return tuple(done[t] for t in targets), tuple(problems)

    return _run_single(
        task="review",
        inputs=inputs,
        contexts=ctx_files,
        brief=brief,
        request=request,
        transports=transports,
        measurements=measurements,
        clips=clips,
        built=built,
        options=options,
        post=post,
        assess=assess,
        base_limitations=extra,
    )


def identify(paths: Sequence[str]) -> tuple[Evidence, int]:
    inputs = _identify_inputs([(p, chr(ord("A") + i), "subject") for i, p in enumerate(paths)])
    return (
        _evidence(
            task="identify",
            status="complete",
            request=Request(question=None, focus=None, measures=(), counterbalance=False),
            inputs=inputs,
            summary=f"Identified {len(inputs)} input{'s' if len(inputs) != 1 else ''}.",
        ),
        0,
    )


def doctor(executable: str) -> dict[str, Any]:
    info: dict[str, Any] = {"executable": executable, "version": None, "supported": False}
    try:
        ver = gemini.version(executable)
        info["version"] = ver
        info["supported"] = ver.split(".")[:2] == gemini.SUPPORTED_SERIES.split(".")
    except gemini.ProviderError:
        pass
    return {
        "gemini_api": info,
        "credential": gemini.credential_status(executable),
        "default_model": Options().model,
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def check(evidence_path: str, audio_paths: Sequence[str]) -> tuple[dict[str, Any], int]:
    """Re-hash inputs of an existing evidence document."""
    raw = Path(evidence_path).read_bytes()
    data = json.loads(raw)
    ev = evidence.from_json(data)
    rows = []
    status = "verified"
    for index, inp in enumerate(ev.inputs):
        path = audio_paths[index] if index < len(audio_paths) else inp.path
        p = Path(path)
        if not p.is_file():
            rows.append({"label": inp.label, "path": str(p), "expected": inp.sha256, "actual": None})
            if status == "verified":
                status = "missing"
            continue
        digest = _sha256_path(p)
        rows.append({"label": inp.label, "path": str(p), "expected": inp.sha256, "actual": digest})
        if digest != inp.sha256 and status != "missing":
            status = "mismatch"
    return (
        {"schema": "agent-listening/check", "version": 1, "status": status, "inputs": rows},
        0 if status == "verified" else 1,
    )
