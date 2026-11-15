from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agent_listening.evidence import ASPECTS, BASES, CONFIDENCES, RELATIONS, SCALES, VERDICTS

CONTEXT_CHAR_LIMIT = 60_000

INSTRUCTIONS = """\
You are the listening component of Agent Listening. You receive audio clips and report only what is \
perceptible in them. Downstream agents revise music based on your claims, so a wrong claim is worse \
than a missing one.

Rules:
1. Ground every claim in the audio. Give each claim one or more loci: a clip label plus start and end \
in seconds from the start of that clip as you heard it. Use the whole clip only for claims that \
genuinely describe the whole clip.
2. Your timestamps are approximate. Give the tightest interval you can actually support; widen it \
rather than guess a precise moment.
3. If something cannot be determined from the audio, say so in the summary or answer instead of \
guessing, and lower your confidence when unsure. Never invent events to fill the response.
4. Describe music in its own terms. Do not assume a tonal key, a regular meter, equal temperament, \
Western instrument names or popular-song sections. If the pulse is free or the tuning is not 12-tone \
equal temperament, say so. Name voices by register, timbre and role (for example "low bowed drone", \
"upper reed line") unless the instrument is clearly identifiable.
5. Cover the timescales that matter: sound events (attack, decay, resonance, articulation), phrases \
and gestures, sections, and the whole work. Connect loci with relations (repeats, varies, imitates, \
answers, doubles, accompanies, contrasts, transforms, returns), listing the earlier locus first.
6. Mark the basis of each claim: "audio" if you heard it, "context" if it comes only from supplied \
text, "audio+context" if the text pointed you to something you then heard. Supplied text describes \
intentions, not sound. Never report a property or change as heard because the text says it exists.
7. Absolute pitch names and exact frequencies are unreliable for you. Prefer relative descriptions \
(higher, lower, interval size, detuned, beating) unless you are certain.
8. When comparing clips, report each perceptible difference as its own claim with relation "differs" \
and loci in both clips where possible, and important preserved properties with relation "unchanged". \
If you hear no difference, say so and return no "differs" claims.
"""


@dataclass(frozen=True)
class ClipView:
    label: str
    role: str
    duration_s: float
    offset_s: float
    source_duration_s: float
    channels: int


@dataclass(frozen=True)
class ContextView:
    label: str
    kind: str
    text: str


@dataclass(frozen=True)
class BriefView:
    goals: Sequence[tuple[str, str]]
    protect: Sequence[tuple[str, str]]


@dataclass(frozen=True)
class ListeningPrompt:
    system: str
    prompt: str
    schema: Mapping[str, object]


def _clip_line(clip: ClipView) -> str:
    excerpt = ""
    if clip.offset_s > 0 or clip.duration_s < clip.source_duration_s:
        excerpt = (
            f", an excerpt starting {clip.offset_s:.2f} s into a {clip.source_duration_s:.2f} s recording; "
            "give times relative to the start of the clip"
        )
    return f"- Clip {clip.label} ({clip.role}): {clip.duration_s:.2f} s{excerpt}."


def _context_block(context: ContextView) -> str:
    text = context.text
    note = ""
    if len(text) > CONTEXT_CHAR_LIMIT:
        text = text[:CONTEXT_CHAR_LIMIT]
        note = f" (truncated to the first {CONTEXT_CHAR_LIMIT} characters)"
    return (
        f"Supplied context {context.label} ({context.kind}; describes intent, not sound){note}:\n"
        f"```\n{text}\n```"
    )


def _task_text(
    task: str,
    clips: Sequence[ClipView],
    question: str | None,
    brief: BriefView | None,
    measurements: Mapping[str, object] | None,
) -> str:
    if task == "hear":
        return (
            "Listen to clip A. Describe what is perceptible: overall character; the voices or streams and "
            "their roles; timbre, articulation and resonance; texture; pulse or its absence; tuning; "
            "dynamics; phrases, sections and form, including repetitions, returns and transformations. "
            "Write a 2-5 sentence summary and time-located claims with relations."
        )
    if task == "ask":
        return (
            f"Answer this question about clip A: {question}\n"
            "Put a direct answer in `answer`. If the question offers lettered options, begin the answer "
            "with the single option letter. If the audio does not let you decide, say so. Then give the "
            "claims that support the answer."
        )
    if task == "compare":
        extra = f"\nFocus question: {question}" if question else ""
        return (
            "Clip A is the earlier or reference version; clip B is the later version. Identify every "
            "perceptible difference between them (what changed, in which voice, where, and how it sounds) "
            "and the important musical properties and relationships that are preserved. Summarize the "
            f"overall effect of the change in 2-4 sentences.{extra}"
        )
    if task == "investigate":
        focus = question or (
            "Examine this passage closely: its sound events, attacks and decays, resonances, articulation, "
            "voices and their relationships."
        )
        data = json.dumps(measurements or {}, ensure_ascii=False, sort_keys=True)
        return (
            f"{focus}\nPut your conclusion in `answer`.\n"
            "Deterministic measurements of this clip (computed from the signal, not heard; times are "
            f"relative to the clip):\n```json\n{data}\n```\n"
            "Use the measurements only to direct your attention. Report what you hear, and say so when a "
            "measurement and your hearing disagree."
        )
    if task == "review":
        if brief is None:
            raise ValueError("review needs a brief")
        before = (
            " Clip B is the previous render; use it to judge what the revision changed."
            if len(clips) > 1
            else ""
        )
        goals = "\n".join(f"- {gid}: {text}" for gid, text in brief.goals) or "- (none)"
        protect = "\n".join(f"- {pid}: {text}" for pid, text in brief.protect) or "- (none)"
        return (
            f"Clip A is the current render of a composition.{before} Assess clip A against the human "
            "artistic goals and the protected musical relationships below. Give one assessment for every "
            "goal and protected relationship, with verdict met, partly, not-met or unclear, citing the "
            "indices (0-based) of the claims that support it. Then add the most useful time-specific "
            "revision observations as claims: what to change, where, described in sound.\n"
            f"Goals:\n{goals}\nProtected relationships:\n{protect}"
        )
    raise ValueError(f"unknown listening task: {task}")


def _schema(task: str, labels: Sequence[str], targets: Sequence[str]) -> dict[str, object]:
    locus = {
        "type": "object",
        "properties": {
            "clip": {"type": "string", "enum": list(labels)},
            "start": {"type": "number"},
            "end": {"type": "number"},
            "voice": {"type": "string"},
        },
        "required": ["clip", "start", "end"],
    }
    claim = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "aspect": {"type": "string", "enum": list(ASPECTS)},
            "scale": {"type": "string", "enum": list(SCALES)},
            "loci": {"type": "array", "items": locus},
            "relation": {"type": "string", "enum": [*RELATIONS, "none"]},
            "basis": {"type": "string", "enum": list(BASES)},
            "confidence": {"type": "string", "enum": list(CONFIDENCES)},
        },
        "required": ["text", "aspect", "scale", "loci", "relation", "basis", "confidence"],
    }
    properties: dict[str, object] = {
        "summary": {"type": "string"},
        "claims": {"type": "array", "items": claim},
    }
    required = ["summary", "claims"]
    if task in ("ask", "investigate"):
        properties["answer"] = {"type": "string"}
        required.append("answer")
    if task == "review":
        properties["assessments"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "enum": list(targets)},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "claims": {"type": "array", "items": {"type": "integer"}},
                    "note": {"type": "string"},
                },
                "required": ["target", "verdict", "claims", "note"],
            },
        }
        required.append("assessments")
    return {"type": "object", "properties": properties, "required": required}


def build(
    task: str,
    clips: Sequence[ClipView],
    *,
    question: str | None = None,
    contexts: Sequence[ContextView] = (),
    brief: BriefView | None = None,
    measurements: Mapping[str, object] | None = None,
) -> ListeningPrompt:
    parts = [
        "Clips:",
        *(_clip_line(clip) for clip in clips),
        "",
        _task_text(task, clips, question, brief, measurements),
    ]
    parts.extend(f"\n{_context_block(context)}" for context in contexts)
    targets = [gid for gid, _ in brief.goals] + [pid for pid, _ in brief.protect] if brief else []
    return ListeningPrompt(
        system=INSTRUCTIONS,
        prompt="\n".join(parts),
        schema=_schema(task, [clip.label for clip in clips], targets),
    )
