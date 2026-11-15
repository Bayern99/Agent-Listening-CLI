from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from agent_listening import evidence

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "src" / "agent_listening" / "evidence-1.schema.json"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _collect_enums(node, out):
    if isinstance(node, dict):
        if "enum" in node:
            out.append(tuple(node["enum"]))
        for v in node.values():
            _collect_enums(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_enums(v, out)


def test_schema_enums_match_constants():
    enums = []
    _collect_enums(_schema(), enums)
    expected = {
        tuple(evidence.TASKS),
        tuple(evidence.ROLES),
        tuple(evidence.ASPECTS),
        tuple(evidence.SCALES),
        (*tuple(evidence.RELATIONS), None),
        tuple(evidence.BASES),
        tuple(evidence.CONFIDENCES),
        tuple(evidence.ORIGINS),
        tuple(evidence.STATUSES),
        tuple(evidence.VERDICTS),
        tuple(evidence.CHECK_RESULTS),
        tuple(evidence.THINKING_LEVELS),
        ("full", "excerpt"),
        ("complete", "partial", "failed"),
    }
    for enum in enums:
        assert enum in expected, f"schema enum not matching constants: {enum}"
    present = {e for e in enums}
    for const in (evidence.ASPECTS, evidence.VERDICTS, evidence.STATUSES, evidence.THINKING_LEVELS):
        assert tuple(const) in present, f"constant {const} missing from schema"


def _sample_evidence() -> evidence.Evidence:
    clips = {"A": evidence.ClipInfo("A", "A", 0.0, 5.0, 5.0)}
    claims, _ = evidence.claims_from_model(
        {
            "summary": "s",
            "claims": [
                {
                    "text": "bright attack",
                    "aspect": "timbre",
                    "scale": "event",
                    "loci": [{"clip": "A", "start": 0.5, "end": 1.0}],
                    "relation": "none",
                    "basis": "audio",
                    "confidence": "high",
                }
            ],
        },
        clips,
    )
    return evidence.Evidence(
        schema=evidence.SCHEMA,
        version=evidence.VERSION,
        task="hear",
        status="complete",
        tool=evidence.Tool("agent-listening", "1.0.0"),
        created_at="2026-11-15T00:00:00+00:00",
        inputs=(
            evidence.Input(
                label="A",
                role="subject",
                path="/x/a.wav",
                sha256="a" * 64,
                pcm_sha256="b" * 64,
                format=evidence.AudioFormat("WAV", "PCM_16", 48000, 2, 240000, 5.0),
            ),
        ),
        context=(),
        brief=evidence.Brief(
            path="/x/brief.json",
            sha256="c" * 64,
            goals=(evidence.Goal("g1", "keep the bell clear"),),
            protect=(evidence.Protection("p1", "bass stays", (evidence.BriefLocus(1.0, 2.0),)),),
        ),
        request=evidence.Request(
            question="q", focus=evidence.Focus(0.0, 5.0), measures=("level",), counterbalance=False
        ),
        transport=(evidence.Transport("A", "/cache/x.flac", "d" * 64, "excerpt", 0.0, 5.0, 48000, 2),),
        measurements=(
            evidence.Measurement("m1", "level", ("A",), "agent-listening.audio/1", {}, {"rms_dbfs": -20.0}),
        ),
        claims=claims,
        assessments=(evidence.Assessment("g1", "met", ("c0",), "n"),),
        answer="yes",
        summary="s",
        producer=evidence.Producer(
            "gemini-api",
            "0.4.5",
            "gemini-4-hydrogen",
            "high",
            "e" * 64,
            ("ix",),
            {"input_tokens": 1, "audio_tokens": 1, "output_tokens": 1, "thought_tokens": 0},
            1.0,
            1,
        ),
        limitations=("MODEL_TIME_APPROXIMATE",),
        error=None,
    )


def test_produced_evidence_validates():
    jsonschema.validate(evidence.to_json(_sample_evidence()), _schema())


def test_full_population_validates():
    ev = _sample_evidence()
    blob = evidence.to_json(ev)
    jsonschema.validate(blob, _schema())
    round_tripped = evidence.to_json(evidence.from_json(blob))
    jsonschema.validate(round_tripped, _schema())


def test_schema_rejects_extra_keys():
    blob = evidence.to_json(_sample_evidence())
    blob["bogus"] = True
    import pytest

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(blob, _schema())
