from __future__ import annotations

import json
from itertools import product

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agent_listening import evidence
from agent_listening.evidence import (
    CHECK_RESULTS,
    Check,
    ClipInfo,
    ContractError,
    Locus,
)


def _clip(input_label: str = "A", offset: float = 0.0, duration: float = 10.0) -> ClipInfo:
    return ClipInfo(
        label=input_label, input=input_label, offset_s=offset, duration_s=duration, input_duration_s=60.0
    )


def test_derive_status_exhaustive():
    check_names = ["locus-in-range", "signal-difference", "pcm-identity", "order-consistency", "other"]
    deterministic = set(evidence.DETERMINISTIC_CHECKS)
    for n_loci, names, results in product(
        (0, 1), product(check_names, check_names), product(CHECK_RESULTS, repeat=2)
    ):
        loci = (Locus("A", 0.0, 1.0),) * n_loci
        checks = tuple(Check(name=n, result=r, detail="t") for n, r in zip(names, results, strict=True))
        status = evidence.derive_status(loci, checks)
        has_fail = any(n in deterministic and r == "fail" for n, r in zip(names, results, strict=True))
        has_bad_locus = n_loci == 0 or any(
            n == "locus-in-range" and r == "fail" for n, r in zip(names, results, strict=True)
        )
        has_pass = any(n in deterministic and r == "pass" for n, r in zip(names, results, strict=True))
        if has_fail:
            assert status == "contradicted", (n_loci, names, results)
        elif has_bad_locus:
            assert status == "ungrounded", (n_loci, names, results)
        elif has_pass:
            assert status == "consistent", (n_loci, names, results)
        else:
            assert status == "unchecked", (n_loci, names, results)


def test_order_consistency_is_informational_only():
    """Model self-agreement checks must not corroborate or contradict."""
    loci = (Locus("A", 0.0, 1.0),)
    for result in ("pass", "fail", "n/a"):
        checks = (Check(name="order-consistency", result=result, detail="t"),)
        assert evidence.derive_status(loci, checks) == "unchecked"


def test_make_claim_derives_and_enforces():
    claim = evidence.make_claim(
        id="c0",
        origin="model",
        text="t",
        aspect="timbre",
        scale="event",
        loci=[Locus("A", 0.0, 1.0)],
        relation=None,
        basis="audio",
        confidence="low",
    )
    assert claim.status == "unchecked"
    with pytest.raises(ContractError):
        evidence.Claim(
            id="c1",
            origin="model",
            text="t",
            aspect="timbre",
            scale="event",
            loci=(Locus("A", 0.0, 1.0),),
            relation=None,
            basis="audio",
            confidence="low",
            checks=(),
            status="consistent",
        )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("12.5-20", (12.5, 20.0)),
        ("1:02-1:30.5", (62.0, 90.5)),
        ("0:00:05-0:00:09", (5.0, 9.0)),
        ("0-0.5", (0.0, 0.5)),
    ],
)
def test_parse_span(text, expected):
    assert evidence.parse_span(text) == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["", "abc", "5-3", "1:99x-2", "-1-5", "1-"])
def test_parse_span_rejects(bad):
    with pytest.raises(ValueError):
        evidence.parse_span(bad)


def _payload(claims):
    return {"summary": "s", "claims": claims}


def _good_claim(**over):
    base = {
        "text": "a bell",
        "aspect": "timbre",
        "scale": "event",
        "loci": [{"clip": "A", "start": 1.0, "end": 2.0}],
        "relation": "none",
        "basis": "audio",
        "confidence": "medium",
    }
    base.update(over)
    return base


def test_claims_from_model_offset_mapping():
    clip = _clip(offset=10.0, duration=5.0)
    claims, problems = evidence.claims_from_model(
        _payload([_good_claim(loci=[{"clip": "A", "start": 1.0, "end": 2.0}])]),
        {"A": clip},
    )
    assert not problems
    assert len(claims) == 1
    loc = claims[0].loci[0]
    assert (loc.input, loc.start, loc.end) == ("A", 11.0, 12.0)
    assert claims[0].checks[0].result == "pass"


@given(
    st.floats(min_value=0, max_value=60),
    st.floats(min_value=0.1, max_value=30),
    st.floats(min_value=0, max_value=10),
)
def test_claims_from_model_offset_property(offset, duration, clip_start):
    clip = _clip(offset=offset, duration=duration)
    claims, _ = evidence.claims_from_model(
        _payload([_good_claim(loci=[{"clip": "A", "start": clip_start, "end": clip_start + 0.5}])]),
        {"A": clip},
    )
    assert len(claims) == 1
    assert claims[0].loci[0].start == pytest.approx(offset + clip_start)


def test_claims_enum_repair():
    claims, problems = evidence.claims_from_model(
        _payload(
            [_good_claim(aspect="loudness", scale="bar", basis="ears", confidence="sure", relation="vibes")]
        ),
        {"A": _clip()},
    )
    c = claims[0]
    assert c.aspect == "other" and c.scale == "event" and c.basis == "audio" and c.confidence == "low"
    assert c.relation is None
    assert len(problems) == 5


def test_claims_relation_none():
    claims, _ = evidence.claims_from_model(_payload([_good_claim(relation="none")]), {"A": _clip()})
    assert claims[0].relation is None
    claims2, problems2 = evidence.claims_from_model(_payload([_good_claim(relation="bogus")]), {"A": _clip()})
    assert claims2[0].relation is None and problems2


def test_claims_out_of_range_and_no_locus():
    claims, _ = evidence.claims_from_model(
        _payload([_good_claim(loci=[{"clip": "A", "start": 9.0, "end": 11.0}])]),
        {"A": _clip(duration=10.0)},
    )
    assert claims[0].checks[0].result == "fail"
    assert claims[0].status == "ungrounded"
    claims2, _ = evidence.claims_from_model(_payload([_good_claim(loci=[])]), {"A": _clip()})
    assert claims2[0].loci == ()
    assert claims2[0].status == "ungrounded"


def test_work_scale_implicit_locus():
    claims, _ = evidence.claims_from_model(
        _payload([_good_claim(scale="work", loci=[])]),
        {"A": _clip(duration=10.0)},
    )
    assert len(claims[0].loci) == 1
    assert claims[0].loci[0].end == 10.0
    assert claims[0].checks[0].detail == "implicit whole-clip locus"


def test_claims_unknown_clip_and_non_dict():
    claims, problems = evidence.claims_from_model(
        {"claims": [_good_claim(loci=[{"clip": "Z", "start": 0, "end": 1}]), "junk", _good_claim(text="")]},
        {"A": _clip()},
    )
    assert len(claims) == 1
    assert claims[0].status == "ungrounded"
    assert len(problems) == 3


def test_apply_identity_check():
    differs = evidence.make_claim(
        id="c0",
        origin="model",
        text="t",
        aspect="timbre",
        scale="event",
        loci=[Locus("A", 0, 1)],
        relation="differs",
        basis="audio",
        confidence="low",
    )
    same = evidence.make_claim(
        id="c1",
        origin="model",
        text="t",
        aspect="timbre",
        scale="event",
        loci=[Locus("A", 0, 1)],
        relation="unchanged",
        basis="audio",
        confidence="low",
    )
    out = evidence.apply_identity_check([differs, same], identical=True, measurement_id="m1")
    assert out[0].status == "contradicted"
    assert out[1].status == "consistent"
    out2 = evidence.apply_identity_check([differs, same], identical=False, measurement_id="m1")
    assert out2 == (differs, same)


def test_apply_difference_check():
    differs = evidence.make_claim(
        id="c0",
        origin="model",
        text="t",
        aspect="timbre",
        scale="event",
        loci=[Locus("A", 5.0, 6.0)],
        relation="differs",
        basis="audio",
        confidence="low",
    )
    same = evidence.make_claim(
        id="c1",
        origin="model",
        text="t",
        aspect="timbre",
        scale="event",
        loci=[Locus("A", 0.0, 2.0)],
        relation="unchanged",
        basis="audio",
        confidence="low",
    )
    regions = {"A": [(5.5, 6.5)], "B": [(5.5, 6.5)]}
    out = evidence.apply_difference_check(
        [differs, same], regions=regions, global_change=False, measurement_id="m2"
    )
    assert out[0].status == "consistent"
    assert out[1].status == "consistent"

    out2 = evidence.apply_difference_check(
        [differs], regions={"A": [(20.0, 21.0)]}, global_change=False, measurement_id="m2"
    )
    assert out2[0].status == "contradicted"

    out3 = evidence.apply_difference_check(
        [same], regions={"A": [(1.0, 2.5)]}, global_change=False, measurement_id="m2"
    )
    assert out3[0].status == "contradicted"

    out4 = evidence.apply_difference_check(
        [differs, same], regions=regions, global_change=True, measurement_id="m2"
    )
    assert all(c.checks[-1].result == "n/a" for c in out4)


def test_apply_order_check():
    c1 = evidence.make_claim(
        id="c0",
        origin="model",
        text="t",
        aspect="timbre",
        scale="event",
        loci=[Locus("A", 5.0, 6.0)],
        relation="differs",
        basis="audio",
        confidence="low",
    )
    c2 = evidence.make_claim(
        id="k0",
        origin="model",
        text="t",
        aspect="timbre",
        scale="event",
        loci=[Locus("A", 5.5, 6.5)],
        relation="differs",
        basis="audio",
        confidence="low",
    )
    first, second = evidence.apply_order_check([c1], [c2])
    assert first[0].checks[-1].name == "order-consistency"
    assert first[0].checks[-1].result == "pass"
    assert second[0].checks[-1].result == "pass"
    first2, _ = evidence.apply_order_check([c1], [])
    assert first2[0].checks[-1].result == "fail"


def test_evidence_roundtrip():
    clips = {"A": _clip()}
    claims, _ = evidence.claims_from_model(
        _payload([_good_claim(loci=[{"clip": "A", "start": 1, "end": 2, "voice": "low drone"}])]), clips
    )
    ev = evidence.Evidence(
        schema=evidence.SCHEMA,
        version=evidence.VERSION,
        task="hear",
        status="complete",
        tool=evidence.Tool("agent-listening", "1.0.0"),
        created_at="2026-11-15T00:00:00+00:00",
        inputs=(),
        context=(),
        brief=None,
        request=evidence.Request(
            question=None, focus=evidence.Focus(1.0, 2.0), measures=(), counterbalance=False
        ),
        transport=(),
        measurements=(),
        claims=claims,
        assessments=(),
        answer=None,
        summary="s",
        producer=None,
        limitations=(),
        error=None,
    )
    blob = evidence.to_json(ev)
    assert evidence.to_json(evidence.from_json(blob)) == blob


def test_from_json_rejects_bad():
    with pytest.raises(ContractError):
        evidence.from_json({"schema": "other", "version": 1})
    with pytest.raises(ContractError):
        evidence.from_json({"schema": evidence.SCHEMA, "version": 2})
    with pytest.raises(ContractError):
        evidence.from_json("nope")


# --- mutation-killing coverage: constants, validation boundaries, messages, codec ---

SHA = "a" * 64
FMT = evidence.AudioFormat("WAV", "PCM_16", 48000, 2, 96000, 2.0)


def test_constants_exact():
    assert evidence.SCHEMA == "agent-listening/evidence"
    assert evidence.VERSION == 1
    assert evidence.TASKS == ("hear", "ask", "compare", "investigate", "review", "identify")
    assert evidence.ROLES == ("subject", "before", "after")
    assert evidence.ASPECTS == (
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
    assert evidence.SCALES == ("event", "phrase", "section", "work")
    assert evidence.RELATIONS == (
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
    assert evidence.BASES == ("audio", "context", "audio+context")
    assert evidence.CONFIDENCES == ("low", "medium", "high")
    assert evidence.ORIGINS == ("model", "measurement", "human")
    assert evidence.STATUSES == ("unchecked", "consistent", "ungrounded", "contradicted")
    assert evidence.VERDICTS == ("met", "partly", "not-met", "unclear")
    assert evidence.CHECK_RESULTS == ("pass", "fail", "n/a")
    assert evidence.EVIDENCE_STATUSES == ("complete", "partial", "failed")
    assert evidence.THINKING_LEVELS == ("minimal", "low", "medium", "high")
    assert evidence.DIFFERENCE_TOLERANCE_S == 2.0
    assert evidence.LOCUS_TOLERANCE_S == 0.5
    assert evidence.IDENTITY_CHECK == "pcm-identity"
    assert evidence.DIFFERENCE_CHECK == "signal-difference"
    assert evidence.LOCUS_CHECK == "locus-in-range"
    assert evidence.ORDER_CHECK == "order-consistency"
    assert evidence.DETERMINISTIC_CHECKS == ("pcm-identity", "signal-difference")


@pytest.mark.parametrize(
    "cls",
    [
        evidence.AudioFormat,
        evidence.Input,
        evidence.ContextFile,
        evidence.BriefLocus,
        evidence.Goal,
        evidence.Protection,
        evidence.Brief,
        evidence.Focus,
        evidence.Request,
        evidence.Locus,
        evidence.Check,
        evidence.Claim,
        evidence.Transport,
        evidence.Measurement,
        evidence.Assessment,
        evidence.Producer,
        evidence.Tool,
        evidence.ErrorBody,
        evidence.Evidence,
        evidence.ClipInfo,
    ],
)
def test_dataclasses_frozen(cls):
    assert cls.__dataclass_params__.frozen


def _evidence_with(**over):
    base = {
        "schema": evidence.SCHEMA,
        "version": evidence.VERSION,
        "task": "hear",
        "status": "complete",
        "tool": evidence.Tool("agent-listening", "1.0.0"),
        "created_at": "2026-01-01T00:00:00+00:00",
        "inputs": (),
        "context": (),
        "brief": None,
        "request": evidence.Request(None, None, (), False),
        "transport": (),
        "measurements": (),
        "claims": (),
        "assessments": (),
        "answer": None,
        "summary": "s",
        "producer": None,
        "limitations": (),
        "error": None,
    }
    base.update(over)
    return evidence.Evidence(**base)


@pytest.mark.parametrize(
    "build",
    [
        lambda: evidence.AudioFormat("WAV", "PCM_16", 0, 2, 96000, 2.0),
        lambda: evidence.AudioFormat("WAV", "PCM_16", -1, 2, 96000, 2.0),
        lambda: evidence.AudioFormat("WAV", "PCM_16", 48000, 0, 96000, 2.0),
        lambda: evidence.AudioFormat("WAV", "PCM_16", 48000, -1, 96000, 2.0),
        lambda: evidence.AudioFormat("WAV", "PCM_16", 48000, 2, -1, 2.0),
        lambda: evidence.Input("", "subject", "/x", SHA, SHA, FMT),
        lambda: evidence.Input("A", "bogus", "/x", SHA, SHA, FMT),
        lambda: evidence.Input("A", "subject", "/x", "short", SHA, FMT),
        lambda: evidence.Input("A", "subject", "/x", SHA.upper(), SHA, FMT),
        lambda: evidence.Input("A", "subject", "/x", SHA, "z" * 64, FMT),
        lambda: evidence.ContextFile("S1", "notes", "/x", SHA, -1),
        lambda: evidence.ContextFile("", "notes", "/x", SHA, 0),
        lambda: evidence.ContextFile("S1", "notes", "/x", "bad", 0),
        lambda: evidence.BriefLocus(2.0, 1.0),
        lambda: evidence.BriefLocus(-1.0, 0.5),
        lambda: evidence.Goal("", "text"),
        lambda: evidence.Protection("", "t"),
        lambda: evidence.Protection("p1", "t", []),
        lambda: evidence.Brief("/b", SHA, [], ()),
        lambda: evidence.Brief("/b", SHA, (), []),
        lambda: evidence.Brief("/b", "bad", (), ()),
        lambda: evidence.Focus(1.0, 1.0),
        lambda: evidence.Focus(-0.5, 1.0),
        lambda: evidence.Request(None, None, [], False),
        lambda: evidence.Locus("", 0.0, 1.0),
        lambda: evidence.Locus("A", 2.0, 1.0),
        lambda: evidence.Locus("A", -1.0, 0.5),
        lambda: evidence.Check("", "pass", "d"),
        lambda: evidence.Check("x", "bogus", "d"),
        lambda: evidence.Transport("", "/p", SHA, "full", 0.0, 1.0, 48000, 2),
        lambda: evidence.Transport("A", "/p", "bad", "full", 0.0, 1.0, 48000, 2),
        lambda: evidence.Transport("A", "/p", SHA, "weird", 0.0, 1.0, 48000, 2),
        lambda: evidence.Transport("A", "/p", SHA, "full", -0.1, 1.0, 48000, 2),
        lambda: evidence.Transport("A", "/p", SHA, "full", 0.0, 0.0, 48000, 2),
        lambda: evidence.Transport("A", "/p", SHA, "full", 0.0, -1.0, 48000, 2),
        lambda: evidence.Transport("A", "/p", SHA, "full", 0.0, 1.0, 0, 2),
        lambda: evidence.Transport("A", "/p", SHA, "full", 0.0, 1.0, 48000, 0),
        lambda: evidence.Measurement("", "k", (), "m", {}, {}),
        lambda: evidence.Measurement("m", "", (), "m", {}, {}),
        lambda: evidence.Measurement("m", "k", [], "m", {}, {}),
        lambda: evidence.Assessment("", "met", (), "n"),
        lambda: evidence.Assessment("t", "bogus", (), "n"),
        lambda: evidence.Assessment("t", "met", [], "n"),
        lambda: evidence.Producer("p", "v", "m", "bogus", SHA, (), {}, 1.0),
        lambda: evidence.Producer("p", "v", "m", "high", "bad", (), {}, 1.0),
        lambda: evidence.Producer("p", "v", "m", "high", SHA, [], {}, 1.0),
        lambda: _evidence_with(schema="other"),
        lambda: _evidence_with(version=2),
        lambda: _evidence_with(task="bogus"),
        lambda: _evidence_with(status="bogus"),
        lambda: _evidence_with(inputs=[]),
        lambda: _evidence_with(limitations=["x"]),
    ],
)
def test_invalid_construction_rejected(build):
    with pytest.raises(ContractError):
        build()


def test_boundary_valid_construction():
    evidence.AudioFormat("WAV", "PCM_16", 48000, 2, 0, 0.0)
    evidence.BriefLocus(0.0, 0.0)
    evidence.Locus("A", 0.0, 0.0)
    evidence.ContextFile("S1", "notes", "/x", SHA, 0)
    evidence.Transport("A", "/p", SHA, "excerpt", 0.0, 0.001, 48000, 2)
    evidence.Focus(0.0, 0.001)


@pytest.mark.parametrize(
    "text,expected", [("1:00:00-1:00:01", (3600.0, 3601.0)), ("2:30:15-2:30:16.25", (9015.0, 9016.25))]
)
def test_parse_span_hours(text, expected):
    assert evidence.parse_span(text) == pytest.approx(expected)


def test_claims_problem_messages():
    claims, problems = evidence.claims_from_model(
        _payload(
            [
                "not-a-dict",
                {**_good_claim(), "text": " "},
                {
                    **_good_claim(),
                    "aspect": "weird",
                    "scale": "weird",
                    "basis": "weird",
                    "confidence": "weird",
                    "relation": "weird",
                },
                {**_good_claim(), "loci": "nope"},
                {
                    **_good_claim(),
                    "loci": [42, {"clip": "Z", "start": 0, "end": 1}, {"clip": "A", "start": "x", "end": 1}],
                },
            ]
        ),
        {"A": _clip()},
        id_prefix="k",
    )
    assert problems == (
        "k0: claim is not an object",
        "k1: missing text",
        "k2: unknown aspect 'weird' mapped to 'other'",
        "k2: unknown scale 'weird' mapped to 'event'",
        "k2: unknown basis 'weird' mapped to 'audio'",
        "k2: unknown confidence 'weird' mapped to 'low'",
        "k2: unknown relation 'weird' dropped",
        "k3: loci missing or not a list",
        "k4: locus 0 is not an object",
        "k4: locus 1 names unknown clip 'Z'",
        "k4: locus 2 has non-numeric times",
    )
    assert [c.id for c in claims] == ["k2", "k3", "k4"]
    assert claims[1].checks[0].detail == "no locus"


def test_claims_locus_filtering_and_offset_clamp():
    clip = _clip(offset=10.0, duration=60.0)
    claims, _ = evidence.claims_from_model(
        _payload(
            [
                _good_claim(
                    loci=[
                        {"clip": "A", "start": 5.0, "end": 3.0},
                        {"clip": "A", "start": -20.0, "end": -5.0},
                    ]
                ),
                _good_claim(loci=[{"clip": "A", "start": -15.0, "end": 5.0, "voice": 123}]),
            ]
        ),
        {"A": clip},
    )
    assert claims[0].loci == ()
    loc = claims[1].loci[0]
    assert loc.start == 0.0 and loc.end == 15.0 and loc.voice is None


def test_claims_work_scale_no_clip():
    claims, problems = evidence.claims_from_model(_payload([_good_claim(scale="work", loci=[])]), {})
    assert claims == ()
    assert problems == ("c0: work-scale claim has no clip to ground",)


def test_apply_identity_check_details():
    differs = evidence.make_claim(
        id="c1",
        origin="model",
        text="t",
        aspect="timbre",
        scale="work",
        loci=[Locus("A", 0.0, 1.0)],
        relation="differs",
        basis="audio",
        confidence="low",
    )
    out = evidence.apply_identity_check([differs], identical=False, measurement_id="m1")
    assert out[0].checks == ()
    out = evidence.apply_identity_check([differs], identical=True, measurement_id="m1")
    chk = out[0].checks[-1]
    assert (chk.name, chk.result, chk.detail, chk.measurement) == (
        "pcm-identity",
        "fail",
        "inputs are PCM-identical",
        "m1",
    )


def test_apply_difference_check_boundaries():
    def claim(rel, start=1.0, end=2.0):
        return evidence.make_claim(
            id="c1",
            origin="model",
            text="t",
            aspect="timbre",
            scale="event",
            loci=[Locus("A", start, end)],
            relation=rel,
            basis="audio",
            confidence="low",
        )

    regions = {"A": [(3.5, 4.5)]}
    out = evidence.apply_difference_check(
        [claim("differs")], regions=regions, global_change=False, measurement_id="m1", tolerance_s=2.0
    )
    chk = out[0].checks[-1]
    assert (
        chk.result == "pass" and chk.detail == "locus overlaps a measured difference region (tolerance 2.0s)"
    )
    out = evidence.apply_difference_check(
        [claim("differs")],
        regions={"A": [(9.0, 10.0)]},
        global_change=False,
        measurement_id="m1",
        tolerance_s=0.5,
    )
    chk = out[0].checks[-1]
    assert chk.result == "fail" and chk.detail == "no measured difference region within 0.5s of any locus"
    # unchanged: overlap of exactly 0.5s is not a breach; >0.5 is
    out = evidence.apply_difference_check(
        [claim("unchanged", 1.0, 2.0)], regions={"A": [(1.5, 2.5)]}, global_change=False, measurement_id="m1"
    )
    assert out[0].checks[-1].result == "pass"
    out = evidence.apply_difference_check(
        [claim("unchanged", 1.0, 2.0)], regions={"A": [(1.4, 2.5)]}, global_change=False, measurement_id="m1"
    )
    chk = out[0].checks[-1]
    assert chk.result == "fail" and chk.detail == "locus overlaps a measured difference region by >0.5s"
    out = evidence.apply_difference_check(
        [claim("differs")], regions=regions, global_change=True, measurement_id="m1"
    )
    chk = out[0].checks[-1]
    assert chk.result == "n/a" and chk.detail == "global change: region check not applicable"


def test_apply_order_check_mismatch_and_details():
    def claim(cid, aspect, inp="A", start=1.0, end=2.0):
        return evidence.make_claim(
            id=cid,
            origin="model",
            text="t",
            aspect=aspect,
            scale="event",
            loci=[Locus(inp, start, end)],
            relation="differs",
            basis="audio",
            confidence="low",
        )

    first, second = evidence.apply_order_check([claim("c0", "pitch")], [claim("k0", "timbre")])
    chk = first[0].checks[-1]
    assert chk.name == "order-consistency" and chk.result == "fail"
    assert chk.detail == "no matching differs-claim in counterbalanced run"
    first, second = evidence.apply_order_check([claim("c0", "pitch")], [claim("k0", "pitch")])
    assert first[0].checks[-1].result == "pass"
    assert first[0].checks[-1].detail == "matching differs-claim in counterbalanced run"
    assert second[0].checks[-1].result == "pass"
    # same aspect but locus on a different input does not match
    first, _ = evidence.apply_order_check([claim("c0", "pitch", "A")], [claim("k0", "pitch", "B")])
    assert first[0].checks[-1].result == "fail"


def test_to_json_float_rounding_and_voice_omission():
    claims, _ = evidence.claims_from_model(
        _payload([_good_claim(loci=[{"clip": "A", "start": 1.23456, "end": 2.0}])]), {"A": _clip()}
    )
    blob = evidence.to_json(_evidence_with(claims=claims))
    assert blob["claims"][0]["loci"][0]["start"] == 1.235
    assert "voice" not in blob["claims"][0]["loci"][0]
    claims2, _ = evidence.claims_from_model(
        _payload([_good_claim(loci=[{"clip": "A", "start": 1, "end": 2, "voice": "v"}])]), {"A": _clip()}
    )
    blob2 = evidence.to_json(_evidence_with(claims=claims2))
    assert blob2["claims"][0]["loci"][0]["voice"] == "v"


def test_from_json_strict_decode():
    blob = evidence.to_json(_evidence_with())
    bad = dict(blob, bogus_key=1)
    with pytest.raises(ContractError):
        evidence.from_json(bad)
    bad = json.loads(json.dumps(blob))
    del bad["summary"]
    with pytest.raises(ContractError):
        evidence.from_json(bad)
    bad = json.loads(json.dumps(blob))
    bad["summary"] = 5
    with pytest.raises(ContractError):
        evidence.from_json(bad)
    bad = json.loads(json.dumps(blob))
    bad["request"]["counterbalance"] = "yes"
    with pytest.raises(ContractError):
        evidence.from_json(bad)
    # omitted optional fields fall back to dataclass defaults
    claims, _ = evidence.claims_from_model(_payload([_good_claim()]), {"A": _clip()})
    blob = evidence.to_json(_evidence_with(claims=claims))
    blob["claims"][0]["loci"][0].pop("voice", None)
    ev = evidence.from_json(blob)
    assert ev.claims[0].loci[0].voice is None


def test_dumps_unicode_and_indent():
    text = evidence.dumps(_evidence_with(summary="café"))
    assert "café" in text and "\n  " in text


def test_boundary_valid_extra():
    evidence.AudioFormat("WAV", "PCM_16", 1, 1, 0, 0.0)
    evidence.Transport("A", "/p", SHA, "full", 0.0, 1.0, 1, 1)


def test_claim_field_tuples_enforced():
    with pytest.raises(ContractError):
        evidence.Claim(
            id="c1",
            origin="model",
            text="t",
            aspect="timbre",
            scale="event",
            loci=[],
            relation=None,
            basis="audio",
            confidence="low",
            checks=(),
            status="ungrounded",
        )
    with pytest.raises(ContractError):
        evidence.Claim(
            id="c1",
            origin="model",
            text="t",
            aspect="timbre",
            scale="event",
            loci=(),
            relation=None,
            basis="audio",
            confidence="low",
            checks=[],
            status="ungrounded",
        )


@pytest.mark.parametrize("bad", ["5-5", 5])
def test_parse_span_rejects_boundary(bad):
    with pytest.raises((ValueError, TypeError)):
        evidence.parse_span(bad)


def test_locus_in_range_tolerance_and_zero_start():
    clip = _clip(duration=10.0)
    # end beyond clip duration but inside LOCUS_TOLERANCE_S (0.5s) still passes
    claims, _ = evidence.claims_from_model(
        _payload([_good_claim(loci=[{"clip": "A", "start": 0.0, "end": 10.3}])]), {"A": clip}
    )
    assert claims[0].checks[0].result == "pass"
    # zero-length locus at clip start is in range
    claims, _ = evidence.claims_from_model(
        _payload([_good_claim(loci=[{"clip": "A", "start": 0.0, "end": 0.0}])]), {"A": clip}
    )
    assert claims[0].checks[0].result == "pass"
    # just outside the tolerance fails
    claims, _ = evidence.claims_from_model(
        _payload([_good_claim(loci=[{"clip": "A", "start": 0.0, "end": 10.6}])]), {"A": clip}
    )
    assert claims[0].checks[0].result == "fail"
