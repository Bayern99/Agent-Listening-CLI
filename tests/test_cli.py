from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import FAKE_GEMINI, read_log

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SR = 48000


def _run(args, env, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "agent_listening", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd or SRC,
    )


def _env(tmp_path, **extra):
    log = tmp_path / "fake-gemini.log"
    env = dict(os.environ)
    env["FAKE_GEMINI_LOG"] = str(log)
    env["AGENT_LISTENING_GEMINI_API"] = str(FAKE_GEMINI)
    env["AGENT_LISTENING_CACHE"] = str(tmp_path / "cache")
    env.pop("FAKE_GEMINI_EXIT", None)
    env.pop("FAKE_GEMINI_REPLY", None)
    env.pop("FAKE_GEMINI_NO_CREDENTIAL", None)
    env.update({k: str(v) for k, v in extra.items()})
    return env, log


def _sine(path, freq=440.0, seconds=1.0, amp=0.2, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    data = np.repeat((amp * np.sin(2 * np.pi * freq * t))[:, None], 2, axis=1)
    sf.write(str(path), data, sr, subtype="PCM_16", format="WAV")
    return path


def _reply(tmp_path, payload):
    p = tmp_path / "reply.json"
    p.write_text(json.dumps(payload))
    return p


def test_version():
    env, _ = _env(Path.cwd())
    proc = _run(["--version"], env)
    assert proc.returncode == 0
    assert proc.stdout.strip() == "agent-listening 1.0.0"


def test_hear_emits_valid_evidence(tmp_path):
    wav = _sine(tmp_path / "a.wav")
    env, log = _env(tmp_path)
    proc = _run(["hear", str(wav)], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    assert ev["schema"] == "agent-listening/evidence"
    assert ev["task"] == "hear"
    assert ev["status"] == "complete"
    assert ev["inputs"][0]["sha256"]
    assert ev["transport"][0]["operation"] == "full"
    assert ev["producer"]["model"] == "gemini-4-hydrogen"
    assert "MODEL_INPUT_REDUCED_RESOLUTION" in ev["limitations"]
    gen = [e for e in read_log(log) if e["argv"][:1] == ["generate"]]
    assert len(gen) == 1
    assert gen[0]["argv"][gen[0]["argv"].index("-i") + 1] == ev["transport"][0]["path"]
    assert str(wav) not in gen[0]["argv"]


def test_ask_answer_and_focus(tmp_path):
    wav = _sine(tmp_path / "a.wav", seconds=3.0)
    env, _log = _env(
        tmp_path, FAKE_GEMINI_REPLY=_reply(tmp_path, {"summary": "s", "answer": "a sine", "claims": []})
    )
    proc = _run(["ask", str(wav), "what is it?", "--focus", "0.5-2.0"], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    assert ev["answer"] == "a sine"
    assert ev["transport"][0]["operation"] == "excerpt"
    assert ev["request"]["focus"] == {"start": 0.5, "end": 2.0}


def test_compare_identical_no_model_call(tmp_path):
    wav = _sine(tmp_path / "a.wav")
    env, log = _env(tmp_path)
    proc = _run(["compare", str(wav), str(wav)], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    assert ev["status"] == "complete"
    assert ev["claims"] == []
    assert ev["summary"] == "Inputs are PCM-identical; no perceptible difference is possible."
    assert [e for e in read_log(log) if e["argv"][:1] == ["generate"]] == []


def test_compare_difference_contradicts_fake_claim(tmp_path):
    rng = np.random.default_rng(1)
    a = _sine(tmp_path / "a.wav", seconds=2.0)
    data = np.repeat((0.2 * np.sin(2 * np.pi * 440 * np.arange(2 * SR) / SR))[:, None], 2, axis=1)
    b_data = data + rng.normal(0, 1e-6, data.shape)
    b = tmp_path / "b.wav"
    sf.write(str(b), b_data, SR, subtype="PCM_16", format="WAV")
    payload = {
        "summary": "s",
        "claims": [
            {
                "text": "something changed",
                "aspect": "timbre",
                "scale": "event",
                "loci": [{"clip": "A", "start": 0.5, "end": 1.0}],
                "relation": "differs",
                "basis": "audio",
                "confidence": "high",
            }
        ],
    }
    env, _log = _env(tmp_path, FAKE_GEMINI_REPLY=_reply(tmp_path, payload))
    proc = _run(["compare", str(a), str(b)], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    assert ev["claims"][0]["status"] == "contradicted"
    assert ev["claims"][0]["relation"] == "differs"


def test_compare_counterbalance(tmp_path):
    a = _sine(tmp_path / "a.wav", seconds=1.0)
    b = _sine(tmp_path / "b.wav", freq=660.0, seconds=1.0)
    payload = {
        "summary": "s",
        "claims": [
            {
                "text": "pitch differs",
                "aspect": "pitch",
                "scale": "work",
                "loci": [],
                "relation": "differs",
                "basis": "audio",
                "confidence": "high",
            }
        ],
    }
    env, log = _env(tmp_path, FAKE_GEMINI_REPLY=_reply(tmp_path, payload))
    proc = _run(["compare", str(a), str(b), "--counterbalance"], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    gens = [e for e in read_log(log) if e["argv"][:1] == ["generate"]]
    assert len(gens) == 2
    ids = {c["id"] for c in ev["claims"]}
    assert any(i.startswith("c") for i in ids) and any(i.startswith("k") for i in ids)

    def inputs_of(argv):
        return [argv[i + 1] for i, arg in enumerate(argv) if arg == "-i"]

    a_path, b_path = ev["transport"][0]["path"], ev["transport"][1]["path"]
    assert inputs_of(gens[0]["argv"]) == [a_path, b_path]
    assert inputs_of(gens[1]["argv"]) == [b_path, a_path]
    second_prompt = gens[1]["argv"][1]
    assert second_prompt.index("Clip B") < second_prompt.index("Clip A")


def test_investigate_with_measure(tmp_path):
    wav = _sine(tmp_path / "a.wav", seconds=2.0)
    env, _ = _env(tmp_path, FAKE_GEMINI_REPLY=_reply(tmp_path, {"summary": "s", "answer": "x", "claims": []}))
    proc = _run(
        ["investigate", str(wav), "--focus", "0.2-1.2", "--measure", "level,spectrum"],
        env,
    )
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    assert ev["task"] == "investigate"
    kinds = {m["kind"] for m in ev["measurements"]}
    assert kinds == {"level", "spectrum"}
    assert ev["request"]["focus"] == {"start": 0.2, "end": 1.2}


def test_review_assessments(tmp_path):
    wav = _sine(tmp_path / "a.wav")
    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps({"goals": ["keep it quiet"], "protect": [{"text": "bass stays"}]}))
    payload = {
        "summary": "s",
        "claims": [
            {
                "text": "quiet sine",
                "aspect": "dynamics",
                "scale": "work",
                "loci": [],
                "relation": "none",
                "basis": "audio",
                "confidence": "high",
            }
        ],
        "assessments": [
            {"target": "g1", "verdict": "met", "claims": [0], "note": "quiet throughout"},
            {"target": "p1", "verdict": "unclear", "claims": [7], "note": "n/a"},
        ],
    }
    env, _ = _env(tmp_path, FAKE_GEMINI_REPLY=_reply(tmp_path, payload))
    proc = _run(["review", str(wav), "--brief", str(brief)], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    assert ev["task"] == "review"
    assert ev["brief"]["goals"][0]["id"] == "g1"
    by_target = {a["target"]: a for a in ev["assessments"]}
    assert by_target["g1"]["verdict"] == "met"
    assert by_target["g1"]["claims"] == ["c0"]
    assert by_target["p1"]["verdict"] == "unclear"
    assert by_target["p1"]["claims"] == []


def test_review_empty_brief_is_usage_error(tmp_path):
    wav = _sine(tmp_path / "a.wav")
    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps({"goals": [], "protect": []}))
    env, _ = _env(tmp_path)
    proc = _run(["review", str(wav), "--brief", str(brief)], env)
    assert proc.returncode == 2


def test_identify(tmp_path):
    wav = _sine(tmp_path / "a.wav")
    env, _ = _env(tmp_path)
    proc = _run(["identify", str(wav)], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    assert ev["task"] == "identify"
    assert len(ev["inputs"]) == 1
    assert ev["inputs"][0]["format"]["sample_rate"] == SR


def test_check_then_tamper(tmp_path):
    wav = _sine(tmp_path / "a.wav")
    env, _ = _env(tmp_path)
    ev_path = tmp_path / "ev.json"
    proc = _run(
        [
            "identify",
            str(wav),
        ],
        env,
    )
    ev_path.write_text(proc.stdout)
    proc2 = _run(["check", str(ev_path)], env)
    assert proc2.returncode == 0, proc2.stderr
    assert json.loads(proc2.stdout)["status"] == "verified"
    _sine(wav, freq=880.0)
    proc3 = _run(["check", str(ev_path)], env)
    assert proc3.returncode == 1
    assert json.loads(proc3.stdout)["status"] == "mismatch"


def test_out_noclobber_and_overwrite(tmp_path):
    wav = _sine(tmp_path / "a.wav")
    out = tmp_path / "ev.json"
    env, log = _env(tmp_path)
    proc = _run(["hear", str(wav), "--out", str(out)], env)
    assert proc.returncode == 0, proc.stderr
    line = json.loads(proc.stdout)
    assert line["evidence"] == str(out)
    ev = json.loads(out.read_text())
    assert ev["schema"] == "agent-listening/evidence"
    log.unlink()
    proc2 = _run(["hear", str(wav), "--out", str(out)], env)
    assert proc2.returncode == 2
    assert not log.exists() or [e for e in read_log(log) if e["argv"][:1] == ["generate"]] == []
    proc3 = _run(["hear", str(wav), "--out", str(out), "--overwrite"], env)
    assert proc3.returncode == 0


def test_out_refuses_audio_ext(tmp_path):
    wav = _sine(tmp_path / "a.wav")
    env, _ = _env(tmp_path)
    proc = _run(["hear", str(wav), "--out", str(tmp_path / "ev.wav")], env)
    assert proc.returncode == 2


def test_out_refuses_input_path(tmp_path):
    wav = _sine(tmp_path / "a.wav")
    env, _ = _env(tmp_path)
    proc = _run(["hear", str(wav), "--out", str(wav)], env)
    assert proc.returncode == 2


def test_provider_failure_emits_failed_evidence(tmp_path):
    wav = _sine(tmp_path / "a.wav")
    env, _ = _env(tmp_path, FAKE_GEMINI_EXIT="3")
    proc = _run(["hear", str(wav)], env)
    assert proc.returncode == 3
    ev = json.loads(proc.stdout)
    assert ev["status"] == "failed"
    assert ev["error"]["code"] == "authentication_error"


def test_dry_run_no_provider_call(tmp_path):
    wav = _sine(tmp_path / "a.wav")
    env, log = _env(tmp_path)
    proc = _run(["hear", str(wav), "--dry-run"], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    assert ev["status"] == "partial"
    assert "DRY_RUN_NO_MODEL_CALL" in ev["limitations"]
    assert [e for e in read_log(log) if e["argv"][:1] == ["generate"]] == []


def test_doctor(tmp_path):
    env, _ = _env(tmp_path)
    proc = _run(["doctor"], env)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["gemini_api"]["version"] == "0.4.5"
    assert out["gemini_api"]["supported"] is True
    assert out["credential"] == "configured"
    env2, _ = _env(tmp_path, FAKE_GEMINI_NO_CREDENTIAL="1")
    proc2 = _run(["doctor"], env2)
    assert proc2.returncode == 3


def test_missing_input_is_failure(tmp_path):
    env, _ = _env(tmp_path)
    proc = _run(["hear", str(tmp_path / "nope.wav")], env)
    assert proc.returncode == 1


def test_cache_dir_expands_home(tmp_path):
    """Default cache dir honours $HOME; a literal ~ must never be created."""
    wav = _sine(tmp_path / "a.wav")
    home = tmp_path / "home"
    home.mkdir()
    env, _ = _env(tmp_path)
    env["HOME"] = str(home)
    del env["AGENT_LISTENING_CACHE"]
    env["PYTHONPATH"] = str(SRC)
    proc = _run(["hear", str(wav), "--dry-run"], env, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    derived = home / ".cache" / "agent-listening" / "derived"
    assert derived.is_dir() and any(derived.glob("*.flac"))
    assert not (tmp_path / "~").exists()


def test_investigate_question_option_any_order(tmp_path):
    wav = _sine(tmp_path / "a.wav", seconds=2.0)
    for argv in (
        ["investigate", "--focus", "0.5-1.5", str(wav), "--question", "what pitch"],
        ["investigate", str(wav), "--question", "what pitch", "--focus", "0.5-1.5"],
    ):
        env, log = _env(tmp_path)
        proc = _run(argv, env)
        assert proc.returncode == 0, proc.stderr
        gens = [e for e in read_log(log) if e["argv"][:1] == ["generate"]]
        assert len(gens) == 1
        assert "what pitch" in gens[0]["argv"][1]
        log.unlink()


def test_producer_attempts_in_evidence(tmp_path, monkeypatch):
    wav = _sine(tmp_path / "a.wav", seconds=1.0)
    env, _ = _env(
        tmp_path,
        FAKE_GEMINI_FAIL_SEQUENCE="rate_limit_error",
        FAKE_GEMINI_FAIL_MESSAGE="Rate limit hit. Please retry in 1s.",
    )
    proc = _run(["hear", str(wav)], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    assert ev["producer"]["attempts"] == 2


# --- measurement-guided compare follow-up ---


def _compare_pair(tmp_path, changes):
    """Two 5 s stereo WAVs; B = A with (start, end, gain) regions amplified."""
    sr = SR
    rng = np.random.default_rng(3)
    a_data = 0.05 * np.repeat(
        (np.sin(2 * np.pi * 330 * np.arange(5 * sr) / sr))[:, None], 2, axis=1
    ) + 0.2 * rng.standard_normal((5 * sr, 2))
    b_data = a_data.copy()
    for start, end, gain in changes:
        b_data[int(start * sr) : int(end * sr)] *= gain
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sf.write(str(a), a_data, sr, subtype="PCM_16", format="WAV")
    sf.write(str(b), b_data, sr, subtype="PCM_16", format="WAV")
    return a, b


def _replies_env(tmp_path, payloads):
    paths = []
    for i, payload in enumerate(payloads):
        p = tmp_path / f"reply{i}.json"
        p.write_text(json.dumps(payload))
        paths.append(str(p))
    return json.dumps(paths)


UNCHANGED_REPLY = {
    "summary": "sounds identical",
    "claims": [
        {
            "text": "no audible difference",
            "aspect": "timbre",
            "scale": "work",
            "loci": [],
            "relation": "unchanged",
            "basis": "audio",
            "confidence": "high",
        }
    ],
}


def _follow_reply():
    return {
        "summary": "the bell gains resonance",
        "claims": [
            {
                "text": "the bell sustains longer in B",
                "aspect": "timbre",
                "scale": "event",
                "loci": [{"clip": "A", "start": 2.0, "end": 3.0}],
                "relation": "differs",
                "basis": "audio",
                "confidence": "high",
            }
        ],
    }


def test_compare_follow_up_fires_on_uncovered_region(tmp_path):
    a, b = _compare_pair(tmp_path, [(1.0, 2.0, 3.0)])
    env, log = _env(tmp_path, FAKE_GEMINI_REPLIES=_replies_env(tmp_path, [UNCHANGED_REPLY, _follow_reply()]))
    proc = _run(["compare", str(a), str(b)], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    gens = [e for e in read_log(log) if e["argv"][:1] == ["generate"]]
    assert len(gens) == 2
    assert "deterministic measurement shows" in gens[1]["argv"][1]
    # main unchanged claim is contradicted by the measurement
    assert ev["claims"][0]["status"] == "contradicted"
    # follow-up: two excerpt transports appended after the two full ones
    follow_tr = ev["transport"][2:]
    assert len(follow_tr) == 2
    assert all(t["operation"] == "excerpt" for t in follow_tr)
    window_start = follow_tr[0]["offset_s"]
    assert window_start == pytest.approx(0.0, abs=0.2)
    # two -i args on the follow-up call are the excerpt paths
    follow_inputs = [gens[1]["argv"][i + 1] for i, v in enumerate(gens[1]["argv"]) if v == "-i"]
    assert follow_inputs == [follow_tr[0]["path"], follow_tr[1]["path"]]
    # follow-up claim: clip-time 2.0-3.0 maps back to input time (offset ~0)
    fclaim = next(c for c in ev["claims"] if c["id"].startswith("f0_"))
    assert fclaim["loci"][0]["start"] == pytest.approx(window_start + 2.0, abs=0.01)
    assert fclaim["loci"][0]["end"] == pytest.approx(window_start + 3.0, abs=0.01)
    kinds = {c["name"]: c["result"] for c in fclaim["checks"]}
    assert kinds["signal-difference"] == "pass"
    assert kinds["measurement-guided"] == "pass"
    assert fclaim["status"] == "consistent"
    assert "MEASUREMENT_GUIDED_FOLLOW_UP" in ev["limitations"]
    assert ev["producer"]["attempts"] == 2
    assert ev["request"]["follow_up"] == 2


def test_compare_follow_up_skipped_when_region_covered(tmp_path):
    a, b = _compare_pair(tmp_path, [(1.0, 2.0, 3.0)])
    covered_reply = {
        "summary": "louder middle",
        "claims": [
            {
                "text": "middle section louder",
                "aspect": "dynamics",
                "scale": "event",
                "loci": [{"clip": "A", "start": 1.0, "end": 2.0}],
                "relation": "differs",
                "basis": "audio",
                "confidence": "high",
            }
        ],
    }
    env, log = _env(tmp_path, FAKE_GEMINI_REPLY=_reply(tmp_path, covered_reply))
    proc = _run(["compare", str(a), str(b)], env)
    assert proc.returncode == 0, proc.stderr
    gens = [e for e in read_log(log) if e["argv"][:1] == ["generate"]]
    assert len(gens) == 1


def test_compare_follow_up_zero_disables(tmp_path):
    a, b = _compare_pair(tmp_path, [(1.0, 2.0, 3.0)])
    env, log = _env(tmp_path)
    proc = _run(["compare", str(a), str(b), "--follow-up", "0"], env)
    assert proc.returncode == 0, proc.stderr
    gens = [e for e in read_log(log) if e["argv"][:1] == ["generate"]]
    assert len(gens) == 1
    assert json.loads(proc.stdout)["request"]["follow_up"] == 0


def test_compare_follow_up_skipped_on_global_change(tmp_path):
    a, b = _compare_pair(tmp_path, [(0.0, 5.0, 4.0)])
    env, log = _env(tmp_path)
    proc = _run(["compare", str(a), str(b)], env)
    assert proc.returncode == 0, proc.stderr
    gens = [e for e in read_log(log) if e["argv"][:1] == ["generate"]]
    assert len(gens) == 1


def test_compare_follow_up_failure_keeps_main_result(tmp_path):
    a, b = _compare_pair(tmp_path, [(1.0, 2.0, 3.0)])
    env, _ = _env(
        tmp_path,
        FAKE_GEMINI_FAIL_SEQUENCE="ok,server_error,server_error,server_error",
        FAKE_GEMINI_FAIL_MESSAGE="High demand. Please retry in 0s.",
    )
    proc = _run(["compare", str(a), str(b)], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    assert ev["status"] == "complete"
    assert any(lim.startswith("FOLLOW_UP_FAILED:") for lim in ev["limitations"])


def test_compare_follow_up_picks_highest_peak(tmp_path):
    a, b = _compare_pair(tmp_path, [(1.0, 1.4, 1.5), (3.0, 3.5, 6.0)])
    env, log = _env(tmp_path, FAKE_GEMINI_REPLIES=_replies_env(tmp_path, [UNCHANGED_REPLY, _follow_reply()]))
    proc = _run(["compare", str(a), str(b), "--follow-up", "1"], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    gens = [e for e in read_log(log) if e["argv"][:1] == ["generate"]]
    assert len(gens) == 2
    # the louder (higher peak_db) region near 3s wins: its excerpt window starts ~1.0s
    assert ev["transport"][2]["offset_s"] == pytest.approx(1.0, abs=0.5)


def test_compare_follow_up_dry_run_skips(tmp_path):
    a, b = _compare_pair(tmp_path, [(1.0, 2.0, 3.0)])
    env, log = _env(tmp_path)
    proc = _run(["compare", str(a), str(b), "--dry-run"], env)
    assert proc.returncode == 0, proc.stderr
    assert not log.exists() or not [e for e in read_log(log) if e["argv"][:1] == ["generate"]]


def _tone_mix(seconds=5.0, seed_base=0):
    """Deterministic three-voice multi-partial mix; seed_base varies start phases."""
    t = np.arange(int(SR * seconds)) / SR

    def voice(f0, partials, amp, seed):
        rng = np.random.default_rng(seed)
        sig = np.zeros_like(t)
        for h, a_ in enumerate(partials, start=1):
            sig += a_ * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
        return amp * sig

    mix = (
        voice(110, [1, 0.5, 0.25, 0.12], 0.15, seed_base * 10 + 1)
        + voice(220, [1, 0.4, 0.2], 0.12, seed_base * 10 + 2)
        + voice(375, [1, 0.45, 0.22, 0.35], 0.10, seed_base * 10 + 3)
    )
    return np.repeat(mix[:, None], 2, axis=1)


def _write_pair(tmp_path, a_data, b_data):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sf.write(str(a), a_data, SR, subtype="PCM_16", format="WAV")
    sf.write(str(b), b_data, SR, subtype="PCM_16", format="WAV")
    return a, b


def _claim(text, relation, clip="A", start=0.0, end=5.0):
    return {
        "text": text,
        "aspect": "timbre",
        "scale": "work",
        "loci": [{"clip": clip, "start": start, "end": end}],
        "relation": relation,
        "basis": "audio",
        "confidence": "high",
    }


def test_compare_phase_only_waveform_global(tmp_path):
    """Phase re-render: waveform diff is global; spectral is clean, so an
    'unchanged' claim is consistent and an unsupported 'differs' is contradicted."""
    a, b = _write_pair(tmp_path, _tone_mix(), _tone_mix(seed_base=1))
    reply = {
        "summary": "identical",
        "claims": [
            _claim("no audible difference", "unchanged"),
            _claim("the bass changed", "differs", start=1.0, end=2.0),
        ],
    }
    env, _ = _env(tmp_path, FAKE_GEMINI_REPLY=_reply(tmp_path, reply))
    proc = _run(["compare", str(a), str(b)], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    diff = next(m for m in ev["measurements"] if m["kind"] == "signal-difference")
    assert diff["result"]["global_change"] is True  # waveform is phase-sensitive
    assert diff["result"]["spectral"]["regions"] == []
    by_rel = {c["relation"]: c for c in ev["claims"]}
    assert by_rel["unchanged"]["status"] == "consistent"
    assert by_rel["differs"]["status"] == "contradicted"


def test_compare_spectral_global_contradicts_unchanged(tmp_path):
    """Global spectral change (waveshaping): 'unchanged' is contradicted even
    though the waveform check is inapplicable."""
    a_data = _tone_mix()
    a, b = _write_pair(tmp_path, a_data, np.tanh(2.5 * a_data) * 0.8)
    reply = {"summary": "same", "claims": [_claim("no audible difference", "unchanged")]}
    env, _ = _env(tmp_path, FAKE_GEMINI_REPLY=_reply(tmp_path, reply))
    proc = _run(["compare", str(a), str(b)], env)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(proc.stdout)
    diff = next(m for m in ev["measurements"] if m["kind"] == "signal-difference")
    assert diff["result"]["spectral"]["global_change"] is True
    assert ev["claims"][0]["status"] == "contradicted"
