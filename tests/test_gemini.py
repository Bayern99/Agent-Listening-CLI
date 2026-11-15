from __future__ import annotations

import json

import numpy as np
import pytest

from agent_listening import gemini
from conftest import FAKE_GEMINI, read_log, write_wav


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_GEMINI_LOG", str(log))
    return log


def _audio(tmp_path):
    write_wav(tmp_path / "a.wav", np.zeros((4800, 1)), 48000)
    return tmp_path / "a.wav"


def test_listen_argv_and_stdin(tmp_path, fake, monkeypatch):
    reply_file = tmp_path / "reply.json"
    reply_file.write_text(json.dumps({"summary": "ok", "claims": [], "answer": "yes"}))
    monkeypatch.setenv("FAKE_GEMINI_REPLY", str(reply_file))
    wav = _audio(tmp_path)
    reply = gemini.listen(
        executable=str(FAKE_GEMINI),
        model="gemini-4-hydrogen",
        thinking="high",
        system="sys",
        prompt="prompt",
        audio=[wav],
        schema={"type": "object"},
    )
    gen = [e for e in read_log(fake) if e["argv"][:1] == ["generate"]]
    assert len(gen) == 1
    argv = gen[0]["argv"]
    assert "gemini-4-hydrogen" in argv
    assert "--stream=false" in argv
    assert "--agent-mode" in argv
    assert str(wav) in argv
    body = json.loads(gen[0]["stdin"])
    assert body["generation_config"]["thinking_level"] == "high"
    assert body["response_format"]["mime_type"] == "application/json"
    assert body["response_format"]["schema"] == {"type": "object"}
    assert reply.payload["answer"] == "yes"
    assert reply.usage == {"input_tokens": 100, "audio_tokens": 64, "output_tokens": 20, "thought_tokens": 7}
    assert reply.cli_version == "0.4.5"


def test_listen_strips_fences(tmp_path, fake, monkeypatch):
    reply_file = tmp_path / "reply.json"
    reply_file.write_text(json.dumps({"summary": "s", "claims": []}))
    monkeypatch.setenv("FAKE_GEMINI_REPLY", str(reply_file))
    reply = gemini.listen(
        executable=str(FAKE_GEMINI),
        model="m",
        thinking="low",
        system="s",
        prompt="p",
        audio=[],
        schema={},
    )
    assert reply.payload["summary"] == "s"


def test_listen_exit3_is_auth(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_GEMINI_EXIT", "3")
    with pytest.raises(gemini.ProviderError) as exc:
        gemini.listen(
            executable=str(FAKE_GEMINI),
            model="m",
            thinking="low",
            system="s",
            prompt="p",
            audio=[],
            schema={},
        )
    assert exc.value.code == "authentication_error"
    assert exc.value.exit_code == 3


def test_listen_missing_executable(tmp_path, fake):
    with pytest.raises(gemini.ProviderError) as exc:
        gemini.listen(
            executable="/nonexistent/gemini-api-zzz",
            model="m",
            thinking="low",
            system="s",
            prompt="p",
            audio=[],
            schema={},
        )
    assert exc.value.code == "PROVIDER_MISSING"


def test_listen_uploads_large_file(tmp_path, fake):
    big = tmp_path / "big.wav"
    write_wav(big, np.zeros((48000, 2)), 48000)
    reply = gemini.listen(
        executable=str(FAKE_GEMINI),
        model="m",
        thinking="low",
        system="s",
        prompt="p",
        audio=[big],
        schema={},
        inline_limit_bytes=1,
    )
    entries = read_log(fake)
    uploads = [e for e in entries if e["argv"][:2] == ["files", "upload"]]
    deletes = [e for e in entries if e["argv"][:2] == ["files", "delete"]]
    gen = [e for e in entries if e["argv"][:1] == ["generate"]]
    assert len(uploads) == 1 and str(big) in uploads[0]["argv"]
    assert len(deletes) == 1 and "files/fake-upload-id" in deletes[0]["argv"]
    assert "files/fake-upload-id" in gen[0]["argv"]
    assert str(big) not in gen[0]["argv"]
    assert reply.payload


def test_version_and_credentials(tmp_path, fake, monkeypatch):
    assert gemini.version(str(FAKE_GEMINI)) == "0.4.5"
    assert gemini.credential_status(str(FAKE_GEMINI)) == "configured"
    monkeypatch.setenv("FAKE_GEMINI_NO_CREDENTIAL", "1")
    assert gemini.credential_status(str(FAKE_GEMINI)) == "missing"


def test_version_parses_other_series(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_GEMINI_VERSION", "0.5.0")
    assert gemini.version(str(FAKE_GEMINI)) == "0.5.0"


def test_listen_rejects_unsupported_cli(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_GEMINI_VERSION", "0.5.0")
    with pytest.raises(gemini.ProviderError) as exc:
        gemini.listen(
            executable=str(FAKE_GEMINI),
            model="m",
            thinking="low",
            system="s",
            prompt="p",
            audio=[],
            schema={},
        )
    assert exc.value.code == "PROVIDER_UNSUPPORTED"
    assert exc.value.exit_code == 3


def test_inline_limit_is_per_request(tmp_path, fake):
    """Total payload over the limit uploads every input, even small ones."""
    small1 = tmp_path / "s1.wav"
    small2 = tmp_path / "s2.wav"
    write_wav(small1, np.zeros((4800, 1)), 48000)
    write_wav(small2, np.zeros((4800, 1)), 48000)
    each = small1.stat().st_size + small2.stat().st_size
    gemini.listen(
        executable=str(FAKE_GEMINI),
        model="m",
        thinking="low",
        system="s",
        prompt="p",
        audio=[small1, small2],
        schema={},
        inline_limit_bytes=each - 1,
    )
    entries = read_log(fake)
    uploads = [e for e in entries if e["argv"][:2] == ["files", "upload"]]
    deletes = [e for e in entries if e["argv"][:2] == ["files", "delete"]]
    gen = next(e for e in entries if e["argv"][:1] == ["generate"])
    assert len(uploads) == 2 and len(deletes) == 2
    assert gen["argv"].count("files/fake-upload-id") == 2
    assert str(small1) not in gen["argv"] and str(small2) not in gen["argv"]


def test_under_limit_stays_inline(tmp_path, fake):
    wav = _audio(tmp_path)
    gemini.listen(
        executable=str(FAKE_GEMINI),
        model="m",
        thinking="low",
        system="s",
        prompt="p",
        audio=[wav],
        schema={},
    )
    entries = read_log(fake)
    assert not [e for e in entries if e["argv"][:2] == ["files", "upload"]]
    gen = next(e for e in entries if e["argv"][:1] == ["generate"])
    assert str(wav) in gen["argv"]


def test_credential_env_source_is_configured(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_GEMINI_CREDENTIAL", "env")
    assert gemini.credential_status(str(FAKE_GEMINI)) == "configured"


def test_credential_keyring_source_is_configured(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_GEMINI_CREDENTIAL", "keyring")
    assert gemini.credential_status(str(FAKE_GEMINI)) == "configured"
