from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

FAKE_GEMINI = Path(__file__).parent / "fake_gemini_api.py"


def write_wav(path: Path, data: np.ndarray, sr: int, subtype: str = "PCM_16", fmt: str = "WAV") -> Path:
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[:, None]
    sf.write(str(path), arr, sr, subtype=subtype, format=fmt)
    return path


def sine(sr: int, seconds: float, freq: float = 440.0, amp: float = 0.2, channels: int = 2) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    mono = amp * np.sin(2 * np.pi * freq * t)
    return np.repeat(mono[:, None], channels, axis=1)


@pytest.fixture()
def wav_factory(tmp_path):
    counter = [0]

    def make(data: np.ndarray, sr: int = 48000, subtype: str = "PCM_16", fmt: str = "WAV") -> Path:
        counter[0] += 1
        path = tmp_path / f"sig{counter[0]}.{fmt.lower()}"
        return write_wav(path, data, sr, subtype=subtype, fmt=fmt)

    return make


@pytest.fixture()
def fake_env(tmp_path):
    """Env for the fake provider; returns (env dict, log path)."""
    log = tmp_path / "fake-gemini.log"

    def make(**extra):
        env = dict(os.environ)
        env["FAKE_GEMINI_LOG"] = str(log)
        env["AGENT_LISTENING_GEMINI_API"] = str(FAKE_GEMINI)
        env.update({k: str(v) for k, v in extra.items()})
        return env, log

    return make


def read_log(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]
