# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "soundfile>=0.12,<0.14"]
# ///
"""Decision-change scenarios: a goal, a defect render and a paired control, and the registered answer.

Each scenario is built so the deciding fact is a property of the rendered sound. Every voice's own
parameters are plausible in both renders; what differs is how the voices interact once rendered.
The answer key comes from the recipe and is checked by construction measurements, never by a model.
"""

from __future__ import annotations

import inspect
import json

import numpy as np

SR = 48_000
NO_CHANGE = "no-change"


def _one_pole(x: np.ndarray, cutoff: float) -> np.ndarray:
    a = float(np.exp(-2 * np.pi * cutoff / SR))
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = (1 - a) * v + a * acc
        y[i] = acc
    return y


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.max(np.abs(x))


def _unit_rms(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x**2))


def trio(p: dict) -> dict[str, np.ndarray]:
    """Three voices: a sustained pad, a melody of short notes, a pulsing bass."""
    t = np.arange(int(p["seconds"] * SR)) / SR
    chord = sum(np.sin(2 * np.pi * p["padHz"] * ratio * t) for ratio in (1.0, 1.498, 2.003))
    pad = p["padAmp"] * _unit_rms(_one_pole(np.tanh(p["padDrive"] * chord), p["padCutoff"]))
    melody = np.zeros_like(t)
    step = p["seconds"] / len(p["melodyHz"])
    for i, hz in enumerate(p["melodyHz"]):
        local = t - i * step
        on = (local >= 0) & (local < step)
        env = np.exp(-3 * local[on] / step) * np.minimum(1, local[on] / 0.01)
        melody[on] = env * (np.sin(2 * np.pi * hz * local[on]) + 0.3 * np.sin(4 * np.pi * hz * local[on]))
    melody = p["melodyAmp"] * _unit(melody)
    pulse = np.exp(-6 * ((t * p["bassRate"]) % 1))
    bass = p["bassAmp"] * _unit(pulse * np.sin(2 * np.pi * p["bassHz"] * t))
    return {"pad": pad, "melody": melody, "bass": bass}


RENDERERS = {"trio": trio}

_TRIO = {
    "seconds": 8.0,
    "padHz": 110.0,
    "padAmp": 0.1,
    "padCutoff": 5000.0,
    "melodyHz": [880.0, 987.8, 1108.7, 1318.5, 1174.7, 987.8, 1108.7, 880.0],
    "melodyAmp": 0.03,
    "bassHz": 55.0,
    "bassRate": 2.0,
    "bassAmp": 0.25,
}

SCENARIOS = {
    "masked-melody": {
        "family": "interaction",
        "provenance": "synthetic, written for this evaluation (issue #17); never used in a model call",
        "goal": {
            "attribution": "human",
            "text": (
                "Each of the three voices should be heard as its own line. If one voice is covered by the "
                "others, name the voice to bring forward. If all three already come through, change nothing."
            ),
        },
        "renderer": "trio",
        "defect": {**_TRIO, "padDrive": 6.0},
        "control": {**_TRIO, "padDrive": 0.3},
        "voices": ["pad", "melody", "bass"],
        "listening_question": "Which separate voices or lines can be followed in this passage, and is any of them hard to follow?",
        "answer": {"defect": "raise:melody", "control": NO_CHANGE},
        # construction check: melody level relative to the other voices inside the melody's own band
        "construction": {
            "voice": "melody",
            "band_hz": [800.0, 2800.0],
            "defect_max_db": -6.0,
            "control_min_db": 6.0,
        },
    },
}


def options(scenario: dict) -> list[str]:
    return [f"raise:{voice}" for voice in scenario["voices"]] + [NO_CHANGE]


def answer_keys(scenarios: dict = SCENARIOS) -> dict:
    return {
        name: {"family": s["family"], "options": options(s), **s["answer"]} for name, s in scenarios.items()
    }


def stems(scenario: dict, render: str) -> dict[str, np.ndarray]:
    return RENDERERS[scenario["renderer"]](scenario[render])


def mix(scenario: dict, render: str) -> np.ndarray:
    return np.sum(list(stems(scenario, render).values()), axis=0)


def source_text(scenario: dict, render: str) -> str:
    """What the caller reads: this render's parameters and all the code that turns them into sound."""
    params = json.dumps(scenario[render], indent=2)
    code = "\n\n".join(
        inspect.getsource(f) for f in (_one_pole, _unit, _unit_rms, RENDERERS[scenario["renderer"]])
    )
    return f"SR = {SR}\np = {params}\n\n{code}"


def band_ratio_db(scenario: dict, render: str) -> float:
    """Registered voice against the sum of the other voices, within the registered band."""
    spec = scenario["construction"]
    parts = stems(scenario, render)
    others = np.sum([x for name, x in parts.items() if name != spec["voice"]], axis=0)
    freqs = np.fft.rfftfreq(len(others), 1 / SR)
    band = (freqs >= spec["band_hz"][0]) & (freqs <= spec["band_hz"][1])

    def power(x: np.ndarray) -> float:
        return float(np.sum(np.abs(np.fft.rfft(x))[band] ** 2))

    return 10 * float(np.log10(power(parts[spec["voice"]]) / power(others)))
