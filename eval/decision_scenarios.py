# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "soundfile>=0.12,<0.14"]
# ///
"""Decision-change scenarios: a goal, a defect render and a paired control, and the registered answer.

Each scenario is built so the deciding fact is a property of the rendered sound.

- interaction: every voice's own parameters are plausible in both renders; what differs is how the
  voices interact once rendered (masking, a difference tone, beating, a shared resonance, register).
- hidden: the source declares a random process. The seed is fixed in the recipe and withheld from the
  caller, so defect and control read identically and only the render says what was drawn.

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


def _sines(t: np.ndarray, hz: float, partials: list[float]) -> np.ndarray:
    return sum(a * np.sin(2 * np.pi * hz * (k + 1) * t) for k, a in enumerate(partials))


def _line(
    t: np.ndarray, hzs: list[float], partials: list[float], decay: float, start: float = 0.0
) -> np.ndarray:
    """Equal-length notes one after another, each with a 10 ms attack and an exponential decay."""
    out = np.zeros_like(t)
    step = (t[-1] - start) / len(hzs)
    for i, hz in enumerate(hzs):
        local = t - start - i * step
        on = (local >= 0) & (local < step)
        out[on] = (
            np.exp(-decay * local[on] / step)
            * np.minimum(1, local[on] / 0.01)
            * _sines(local[on], hz, partials)
        )
    return out


def _lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(np.fft.rfft(x) / np.sqrt(1 + (f / cutoff) ** 8), len(x))


def _comb(x: np.ndarray, delay_ms: float, feedback: float) -> np.ndarray:
    """Feedback comb y[n] = x[n] + feedback * y[n - D], applied in the frequency domain."""
    d = round(SR * delay_ms / 1000)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(np.fft.rfft(x) / (1 - feedback * np.exp(-2j * np.pi * f * d / SR)), len(x))


def ring(p: dict) -> dict[str, np.ndarray]:
    """A lead line, a ring-modulated bell tone, a pulsing bass."""
    t = np.arange(int(p["seconds"] * SR)) / SR
    lead = p["leadAmp"] * _unit(_line(t, p["leadHz"], [1.0, 0.3], 3.0))
    bell = p["bellAmp"] * _unit_rms(
        np.sin(2 * np.pi * p["bellHz"] * t) * np.sin(2 * np.pi * p["bellModHz"] * t)
    )
    bass = p["bassAmp"] * _unit(_line(t, p["bassHz"], [1.0], 4.0))
    return {"lead": lead, "bell": bell, "bass": bass}


def consort(p: dict) -> dict[str, np.ndarray]:
    """A held organ tone, a two-partial chime, a pulsing bass."""
    t = np.arange(int(p["seconds"] * SR)) / SR
    organ = p["organAmp"] * _unit(_sines(t, p["organHz"], p["organPartials"]))
    chime = p["chimeAmp"] * _unit_rms(
        np.sin(2 * np.pi * p["chimeHz"] * t) + np.sin(2 * np.pi * p["chimeHz"] * p["chimeRatio"] * t)
    )
    bass = p["bassAmp"] * _unit(_line(t, p["bassHz"], [1.0, 0.2], 4.0))
    return {"organ": organ, "chime": chime, "bass": bass}


def room(p: dict) -> dict[str, np.ndarray]:
    """Three lines played into one resonant space (a feedback comb on the shared bus)."""
    t = np.arange(int(p["seconds"] * SR)) / SR
    dry = {
        "low": p["lowAmp"] * _unit(_line(t, p["lowHz"], [1.0, 0.2], 1.5)),
        "mid": p["midAmp"] * _unit(_line(t, p["midHz"], [1.0, 0.2], 1.5)),
        "high": p["highAmp"] * _unit(_line(t, p["highHz"], [1.0, 0.2], 1.5)),
    }
    return {name: _comb(x, p["combMs"], p["combFeedback"]) for name, x in dry.items()}


def weave(p: dict) -> dict[str, np.ndarray]:
    """A lead line, a counter line given as semitone steps above a base pitch, a bass."""
    t = np.arange(int(p["seconds"] * SR)) / SR
    lead = p["leadAmp"] * _unit(_line(t, p["leadHz"], [1.0, 0.1], 2.0))
    counter_hz = [
        p["counterBaseHz"] * 2 ** ((step + p["counterTranspose"]) / 12) for step in p["counterSteps"]
    ]
    counter = p["counterAmp"] * _unit(_line(t, counter_hz, [1.0, 0.1], 2.0))
    bass = p["bassAmp"] * _unit(_line(t, p["bassHz"], [1.0, 0.2], 3.0))
    return {"lead": lead, "counter": counter, "bass": bass}


def aleatoric(p: dict) -> dict[str, np.ndarray]:
    """Three lines; each performance draws a level, tuning, entry delay and filter per voice."""
    rng = np.random.default_rng(p["seed"])
    t = np.arange(int(p["seconds"] * SR)) / SR
    gain_db = rng.uniform(*p["gainDb"], 3)
    cents = rng.uniform(*p["detuneCents"], 3)
    late = rng.uniform(*p["lateSeconds"], 3)
    cutoff = np.exp(rng.uniform(*np.log(p["cutoffHz"]), 3))
    out = {}
    for i, name in enumerate(("low", "mid", "high")):
        hzs = [hz * 2 ** (cents[i] / 1200) for hz in p[name + "Hz"]]
        voice = _lowpass(_line(t, hzs, [1.0, 0.5, 0.33, 0.25, 0.2, 0.17], 2.0, late[i]), cutoff[i])
        out[name] = p["amp"] * 10 ** (gain_db[i] / 20) * voice
    return out


def wander(p: dict) -> dict[str, np.ndarray]:
    """A drone and a melody that takes a random step of -2..2 semitones before every note."""
    rng = np.random.default_rng(p["seed"])
    t = np.arange(int(p["seconds"] * SR)) / SR
    steps = np.cumsum(rng.integers(-2, 3, p["notes"]))
    melody = p["melodyAmp"] * _unit(
        _line(t, [p["startHz"] * 2 ** (s / 12) for s in steps - steps[0]], [1.0, 0.3], 2.0)
    )
    drone = p["droneAmp"] * _unit(_sines(t, p["droneHz"], [1.0, 0.3]))
    return {"melody": melody, "drone": drone}


RENDERERS = {
    "trio": (trio, [_one_pole, _unit, _unit_rms]),
    "ring": (ring, [_unit, _unit_rms, _sines, _line]),
    "consort": (consort, [_unit, _unit_rms, _sines, _line]),
    "room": (room, [_unit, _sines, _line, _comb]),
    "weave": (weave, [_unit, _sines, _line]),
    "aleatoric": (aleatoric, [_sines, _line, _lowpass]),
    "wander": (wander, [_unit, _sines, _line]),
}
HIDDEN = ("seed",)

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
_RING = {
    "seconds": 8.0,
    "leadHz": [659.3, 740.0, 830.6, 987.8, 880.0, 740.0, 830.6, 659.3],
    "leadAmp": 0.15,
    "bellHz": 250.0,
    "bellAmp": 0.15,
    "bassHz": [55.0, 61.7, 65.4, 55.0, 61.7, 65.4, 61.7, 55.0],
    "bassAmp": 0.08,
}
_CONSORT = {
    "seconds": 8.0,
    "organHz": 220.0,
    "organPartials": [1.0, 0.5, 0.35, 0.3],
    "organAmp": 0.2,
    "chimeHz": 330.0,
    "chimeAmp": 0.06,
    "bassHz": [55.0, 55.0, 82.4, 55.0],
    "bassAmp": 0.2,
}
_ROOM = {
    "seconds": 8.0,
    "lowHz": [110.0, 123.5, 110.0, 98.0],
    "lowAmp": 0.12,
    "midHz": [196.0, 196.0, 196.0, 196.0],
    "midAmp": 0.12,
    "highHz": [329.6, 370.0, 415.3, 370.0],
    "highAmp": 0.12,
    "combFeedback": 0.7,
}
_WEAVE = {
    "seconds": 8.0,
    "leadHz": [587.3, 659.3, 740.0, 880.0, 784.0, 659.3, 740.0, 587.3],
    "leadAmp": 0.15,
    "counterBaseHz": 110.0,
    "counterSteps": [7, 5, 3, 0, 2, 5, 3, 7],
    "counterAmp": 0.15,
    "bassHz": [73.4, 73.4, 55.0, 73.4],
    "bassAmp": 0.2,
}
_ALEA = {
    "seconds": 8.0,
    "amp": 0.15,
    "lowHz": [110.0, 130.8, 146.8, 130.8],
    "midHz": [220.0, 246.9, 220.0, 196.0],
    "highHz": [440.0, 523.3, 587.3, 523.3],
    "gainDb": [0.0, 0.0],
    "detuneCents": [0.0, 0.0],
    "lateSeconds": [0.0, 0.0],
    "cutoffHz": [9000.0, 9000.0],
}
_WANDER = {
    "seconds": 8.0,
    "notes": 16,
    "startHz": 440.0,
    "melodyAmp": 0.15,
    "droneHz": 110.0,
    "droneAmp": 0.1,
}
_VOICES3 = ["low", "mid", "high"]


def _scenario(
    family: str, kind: str, goal: str, question: str, renderer: str, actions: list[str], **rest
) -> dict:
    return {
        "family": family,
        "kind": kind,
        "provenance": "synthetic, written for this evaluation (issue #16); not from the 1.0 corpus or holdout",
        "goal": {"attribution": "human", "text": goal},
        "listening_question": question,
        "renderer": renderer,
        "options": [*actions, NO_CHANGE],
        **rest,
    }


def _hidden(
    kind: str,
    goal: str,
    question: str,
    action: str,
    spread: dict,
    seeds: tuple[int, int],
    voice: str,
    construction: dict,
) -> dict:
    params = {**_ALEA, **spread}
    return _scenario(
        "hidden",
        kind,
        goal,
        question,
        "aleatoric",
        [f"{action}:{v}" for v in _VOICES3],
        defect={**params, "seed": seeds[0]},
        control={**params, "seed": seeds[1]},
        answer={"defect": f"{action}:{voice}", "control": NO_CHANGE},
        construction=construction,
    )


SCENARIOS = {
    "masked-melody": _scenario(
        "interaction",
        "masking",
        "Each of the three voices should be heard as its own line. If one voice is covered by the others, name the voice to bring forward. If all three already come through, change nothing.",
        "Which separate voices or lines can be followed in this passage, and is any of them hard to follow?",
        "trio",
        ["raise:pad", "raise:melody", "raise:bass"],
        defect={**_TRIO, "padDrive": 6.0},
        control={**_TRIO, "padDrive": 0.3},
        answer={"defect": "raise:melody", "control": NO_CHANGE},
        construction={
            "measure": "covered_db",
            "bands": {"pad": [90, 400], "melody": [800, 2800], "bass": [40, 80]},
            "bad": 6.0,
            "ok": -6.0,
        },
    ),
    "masked-bass": _scenario(
        "interaction",
        "difference-tone",
        "Each of the three voices should be heard as its own line. If one voice is covered by the others, name the voice to bring forward. If all three already come through, change nothing.",
        "Which separate voices or lines can be followed in this passage, and is any of them hard to follow?",
        "ring",
        ["raise:lead", "raise:bell", "raise:bass"],
        defect={**_RING, "bellModHz": 190.0},
        control={**_RING, "bellModHz": 110.0},
        answer={"defect": "raise:bass", "control": NO_CHANGE},
        construction={
            "measure": "covered_db",
            "bands": {"lead": [600, 1100], "bell": [100, 500], "bass": [40, 80]},
            "bad": 6.0,
            "ok": -6.0,
        },
    ),
    "beating-chime": _scenario(
        "interaction",
        "beating",
        "The organ tone is the tuning reference and should sound steady. If another voice makes it waver by beating against it, name the voice to retune. If the organ is steady, change nothing.",
        "Does any sustained sound in this passage waver, pulse or beat, and if so which sounds are involved?",
        "consort",
        ["retune:chime", "retune:bass"],
        defect={**_CONSORT, "chimeRatio": 2.68},
        control={**_CONSORT, "chimeRatio": 2.5},
        answer={"defect": "retune:chime", "control": NO_CHANGE},
        construction={
            "measure": "beat_depth",
            "reference": "organ",
            "band": [840, 920],
            "bad": 0.3,
            "ok": 0.05,
        },
    ),
    "ringing-room": _scenario(
        "interaction",
        "resonance",
        "The three lines should sit at even weight in the room. If the room makes one line boom or ring out over the others, name the voice to pull back. If they are even, change nothing.",
        "How do the separate lines balance in this passage, and does any of them ring or boom more than the others?",
        "room",
        ["lower:low", "lower:mid", "lower:high"],
        defect={**_ROOM, "combMs": 5.1},
        control={**_ROOM, "combMs": 3.7},
        answer={"defect": "lower:mid", "control": NO_CHANGE},
        construction={"measure": "gain_db", "override": {"combFeedback": 0.0}, "bad": 6.0, "ok": 3.0},
    ),
    "crowded-register": _scenario(
        "interaction",
        "register",
        "The lead should have its register to itself. If another voice plays in the lead's register, name the voice to move down. If the lead has clear space, change nothing.",
        "Which lines are in this passage, and do any of them share the same pitch range?",
        "weave",
        ["move-down:counter", "move-down:bass"],
        defect={**_WEAVE, "counterTranspose": 24},
        control={**_WEAVE, "counterTranspose": 0},
        answer={"defect": "move-down:counter", "control": NO_CHANGE},
        construction={
            "measure": "presence_db",
            "reference": "lead",
            "band": [500, 1000],
            "bad": -6.0,
            "ok": -15.0,
        },
    ),
    "uneven-levels": _hidden(
        "level",
        "The three voices should come out about equally loud. If one voice came out clearly quieter than the other two in this performance, name the voice to bring up. If they are even, change nothing.",
        "How do the three lines balance in loudness in this passage?",
        "raise",
        {"gainDb": [-24.0, 0.0]},
        (82, 13),
        "high",
        {"measure": "below_loudest_db", "bad": 14.0, "ok": 6.0},
    ),
    "stray-tuning": _hidden(
        "tuning",
        "The three voices should be in tune with each other. If one voice came out clearly out of tune in this performance, name the voice to retune. If they are in tune, change nothing.",
        "Are the lines in this passage in tune with each other?",
        "retune",
        {"detuneCents": [-60.0, 60.0]},
        (577, 748),
        "low",
        {"measure": "off_pitch_cents", "bad": 40.0, "ok": 10.0},
    ),
    "late-entry": _hidden(
        "timing",
        "The three voices should begin together. If one voice came in clearly after the other two in this performance, name the voice to move earlier. If they begin together, change nothing.",
        "When does each line enter at the start of this passage?",
        "earlier",
        {"lateSeconds": [0.0, 1.0]},
        (1543, 7381),
        "mid",
        {"measure": "entry_seconds", "bad": 0.5, "ok": 0.1},
    ),
    "dull-voice": _hidden(
        "brightness",
        "The three voices should share a similar, open tone. If one voice came out clearly duller than the other two in this performance, name the voice to brighten. If they match, change nothing.",
        "How does the tone colour of each line compare in this passage?",
        "brighten",
        {"cutoffHz": [250.0, 9000.0]},
        (368, 173),
        "high",
        {
            "measure": "gain_db",
            "override": {"cutoffHz": [9000.0, 9000.0]},
            "band": [1500, 8000],
            "sign": -1.0,
            "bad": 20.0,
            "ok": 6.0,
        },
    ),
    "drifting-line": _scenario(
        "hidden",
        "drift",
        "The melody should end close to the pitch it started on. If in this performance it ended more than a fourth above its start, shift it down; if more than a fourth below, shift it up. If it ended close, change nothing.",
        "How does the pitch of the melody at the end of this passage compare with its pitch at the start?",
        "wander",
        ["shift:down", "shift:up"],
        defect={**_WANDER, "seed": 4},
        control={**_WANDER, "seed": 35},
        answer={"defect": "shift:down", "control": NO_CHANGE},
        construction={"measure": "drift_semitones", "defect": [6.0, None], "control": [-2.0, 2.0]},
    ),
}


def chance(scenario: dict) -> float:
    return 1 / len(scenario["options"])


def answer_keys(scenarios: dict = SCENARIOS) -> dict:
    return {
        name: {"family": s["family"], "options": s["options"], **s["answer"]} for name, s in scenarios.items()
    }


def stems(scenario: dict, render: str, override: dict | None = None) -> dict[str, np.ndarray]:
    return RENDERERS[scenario["renderer"]][0]({**scenario[render], **(override or {})})


def mix(scenario: dict, render: str) -> np.ndarray:
    return np.sum(list(stems(scenario, render).values()), axis=0)


def source_text(scenario: dict, render: str) -> str:
    """What the caller reads: the code that makes the sound and this render's parameters, hidden keys left out."""
    renderer, helpers = RENDERERS[scenario["renderer"]]
    shown = {k: v for k, v in scenario[render].items() if k not in HIDDEN}
    withheld = [k for k in scenario[render] if k in HIDDEN]
    note = "".join(
        f'# p["{k}"] was fixed when this performance was rendered and is not shown\n' for k in withheld
    )
    code = "\n\n".join(inspect.getsource(f) for f in (*helpers, renderer))
    return f"SR = {SR}\n{note}p = {json.dumps(shown, indent=2)}\n\n{code}"


# --- construction measures: what the recipe claims about the rendered sound ---------------------


def _band_power(x: np.ndarray, band: list[float] | None) -> float:
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    if band is None:
        return float(np.sum(spectrum))
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(np.sum(spectrum[(f >= band[0]) & (f <= band[1])]))


def _db(ratio: float) -> float:
    return 10 * float(np.log10(max(ratio, 1e-12)))


def _others(parts: dict, voice: str) -> np.ndarray:
    return np.sum([x for name, x in parts.items() if name != voice], axis=0)


def covered_db(scenario: dict, render: str, voice: str, spec: dict) -> float:
    """How far the other voices exceed this voice inside this voice's own band."""
    parts = stems(scenario, render)
    band = spec["bands"][voice]
    return _db(_band_power(_others(parts, voice), band) / _band_power(parts[voice], band))


def presence_db(scenario: dict, render: str, voice: str, spec: dict) -> float:
    """This voice's level inside the reference voice's band, relative to the reference."""
    parts = stems(scenario, render)
    return _db(_band_power(parts[voice], spec["band"]) / _band_power(parts[spec["reference"]], spec["band"]))


def beat_depth(scenario: dict, render: str, voice: str, spec: dict) -> float:
    """Envelope fluctuation (std / mean) of reference plus this voice inside the band."""
    parts = stems(scenario, render)
    x = parts[spec["reference"]] + parts[voice]
    f = np.fft.fftfreq(len(x), 1 / SR)
    analytic = np.fft.ifft(np.where((f >= spec["band"][0]) & (f <= spec["band"][1]), 2 * np.fft.fft(x), 0))
    envelope = np.abs(analytic)[SR // 2 : -SR // 2]
    return float(np.std(envelope) / np.mean(envelope))


def gain_db(scenario: dict, render: str, voice: str, spec: dict) -> float:
    """This voice's level against the same voice rendered with the override parameters."""
    band = spec.get("band")
    here = _band_power(stems(scenario, render)[voice], band)
    there = _band_power(stems(scenario, render, spec["override"])[voice], band)
    return spec.get("sign", 1.0) * _db(here / there)


def below_loudest_db(scenario: dict, render: str, voice: str, spec: dict) -> float:
    parts = stems(scenario, render)
    return _db(max(_band_power(x, None) for x in parts.values()) / _band_power(parts[voice], None))


def _peak_hz(x: np.ndarray, low: float, high: float) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)), 8 * len(x)))
    f = np.fft.rfftfreq(8 * len(x), 1 / SR)
    inside = (f >= low) & (f <= high)
    return float(f[inside][np.argmax(spectrum[inside])])


def off_pitch_cents(scenario: dict, render: str, voice: str, spec: dict) -> float:
    """Distance of the voice's first note from its written pitch."""
    written = scenario[render][voice + "Hz"]
    first = stems(scenario, render)[voice][: int(scenario[render]["seconds"] * SR / len(written))]
    return abs(1200 * float(np.log2(_peak_hz(first, written[0] * 0.9, written[0] * 1.1) / written[0])))


def entry_seconds(scenario: dict, render: str, voice: str, spec: dict) -> float:
    x = np.abs(stems(scenario, render)[voice])
    return float(np.argmax(x > 0.05 * np.max(x)) / SR)


def drift_semitones(scenario: dict, render: str) -> float:
    """Last melody note against the first, measured from the rendered melody."""
    p = scenario[render]
    melody = stems(scenario, render)["melody"]
    n = len(melody) // p["notes"]
    first, last = _peak_hz(melody[:n], 100, 2000), _peak_hz(melody[-n:], 100, 2000)
    return 12 * float(np.log2(last / first))


MEASURES = {
    f.__name__: f
    for f in (covered_db, presence_db, beat_depth, gain_db, below_loudest_db, off_pitch_cents, entry_seconds)
}


def voices(scenario: dict) -> list[str]:
    return [option.split(":")[1] for option in scenario["options"] if option != NO_CHANGE]


def construction_failures(scenario: dict) -> list[str]:
    """Empty when both renders sound the way the recipe and answer key say they do."""
    spec = scenario["construction"]
    failures = []
    if spec["measure"] == "drift_semitones":
        for render in ("defect", "control"):
            value, (low, high) = drift_semitones(scenario, render), spec[render]
            if (low is not None and value < low) or (high is not None and value > high):
                failures.append(f"{render}: drift {value:.2f} outside {spec[render]}")
        return failures
    measure = MEASURES[spec["measure"]]
    target = scenario["answer"]["defect"].split(":")[1]
    for render in ("defect", "control"):
        for voice in voices(scenario):
            value = measure(scenario, render, voice, spec)
            if render == "defect" and voice == target:
                if value < spec["bad"]:
                    failures.append(f"defect/{voice}: {value:.2f} < bad {spec['bad']}")
            elif value > spec["ok"]:
                failures.append(f"{render}/{voice}: {value:.2f} > ok {spec['ok']}")
    return failures
