# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "soundfile>=0.12,<0.14", "scipy>=1.13,<2"]
# ///
"""Build every evaluation input from hash-verified originals into $AGENT_LISTENING_EVAL_DIR/cases."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).parent))
import corpus

EVAL_DIR = Path(os.environ.get("AGENT_LISTENING_EVAL_DIR", Path.home() / ".cache/agent-listening-eval"))
SOURCES = EVAL_DIR / "sources"
CASES = EVAL_DIR / "cases"
SC_DIR = Path(__file__).parent / "sc"
SCLANG = os.environ.get("SCLANG", "/Applications/SuperCollider.app/Contents/MacOS/sclang")
P = corpus.EDIT_PARAMS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources() -> None:
    expected = json.loads((Path(__file__).parent / "source_hashes.json").read_text())
    bad = [name for name, digest in expected.items() if sha256(SOURCES / name) != digest]
    if bad:
        raise SystemExit(f"source hash mismatch: {bad}")


def recipe_key(recipe: dict) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]


def window(n: int, sr: int, start: float, end: float) -> np.ndarray:
    w = np.zeros(n)
    fade = int(P["crossfade_s"] * sr)
    a, b = int(start * sr), min(int(end * sr), n)
    w[a:b] = 1.0
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
    w[a : a + fade] = ramp[: len(w[a : a + fade])]
    w[max(b - fade, 0) : b] = ramp[::-1][-len(w[max(b - fade, 0) : b]) :]
    return w


def splice(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w[:, None] if x.ndim == 2 else w
    return x * (1 - w) + y * w


def pitch_shift(x: np.ndarray, sr: int, cents: float) -> np.ndarray:
    ratio = 2 ** (cents / 1200)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        sf.write(src, x, sr, subtype="FLOAT")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(src),
                "-af",
                f"asetrate={sr * ratio:.6f},aresample={sr},atempo={1 / ratio:.8f}",
                "-c:a",
                "pcm_f32le",
                str(dst),
            ],
            check=True,
        )
        y, _ = sf.read(dst, always_2d=x.ndim == 2)
    y = y[: len(x)]
    return np.pad(y, [(0, len(x) - len(y))] + ([(0, 0)] if x.ndim == 2 else []))


def peaking(x: np.ndarray, sr: int, f0: float, q: float, gain_db: float) -> np.ndarray:
    a_gain = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * a_gain, -2 * np.cos(w0), 1 - alpha * a_gain])
    a = np.array([1 + alpha / a_gain, -2 * np.cos(w0), 1 - alpha / a_gain])
    return lfilter(b / a[0], a / a[0], x, axis=0)


def noise(x: np.ndarray, seed_text: str) -> np.ndarray:
    rng = np.random.default_rng(int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16))
    return x + rng.standard_normal(x.shape) * 10 ** (P["near_null_dbfs"] / 20)


def stems(piece: str) -> tuple[dict[str, np.ndarray], int]:
    out, sr = {}, 0
    for inst in corpus.INSTRUMENTS:
        out[inst], sr = sf.read(SOURCES / "bach10" / piece / f"{piece}-{inst}.wav")
    return out, sr


def notes(piece: str) -> np.ndarray:
    return np.loadtxt(SOURCES / "bach10" / piece / f"{piece}.txt")


def edit_stem(x: np.ndarray, sr: int, recipe: dict) -> np.ndarray:
    start, end = recipe["span"]
    w = window(len(x), sr, start, end)
    edit = recipe["edit"]
    if edit == "mute":
        return x * (1 - w)
    if edit == "timing":
        d = int(P["timing"]["delay_s"] * sr)
        return splice(x, np.concatenate([np.zeros(d), x[:-d]]), w)
    if edit == "pitch":
        return splice(x, pitch_shift(x, sr, P["pitch"]["cents"]), w)
    if edit == "resonance":
        r = P["resonance"]
        return splice(x, peaking(x, sr, r["f0_hz"], r["q"], r["gain_db"]), w)
    if edit == "articulation":
        voice = corpus.INSTRUMENTS.index(recipe["instrument"]) + 1
        onsets = sorted(row[0] / 1000 for row in notes(recipe["piece"]) if int(row[3]) == voice)
        gate = np.ones(len(x))
        fade = int(0.02 * sr)
        for onset, following in zip(onsets, [*onsets[1:], onsets[-1] + 1.0], strict=True):
            if not start <= onset < end:
                continue
            keep = max(
                (following - onset) * P["articulation"]["keep_fraction"], P["articulation"]["min_keep_s"]
            )
            a, b = int((onset + keep) * sr), int(min(following, end) * sr)
            if b - a <= fade:
                continue
            gate[a : a + fade] = np.minimum(gate[a : a + fade], np.linspace(1, 0, fade))
            gate[a + fade : b] = 0
        return x * gate
    raise ValueError(edit)


def render_sc(template: str, params: dict, out: Path) -> None:
    literal = (
        "("
        + ", ".join(f"{k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in params.items())
        + ")"
    )
    source = (
        (SC_DIR / template).read_text().replace("PARAMS", literal).replace("OUTPATH", json.dumps(str(out)))
    )
    scd = out.with_suffix(".scd")
    scd.write_text(source)
    subprocess.run([SCLANG, str(scd)], check=True, timeout=300, capture_output=True)
    if not out.exists():
        raise RuntimeError(f"sclang produced no output for {scd}")


def decode(source: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(SOURCES / source, always_2d=True)
    return x, sr


def write(path: Path, x: np.ndarray, sr: int, fmt: str = "WAV") -> None:
    sf.write(path, x, sr, subtype="PCM_24", format=fmt)


def build(recipe: dict) -> Path:
    op = recipe["op"]
    ext = ".flac" if op == "bach10-identical-flac" else ".wav"
    out = CASES / f"{op}-{recipe_key(recipe)}{ext}"
    if out.exists():
        return out
    gain = P["mix_gain"]
    if op in ("bach10-mix", "bach10-edit", "bach10-nearnull", "bach10-identical-flac"):
        parts, sr = stems(recipe["piece"])
        if op == "bach10-edit":
            parts[recipe["instrument"]] = edit_stem(parts[recipe["instrument"]], sr, recipe)
        mix = sum(parts.values()) * gain
        if op == "bach10-nearnull":
            mix = noise(mix, recipe["piece"])
        write(out, mix, sr, "FLAC" if ext == ".flac" else "WAV")
    elif op == "decode":
        x, sr = decode(recipe["source"])
        write(out, x, sr)
    elif op in ("span-pitch", "span-gain", "span-resonance", "near-null"):
        x, sr = decode(recipe["source"])
        if op == "near-null":
            y = noise(x, recipe["source"])
        else:
            w = window(len(x), sr, *recipe["span"])
            if op == "span-pitch":
                y = splice(x, pitch_shift(x, sr, recipe["cents"]), w)
            elif op == "span-gain":
                y = splice(x, x * 10 ** (recipe["gain_db"] / 20), w)
            else:
                y = splice(x, peaking(x, sr, recipe["f0_hz"], recipe["q"], recipe["gain_db"]), w)
        write(out, y, sr)
    elif op == "copy":
        shutil.copyfile(SOURCES / recipe["source"], out)
    elif op == "sc-canon":
        render_sc("canon.scd", recipe["params"], out)
    elif op == "sc-canon-nearnull":
        base = build({"op": "sc-canon", "params": recipe["params"]})
        x, sr = sf.read(base, always_2d=True)
        write(out, noise(x, "sc-canon"), sr)
    elif op == "sc-longform":
        render_sc("longform.scd", recipe["params"], out)
    else:
        raise ValueError(op)
    return out


def main() -> None:
    CASES.mkdir(parents=True, exist_ok=True)
    verify_sources()
    manifest = {}
    for case in corpus.cases():
        recipes = list(case["inputs"])
        if "misleading_context" in case:
            recipes.append(case["misleading_context"])
        for recipe in recipes:
            path = build(recipe)
            manifest[recipe_key(recipe)] = {"recipe": recipe, "path": str(path), "sha256": sha256(path)}
    (EVAL_DIR / "build-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"built {len(manifest)} inputs into {CASES}")


if __name__ == "__main__":
    main()
