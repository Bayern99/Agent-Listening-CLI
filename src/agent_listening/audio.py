"""Audio identity, transport derivation and deterministic measurements.

Imports: evidence, numpy, soundfile. Originals are only ever opened read-only;
the sole writer is _write_derived for content-addressed cache files.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf

from agent_listening.evidence import AudioFormat, Input, Measurement, Transport

METHOD = "agent-listening.audio/1"

_HASH_BLOCK = 1 << 20
_PCM_SUBTYPES = frozenset({"PCM_S8", "PCM_U8", "PCM_16", "PCM_24", "PCM_32"})

_SPECTRAL_WINDOW_S = 0.2
_SPECTRAL_HOP_S = 0.1
_SPECTRAL_BAND_MIN_HZ = 60.0
_SPECTRAL_BAND_MAX_HZ = 16000.0
_SPECTRAL_BANDS_PER_OCTAVE = 12
_SPECTRAL_DYNAMIC_DB = 60.0
_SPECTRAL_CAP_DB = 30.0
_SPECTRAL_SILENCE_DBFS = -60.0
_SPECTRAL_THRESHOLD_DB = 3.0


class IdentityError(Exception):
    """A source file no longer matches its recorded identity."""


class AudioError(Exception):
    """The file cannot be decoded as audio."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _pcm_stream(subtype: str) -> str:
    return "int32" if subtype in _PCM_SUBTYPES else "float32"


def _pcm_sha256(path: Path, subtype: str, sample_rate: int, channels: int) -> str:
    dtype = _pcm_stream(subtype)
    digest = hashlib.sha256(f"{sample_rate}:{channels}:{dtype}\n".encode())
    with sf.SoundFile(path) as fh:
        while True:
            block = fh.read(1 << 17, dtype=dtype, always_2d=True)
            if block.size == 0:
                break
            digest.update(np.ascontiguousarray(block).tobytes())
    return digest.hexdigest()


def identify(path: str | Path, label: str, role: str) -> Input:
    """Identify an audio file: byte hash, decoded-PCM hash and format facts."""
    p = Path(path)
    if not p.is_file():
        raise AudioError(f"not a readable audio file: {p}")
    try:
        info = sf.info(str(p), verbose=False)
    except (sf.LibsndfileError, RuntimeError, TypeError) as exc:
        raise AudioError(f"not a readable audio file: {p}") from exc
    sha = _file_sha256(p)
    frames = int(info.frames)
    duration_s = frames / info.samplerate if info.samplerate else 0.0
    fmt = AudioFormat(
        container=str(info.format),
        subtype=str(info.subtype),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=frames,
        duration_s=duration_s,
    )
    pcm = _pcm_sha256(p, fmt.subtype, fmt.sample_rate, fmt.channels)
    return Input(label=label, role=role, path=str(p), sha256=sha, pcm_sha256=pcm, format=fmt)


def _derive_subtype(subtype: str) -> str:
    return "PCM_16" if subtype in {"PCM_S8", "PCM_U8", "PCM_16"} else "PCM_24"


def _write_derived(
    target: Path,
    source: Path,
    start_frame: int,
    frames: int,
    sample_rate: int,
    channels: int,
    subtype: str,
    cache_dir: Path,
) -> None:
    """Write an excerpt/whole copy atomically; the only audio writer."""
    cache_root = cache_dir.resolve()
    resolved = target.resolve()
    if cache_root not in resolved.parents:
        raise IdentityError(f"derived target outside cache dir: {target}")
    if resolved == source.resolve():
        raise IdentityError(f"derived target equals source: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with (
            sf.SoundFile(str(source)) as src,
            sf.SoundFile(
                str(tmp), "w", samplerate=sample_rate, channels=channels, subtype=subtype, format="FLAC"
            ) as dst,
        ):
            src.seek(start_frame)
            remaining = frames
            while remaining > 0:
                block = src.read(min(remaining, 1 << 17), dtype="float64", always_2d=True)
                if block.size == 0:
                    break
                dst.write(block)
                remaining -= int(block.shape[0])
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def transport(inp: Input, cache_dir: str | Path, focus: tuple[float, float] | None) -> Transport:
    """Produce the file the model hears: whole file or excerpt, lossless FLAC."""
    cache = Path(cache_dir)
    source = Path(inp.path)
    if _file_sha256(source) != inp.sha256:
        raise IdentityError(f"input {inp.label} changed since identification: {source}")
    sr = inp.format.sample_rate
    start_frame = 0
    frames = inp.format.frames
    operation = "full"
    offset_s = 0.0
    if focus is not None:
        start, end = focus
        start_frame = round(start * sr)
        end_frame = min(round(end * sr), inp.format.frames)
        start_frame = min(start_frame, end_frame)
        frames = max(0, end_frame - start_frame)
        operation = "excerpt"
        offset_s = start_frame / sr
        if frames <= 0:
            raise AudioError(f"focus yields empty excerpt on {inp.label}")
    subtype = _derive_subtype(inp.format.subtype)
    param_key = f"{operation}:{start_frame}:{frames}:{subtype}".encode()
    param_hash = hashlib.sha256(param_key).hexdigest()[:16]
    target = cache / "derived" / f"{inp.sha256}-{operation}-{param_hash}.flac"
    _write_derived(target, source, start_frame, frames, sr, inp.format.channels, subtype, cache)
    derived_sha = _file_sha256(target)
    return Transport(
        input=inp.label,
        path=str(target),
        sha256=derived_sha,
        operation=operation,
        offset_s=offset_s,
        duration_s=frames / sr,
        sample_rate=sr,
        channels=inp.format.channels,
    )


def pcm_identity(a: Input, b: Input) -> Measurement:
    return Measurement(
        id="pcm-identity",
        kind="pcm-identity",
        inputs=(a.label, b.label),
        method=METHOD,
        params={},
        result={"identical": a.pcm_sha256 == b.pcm_sha256},
    )


def _mono(path: Path, start_frame: int, frames: int) -> npt.NDArray[Any]:
    with sf.SoundFile(str(path)) as fh:
        fh.seek(start_frame)
        data = fh.read(frames, dtype="float64", always_2d=True)
    if data.size == 0:
        return np.zeros(0)
    return np.asarray(data.mean(axis=1), dtype=np.float64)


def _decimate(mono: npt.NDArray[Any], factor: int) -> npt.NDArray[Any]:
    if factor <= 1:
        return mono
    usable = mono[: mono.size - mono.size % factor]
    return np.asarray(usable.reshape(-1, factor).mean(axis=1))


def _xcorr(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """FFT cross-correlation c[k] = sum_n a[n+k] * b[n], k = 0..size(a)+size(b)-2."""
    n = a.size + b.size - 1
    nfft = 1 << max(0, (n - 1).bit_length()) if n > 1 else 1
    return np.asarray(np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft))


def _estimate_lag(
    a_path: Path, b_path: Path, sr: int, a_frames: int, b_frames: int, max_lag_s: float
) -> tuple[int, str]:
    factor = max(1, round(sr / 2000))
    head = int(min(a_frames, b_frames, 60 * sr))
    a_dec = _decimate(_mono(a_path, 0, head), factor)
    b_dec = _decimate(_mono(b_path, 0, head), factor)
    max_lag = int(max_lag_s * sr / factor)
    if a_dec.size <= max_lag or b_dec.size <= max_lag or a_dec.size == 0:
        return 0, "low"
    corr = _xcorr(a_dec - a_dec.mean(), b_dec - b_dec.mean())
    window = np.concatenate([corr[corr.size - max_lag :], corr[: max_lag + 1]])
    norm = float(np.sqrt(np.sum(a_dec**2) * np.sum(b_dec**2)))
    if norm <= 0:
        return 0, "low"
    peak_idx = int(np.argmax(np.abs(window))) - max_lag
    peak = float(window[peak_idx + max_lag] / norm)
    if abs(peak) < 0.5:
        return 0, "low"
    coarse = -(peak_idx * factor)  # lag > 0 means B events occur lag seconds later than in A

    span = 4 * factor
    win_s = 2.0
    coarse_env = _decimate(np.abs(_mono(a_path, 0, head)), max(1, sr // 100))
    anchor = int(np.argmax(coarse_env) * max(1, sr // 100))
    start = max(0, min(anchor - int(win_s * sr / 2), head - int(win_s * sr)))
    n = int(win_s * sr)
    a_seg = _mono(a_path, start, n)
    b_lo = start + coarse - span
    if b_lo < 0 or b_lo + n + 2 * span > b_frames or a_seg.size == 0:
        return coarse, "high"
    b_seg = _mono(b_path, b_lo, n + 2 * span)
    corr2 = _xcorr(b_seg - b_seg.mean(), a_seg - a_seg.mean())[: 2 * span + 1]
    norm2 = float(np.sqrt(np.sum(a_seg**2) * np.sum(b_seg**2)))
    if norm2 <= 0 or corr2.size == 0:
        return coarse, "high"
    j = int(np.argmax(np.abs(corr2)))
    lag = coarse - span + j
    confidence = "high" if abs(float(corr2[j]) / norm2) >= 0.5 else "low"
    return lag, confidence


def _window_ratios(
    a_path: Path, b_path: Path, a_start: int, b_start: int, frames: int, sr: int, same_channels: bool
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Per-window residual ratio in dB and a mask of windows with signal."""
    win = int(0.1 * sr)
    hop = int(0.05 * sr)
    n_win = max(0, (frames - win) // hop + 1)
    ratio = np.zeros(n_win)
    alive = np.zeros(n_win, dtype=bool)
    floor_per_sample = 10 ** (-90 / 10)
    block = int(10 * sr)
    for base in range(0, n_win, max(1, (block - win) // hop)):
        win_lo = base * hop
        win_hi = min(frames, (base + max(1, (block - win) // hop) - 1) * hop + win)
        span = win_hi - win_lo
        a = _read_frames(a_path, a_start + win_lo, span)
        b = _read_frames(b_path, b_start + win_lo, span)
        if not same_channels:
            a = a.mean(axis=1, keepdims=True)
            b = b.mean(axis=1, keepdims=True)
        diff2 = (a - b) ** 2
        a2 = a**2
        b2 = b**2
        count = (span - win) // hop + 1
        for w in range(count):
            seg = slice(w * hop, w * hop + win)
            num = float(diff2[seg].sum())
            den = max(float(a2[seg].sum()), float(b2[seg].sum()), win * floor_per_sample * a.shape[1])
            if (
                float(a2[seg].sum()) <= win * floor_per_sample * a.shape[1]
                and float(b2[seg].sum()) <= win * floor_per_sample * b.shape[1]
            ):
                continue
            ratio[base + w] = 10 * np.log10(max(num, 1e-30) / den)
            alive[base + w] = True
    return ratio, alive


def _read_frames(path: Path, start: int, frames: int) -> npt.NDArray[Any]:
    with sf.SoundFile(str(path)) as fh:
        fh.seek(max(0, start))
        return np.asarray(fh.read(frames, dtype="float64", always_2d=True), dtype=np.float64)


def _spectral_distances(
    a_path: Path, b_path: Path, a_start: int, b_start: int, frames: int, sr: int
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Phase-invariant per-window spectral distance in dB over a 1/12-octave grid."""
    win = int(_SPECTRAL_WINDOW_S * sr)
    hop = int(_SPECTRAL_HOP_S * sr)
    n_win = max(0, (frames - win) // hop + 1)
    distance = np.zeros(n_win)
    alive = np.zeros(n_win, dtype=bool)
    if n_win == 0:
        return distance, alive
    fmax = min(_SPECTRAL_BAND_MAX_HZ, sr / 2)
    n_edges = int(_SPECTRAL_BANDS_PER_OCTAVE * np.log2(fmax / _SPECTRAL_BAND_MIN_HZ)) + 1
    edge_hz = _SPECTRAL_BAND_MIN_HZ * 2.0 ** (np.arange(n_edges + 1) / _SPECTRAL_BANDS_PER_OCTAVE)
    edge_hz = edge_hz[edge_hz <= fmax]
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    edge_bins = np.concatenate([np.searchsorted(freqs, edge_hz), [np.searchsorted(freqs, fmax)]])
    hann = np.hanning(win)
    silence_rms = 10 ** (_SPECTRAL_SILENCE_DBFS / 20)
    block = int(10 * sr)
    per_block = max(1, (block - win) // hop)
    for base in range(0, n_win, per_block):
        win_lo = base * hop
        win_hi = min(frames, (base + per_block - 1) * hop + win)
        span = win_hi - win_lo
        a = _read_frames(a_path, a_start + win_lo, span).mean(axis=1)
        b = _read_frames(b_path, b_start + win_lo, span).mean(axis=1)
        count = (span - win) // hop + 1
        for w in range(count):
            seg = slice(w * hop, w * hop + win)
            xa, xb = a[seg], b[seg]
            if float(np.sqrt(np.mean(xa**2))) < silence_rms and float(np.sqrt(np.mean(xb**2))) < silence_rms:
                continue
            alive[base + w] = True
            power_a = np.abs(np.fft.rfft(xa * hann)) ** 2
            power_b = np.abs(np.fft.rfft(xb * hann)) ** 2
            cum_a = np.concatenate(([0.0], np.cumsum(power_a)))
            cum_b = np.concatenate(([0.0], np.cumsum(power_b)))
            level_a = 10 * np.log10(np.maximum(cum_a[edge_bins[1:]] - cum_a[edge_bins[:-1]], 1e-30))
            level_b = 10 * np.log10(np.maximum(cum_b[edge_bins[1:]] - cum_b[edge_bins[:-1]], 1e-30))
            top = max(float(level_a.max()), float(level_b.max()))
            considered = np.maximum(level_a, level_b) > top - _SPECTRAL_DYNAMIC_DB
            deltas = np.minimum(np.abs(level_a - level_b), _SPECTRAL_CAP_DB)[considered]
            distance[base + w] = float(deltas.mean()) if deltas.size else 0.0
    return distance, alive


def _spectral_result(
    a_path: Path, b_path: Path, a_start: int, b_start: int, frames: int, sr: int
) -> dict[str, object]:
    """Phase-invariant spectral difference regions in A's time base."""
    distance, alive = _spectral_distances(a_path, b_path, a_start, b_start, frames, sr)
    marked = (distance > _SPECTRAL_THRESHOLD_DB) & alive
    raw = _merge_regions(
        marked,
        sr,
        _SPECTRAL_HOP_S,
        0.25,
        distance,
        offset_s=a_start / sr,
        win_s=_SPECTRAL_WINDOW_S,
    )
    regions = [{"start": round(r[0], 3), "end": round(r[1], 3), "peak_db": round(r[2], 1)} for r in raw]
    live = int(alive.sum())
    changed = int(marked.sum())
    changed_fraction = changed / live if live else 0.0
    return {
        "available": True,
        "window_s": _SPECTRAL_WINDOW_S,
        "hop_s": _SPECTRAL_HOP_S,
        "grid": "1/12-octave 60 Hz-16 kHz",
        "threshold_db": _SPECTRAL_THRESHOLD_DB,
        "regions": regions,
        "changed_fraction": round(changed_fraction, 3),
        "max_distance_db": round(float(distance.max()) if distance.size else 0.0, 1),
        "global_change": changed_fraction > 0.8,
    }


def _merge_regions(
    masked: npt.NDArray[Any],
    sr: int,
    hop_s: float,
    merge_gap_s: float,
    ratio: npt.NDArray[Any],
    offset_s: float = 0.0,
    win_s: float = 0.1,
) -> list[tuple[float, float, float]]:
    regions: list[tuple[float, float, float]] = []
    gap_limit = round(merge_gap_s / hop_s)
    current: list[int] | None = None
    last_idx = -1
    for idx in np.nonzero(masked)[0]:
        if current is not None and idx - last_idx <= gap_limit + 1:
            current.append(int(idx))
        else:
            if current:
                regions.append(_region_span(current, sr, hop_s, ratio, offset_s, win_s))
            current = [int(idx)]
        last_idx = int(idx)
    if current:
        regions.append(_region_span(current, sr, hop_s, ratio, offset_s, win_s))
    return regions


def _region_span(
    indices: Sequence[int],
    sr: int,
    hop_s: float,
    ratio: npt.NDArray[Any],
    offset_s: float,
    win_s: float,
) -> tuple[float, float, float]:
    start = offset_s + indices[0] * hop_s
    end = offset_s + indices[-1] * hop_s + win_s
    peak = float(ratio[indices].max())
    return start, end, peak


def signal_difference(a: Input, b: Input) -> Measurement:
    """Measured difference regions between two same-rate inputs."""
    params = {
        "window_s": 0.1,
        "hop_s": 0.05,
        "threshold_db": -30,
        "max_lag_s": 1.0,
        "merge_gap_s": 0.25,
        "silence_dbfs": -90,
    }
    if a.format.sample_rate != b.format.sample_rate:
        return Measurement(
            id="signal-difference",
            kind="signal-difference",
            inputs=(a.label, b.label),
            method=METHOD,
            params=params,
            result={
                "available": False,
                "reason": f"sample rates differ ({a.format.sample_rate} vs {b.format.sample_rate})",
                "spectral": {"available": False},
            },
        )
    sr = a.format.sample_rate
    lag, lag_confidence = _estimate_lag(
        Path(a.path), Path(b.path), sr, a.format.frames, b.format.frames, params["max_lag_s"]
    )

    a_start, b_start = 0, lag
    if lag < 0:
        a_start, b_start = -lag, 0
    frames = min(a.format.frames - a_start, b.format.frames - b_start)
    same_channels = a.format.channels == b.format.channels
    ratio, alive = _window_ratios(Path(a.path), Path(b.path), a_start, b_start, frames, sr, same_channels)
    hop_s = float(params["hop_s"])
    marked = (ratio > float(params["threshold_db"])) & alive
    raw_regions = _merge_regions(
        marked, sr, hop_s, float(params["merge_gap_s"]), ratio, offset_s=a_start / sr
    )
    regions: list[dict[str, float | str]] = [
        {"start": round(r[0], 3), "end": round(r[1], 3), "peak_db": round(r[2], 1)} for r in raw_regions
    ]

    a_tail = a.format.frames - (a_start + frames)
    b_tail = b.format.frames - (b_start + frames)
    if a_tail / sr > 0.1:
        regions.append(
            {
                "start": round((a_start + frames) / sr, 3),
                "end": round((a_start + frames + a_tail) / sr, 3),
                "peak_db": 0.0,
                "only_in": a.label,
            }
        )
    elif b_tail / sr > 0.1:
        regions.append(
            {
                "start": round((a_start + frames) / sr, 3),
                "end": round((a_start + frames + b_tail) / sr, 3),
                "peak_db": 0.0,
                "only_in": b.label,
            }
        )
    regions.sort(key=lambda r: float(r["start"]))
    while len(regions) > 200:
        gaps = [
            (float(regions[i + 1]["start"]) - float(regions[i]["end"]), i) for i in range(len(regions) - 1)
        ]
        _, i = min(gaps)
        merged: dict[str, float | str] = {
            "start": regions[i]["start"],
            "end": regions[i + 1]["end"],
            "peak_db": max(float(regions[i]["peak_db"]), float(regions[i + 1]["peak_db"])),
        }
        if "only_in" in regions[i] and regions[i].get("only_in") == regions[i + 1].get("only_in"):
            merged["only_in"] = regions[i]["only_in"]
        regions[i : i + 2] = [merged]

    live = int(alive.sum())
    changed = int(marked.sum())
    changed_fraction = changed / live if live else 0.0
    max_ratio = float(ratio.max()) if ratio.size else 0.0
    spectral = _spectral_result(Path(a.path), Path(b.path), a_start, b_start, frames, sr)
    return Measurement(
        id="signal-difference",
        kind="signal-difference",
        inputs=(a.label, b.label),
        method=METHOD,
        params=params,
        result={
            "available": True,
            "lag_s": round(lag / sr, 6),
            "lag_confidence": lag_confidence,
            "regions": regions,
            "changed_fraction": round(changed_fraction, 3),
            "max_ratio_db": round(max_ratio, 1),
            "global_change": changed_fraction > 0.8,
            "spectral": spectral,
        },
    )


def _db(x: float) -> float:
    return float(10 * np.log10(max(x, 1e-30)))


def _focus_frames(inp: Input, focus: tuple[float, float] | None) -> tuple[int, int]:
    sr = inp.format.sample_rate
    if focus is None:
        return 0, inp.format.frames
    start = round(focus[0] * sr)
    end = min(round(focus[1] * sr), inp.format.frames)
    return start, max(0, end - start)


def _cap_series(series: list[dict[str, object]], limit: int = 400) -> list[dict[str, object]]:
    if len(series) <= limit:
        return series
    step = (len(series) - 1) / (limit - 1)
    return [series[round(i * step)] for i in range(limit)]


def _level(data: npt.NDArray[Any], sr: int, base_s: float) -> dict[str, object]:
    win = int(0.5 * sr)
    series: list[dict[str, object]] = []
    sq = data**2
    for start in range(0, data.shape[0] - win + 1, win):
        seg = data[start : start + win]
        rms = float(np.sqrt(sq[start : start + win].mean()))
        peak = float(np.abs(seg).max())
        series.append(
            {
                "t": round(base_s + start / sr, 3),
                "rms_dbfs": round(20 * np.log10(max(rms, 1e-10)), 1),
                "peak_dbfs": round(20 * np.log10(max(peak, 1e-10)), 1),
            }
        )
    rms_all = float(np.sqrt(sq.mean())) if sq.size else 0.0
    peak_all = float(np.abs(data).max()) if data.size else 0.0
    return {
        "windows": _cap_series(series),
        "rms_dbfs": round(20 * np.log10(max(rms_all, 1e-10)), 1),
        "peak_dbfs": round(20 * np.log10(max(peak_all, 1e-10)), 1),
        "crest_db": round(20 * np.log10(max(peak_all, 1e-10)) - 20 * np.log10(max(rms_all, 1e-10)), 1),
    }


def _spectrum(data: npt.NDArray[Any], sr: int, base_s: float) -> dict[str, object]:
    mono = data.mean(axis=1)
    win = int(0.5 * sr)
    nfft = 2048
    hop = 1024
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    window = np.hanning(nfft)

    def frame_mags(seg: npt.NDArray[Any]) -> npt.NDArray[Any]:
        count = max(0, (seg.size - nfft) // hop + 1)
        if count == 0:
            return np.zeros((0, freqs.size))
        out = np.empty((count, freqs.size))
        for i in range(count):
            out[i] = np.abs(np.fft.rfft(seg[i * hop : i * hop + nfft] * window))
        return out

    def centroid_rolloff(mags: npt.NDArray[Any]) -> tuple[float, float]:
        if mags.shape[0] == 0:
            return 0.0, 0.0
        power = mags**2
        total = power.sum(axis=1)
        total[total <= 0] = 1e-30
        centroid = float(((power * freqs).sum(axis=1) / total).mean())
        cumsum = np.cumsum(power, axis=1) / total[:, None]
        rolloff_idx = np.argmax(cumsum >= 0.85, axis=1)
        rolloff = float(freqs[rolloff_idx].mean())
        return centroid, rolloff

    series: list[dict[str, object]] = []
    for start in range(0, mono.size - win + 1, win):
        c, r = centroid_rolloff(frame_mags(mono[start : start + win]))
        series.append(
            {"t": round(base_s + start / sr, 3), "centroid_hz": round(c, 1), "rolloff85_hz": round(r, 1)}
        )
    all_mags = frame_mags(mono)
    total_power = float((all_mags**2).sum()) or 1e-30
    bands: dict[str, float] = {}
    edges = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    for i, center in enumerate(edges):
        lo = center / np.sqrt(2)
        hi = (edges[i + 1] / np.sqrt(2)) if i + 1 < len(edges) else min(center * np.sqrt(2), sr / 2)
        band_power = float((all_mags[:, (freqs >= lo) & (freqs < hi)] ** 2).sum())
        bands[str(center)] = round(10 * np.log10(max(band_power, 1e-30) / total_power), 1)
    return {"windows": _cap_series(series), "octave_bands_db": bands}


def _stereo(data: npt.NDArray[Any], sr: int, base_s: float) -> dict[str, object]:
    if data.shape[1] < 2:
        return {"available": False, "reason": "mono input"}
    left, right = data[:, 0], data[:, 1]
    win = int(0.5 * sr)
    series: list[dict[str, object]] = []
    for start in range(0, data.shape[0] - win + 1, win):
        lef, rig = left[start : start + win], right[start : start + win]
        denom = float(np.sqrt((lef**2).sum() * (rig**2).sum()))
        corr = float((lef * rig).sum() / denom) if denom > 0 else 0.0
        mid = (lef + rig) / 2
        side = (lef - rig) / 2
        sm = float((side**2).sum() / max((mid**2).sum(), 1e-30))
        bal = float((lef**2).sum() / max((rig**2).sum(), 1e-30))
        series.append(
            {
                "t": round(base_s + start / sr, 3),
                "corr": round(corr, 3),
                "side_mid_db": round(10 * np.log10(max(sm, 1e-30)), 1),
                "lr_balance_db": round(10 * np.log10(max(bal, 1e-30)), 1),
            }
        )
    return {"windows": _cap_series(series)}


def _onsets(data: npt.NDArray[Any], sr: int, base_s: float) -> dict[str, object]:
    mono = data.mean(axis=1)
    nfft, hop = 2048, 512
    if mono.size < nfft:
        return {"count": 0, "onsets": []}
    count = (mono.size - nfft) // hop + 1
    window = np.hanning(nfft)
    flux = np.zeros(count)
    prev = np.abs(np.fft.rfft(mono[:nfft] * window))
    for i in range(1, count):
        mag = np.abs(np.fft.rfft(mono[i * hop : i * hop + nfft] * window))
        flux[i] = float(np.maximum(mag - prev, 0).sum())
        prev = mag
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median)))
    threshold = median + 8 * mad if mad > 0 else median + 1e-6
    min_gap_s = 0.05
    picks: list[tuple[float, float]] = []
    order = np.argsort(flux)[::-1]
    for idx in order:
        if len(picks) >= 200 or flux[idx] <= threshold:
            break
        t = (idx * hop + nfft // 2) / sr
        if any(abs(t - p[0]) < min_gap_s for p in picks):
            continue
        picks.append((t, float(flux[idx])))
    picks.sort()
    strength_max = float(flux.max()) or 1.0
    return {
        "count": len(picks),
        "onsets": [{"t": round(base_s + t, 3), "strength": round(s / strength_max, 3)} for t, s in picks],
    }


_DESCRIBE_KINDS = ("level", "spectrum", "stereo", "onsets")


def describe(inp: Input, focus: tuple[float, float] | None, kinds: Sequence[str]) -> list[Measurement]:
    """Deterministic measurements over a focus span (original time base)."""
    start_frame, frames = _focus_frames(inp, focus)
    data = _read_frames(Path(inp.path), start_frame, frames)
    base_s = start_frame / inp.format.sample_rate
    out = []
    for kind in kinds:
        if kind not in _DESCRIBE_KINDS:
            raise AudioError(f"unknown measure kind: {kind}")
        if kind == "level":
            result = _level(data, inp.format.sample_rate, base_s)
        elif kind == "spectrum":
            result = _spectrum(data, inp.format.sample_rate, base_s)
        elif kind == "stereo":
            result = _stereo(data, inp.format.sample_rate, base_s)
        else:
            result = _onsets(data, inp.format.sample_rate, base_s)
        out.append(
            Measurement(
                id=kind,
                kind=kind,
                inputs=(inp.label,),
                method=METHOD,
                params={
                    "focus": [
                        round(start_frame / inp.format.sample_rate, 3),
                        round(frames / inp.format.sample_rate, 3),
                    ]
                    if focus is not None
                    else None
                },
                result=result,
            )
        )
    return out
