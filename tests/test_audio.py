from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_listening import audio
from conftest import sine, write_wav

SR = 48000


def _id(path, label="A", role="subject"):
    return audio.identify(path, label, role)


def test_identify_sha_matches_hashlib(wav_factory):
    path = wav_factory(sine(SR, 0.5))
    inp = _id(path)
    assert inp.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert inp.format.sample_rate == SR
    assert inp.format.channels == 2


def test_identify_rejects_non_audio(tmp_path):
    junk = tmp_path / "junk.txt"
    junk.write_text("not audio at all")
    with pytest.raises(audio.AudioError):
        _id(junk)


@given(st.floats(min_value=200, max_value=2000), st.floats(min_value=0.2, max_value=1.0))
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_pcm_sha256_wav_flac_invariant(tmp_path, freq, seconds):
    data = (sine(SR, seconds, freq=freq) * 32767).astype(np.int16)
    wav = tmp_path / "a.wav"
    flac = tmp_path / "a.flac"
    sf.write(str(wav), data, SR, subtype="PCM_16", format="WAV")
    sf.write(str(flac), data, SR, subtype="PCM_16", format="FLAC")
    a, b = _id(wav), _id(flac)
    assert a.pcm_sha256 == b.pcm_sha256
    assert a.sha256 != b.sha256


def test_pcm_sha256_changes_on_one_sample(wav_factory, tmp_path):
    data = sine(SR, 0.5)
    path = wav_factory(data)
    data2 = data.copy()
    data2[1234, 0] = -data2[1234, 0]
    path2 = write_wav(tmp_path / "changed.wav", data2, SR)
    assert _id(path).pcm_sha256 != _id(path2).pcm_sha256


def test_transport_excerpt(wav_factory, tmp_path):
    data = sine(SR, 2.0)
    path = wav_factory(data)
    inp = _id(path)
    tr = audio.transport(inp, tmp_path / "cache", (0.5, 1.5))
    assert tr.operation == "excerpt"
    assert tr.offset_s == pytest.approx(0.5, abs=1e-4)
    assert tr.duration_s == pytest.approx(1.0, abs=1e-3)
    out = sf.read(tr.path, dtype="float64", always_2d=True)[0]
    ref = data[int(0.5 * SR) : int(1.5 * SR)]
    assert out.shape[0] == ref.shape[0]
    assert np.abs(out - ref).max() < 1e-3
    assert Path(tr.path).parent == (tmp_path / "cache" / "derived")
    assert tr.sha256 == hashlib.sha256(Path(tr.path).read_bytes()).hexdigest()


def test_transport_full_and_reuse(wav_factory, tmp_path):
    inp = _id(wav_factory(sine(SR, 0.5)))
    t1 = audio.transport(inp, tmp_path / "cache", None)
    t2 = audio.transport(inp, tmp_path / "cache", None)
    assert t1.path == t2.path and t1.sha256 == t2.sha256
    assert t1.operation == "full"


def test_transport_regenerates_tampered_cache(wav_factory, tmp_path):
    inp = _id(wav_factory(sine(SR, 0.5)))
    cache = tmp_path / "cache"
    t1 = audio.transport(inp, cache, None)
    Path(t1.path).write_bytes(b"tampered")
    t2 = audio.transport(inp, cache, None)
    assert t2.path == t1.path
    assert t2.sha256 == t1.sha256
    assert hashlib.sha256(Path(t2.path).read_bytes()).hexdigest() == t1.sha256


def test_transport_detects_modification(wav_factory, tmp_path):
    data = sine(SR, 1.0)
    path = wav_factory(data)
    inp = _id(path)
    write_wav(path, sine(SR, 1.0, freq=880.0), SR)
    with pytest.raises(audio.IdentityError):
        audio.transport(inp, tmp_path / "cache", None)


def test_write_derived_refuses_outside_cache(wav_factory, tmp_path):
    src = wav_factory(sine(SR, 0.5))
    with pytest.raises(audio.IdentityError):
        audio._write_derived(
            tmp_path / "elsewhere" / "out.flac", src, 0, 1000, SR, 2, "PCM_16", tmp_path / "cache"
        )


def test_write_derived_refuses_source_collision(wav_factory, tmp_path):
    src = wav_factory(sine(SR, 0.5))
    with pytest.raises(audio.IdentityError):
        audio._write_derived(src, src, 0, 1000, SR, 2, "PCM_16", src.parent)


def test_transport_refuses_target_collision(wav_factory, tmp_path):
    inp = _id(wav_factory(sine(SR, 0.5)))
    cache = tmp_path / "cache"
    cache.mkdir()
    blocker = cache / "derived"
    blocker.write_text("a directory should be here")
    with pytest.raises(OSError):
        audio.transport(inp, cache, None)


def test_source_untouched_after_transport(wav_factory, tmp_path):
    path = wav_factory(sine(SR, 0.5))
    before = path.read_bytes()
    inp = _id(path)
    audio.transport(inp, tmp_path / "cache", (0.1, 0.4))
    assert path.read_bytes() == before


def test_pcm_identity_measurement(wav_factory):
    data = sine(SR, 0.5)
    a = _id(wav_factory(data))
    b = _id(wav_factory(data.copy()))
    m = audio.pcm_identity(a, b)
    assert m.result["identical"] is True
    c = _id(wav_factory(sine(SR, 0.5, freq=880.0)))
    assert audio.pcm_identity(a, c).result["identical"] is False


def test_signal_difference_identical(wav_factory):
    data = sine(SR, 1.0)
    a = _id(wav_factory(data))
    b = _id(wav_factory(data.copy()))
    m = audio.signal_difference(a, b)
    assert m.result["available"] is True
    assert m.result["regions"] == []
    assert m.result["global_change"] is False


def test_signal_difference_low_noise(wav_factory):
    rng = np.random.default_rng(0)
    data = sine(SR, 2.0)
    noisy = data + rng.normal(0, 1e-6, data.shape)
    a = _id(wav_factory(data))
    b = _id(wav_factory(noisy))
    m = audio.signal_difference(a, b)
    assert m.result["regions"] == [], m.result


@given(st.floats(min_value=0.2, max_value=1.4))
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_signal_difference_localized_change(tmp_path, change_start):
    seconds = 2.0
    a_data = sine(SR, seconds, freq=440.0, amp=0.2)
    b_data = a_data.copy()
    lo, hi = int(change_start * SR), int((change_start + 0.5) * SR)
    b_data[lo:hi] *= 3.0
    a = _id(write_wav(tmp_path / "a.wav", a_data, SR))
    b = _id(write_wav(tmp_path / "b.wav", b_data, SR))
    m = audio.signal_difference(a, b)
    regions = m.result["regions"]
    assert regions, m.result
    assert any(
        r["start"] <= change_start + 0.15 and r["end"] >= change_start + 0.5 - 0.15 for r in regions
    ), regions


def test_signal_difference_delayed_copy(tmp_path):
    sr = SR
    rng = np.random.default_rng(1)
    # pure tones are periodic, so a delay is not identifiable; use noise-dominated signal
    a_data = 0.05 * sine(sr, 2.0, freq=330.0) + 0.25 * rng.standard_normal((int(2.0 * sr), 2))
    pad = np.zeros((int(0.3 * sr), 2))
    b_data = np.concatenate([pad, a_data])
    a = _id(write_wav(tmp_path / "a.wav", a_data, sr))
    b = _id(write_wav(tmp_path / "b.wav", b_data, sr))
    m = audio.signal_difference(a, b)
    assert m.result["lag_s"] == pytest.approx(0.3, abs=0.05), m.result
    assert m.result["regions"] == [], m.result


def test_signal_difference_advanced_b_regions_in_a_time(tmp_path):
    """B = A advanced by 0.3 s (negative lag): regions must be in A's time base."""
    sr = SR
    rng = np.random.default_rng(2)
    a_data = 0.05 * sine(sr, 2.0, freq=330.0) + 0.25 * rng.standard_normal((int(2.0 * sr), 2))
    b_data = a_data[int(0.3 * sr) :].copy()
    b_data[int(0.8 * sr) : int(1.3 * sr)] *= 3.0  # corresponds to A-time 1.1-1.6 s
    a = _id(write_wav(tmp_path / "a.wav", a_data, sr))
    b = _id(write_wav(tmp_path / "b.wav", b_data, sr))
    m = audio.signal_difference(a, b)
    assert m.result["lag_s"] == pytest.approx(-0.3, abs=0.05), m.result
    regions = m.result["regions"]
    assert any(r["start"] <= 1.25 and r["end"] >= 1.45 for r in regions), regions
    assert all("only_in" not in r or r["start"] > 0.2 for r in regions)


def test_signal_difference_residual_floor_in_near_silence(tmp_path):
    """-100 dBFS noise on a decaying tail is ratio-high but inaudible: no regions."""
    t = np.arange(int(4 * SR)) / SR
    env = 0.1 * np.exp(-t / 0.37)
    a_data = np.stack([env * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
    rng = np.random.default_rng(3)
    b_data = a_data + rng.normal(0, 1e-5, a_data.shape)
    a = _id(write_wav(tmp_path / "a.wav", a_data, SR))
    b = _id(write_wav(tmp_path / "b.wav", b_data, SR))
    m = audio.signal_difference(a, b)
    assert m.params["residual_floor_dbfs"] == -80
    assert m.result["regions"] == [], m.result["regions"]


def test_signal_difference_residual_floor_passes_real_change(tmp_path):
    """A residual around -40 dBFS is above the audibility floor and still marked."""
    t = np.arange(int(4 * SR)) / SR
    env = 0.1 * np.exp(-t / 0.37)
    a_data = np.stack([env * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
    rng = np.random.default_rng(4)
    b_data = a_data.copy()
    lo, hi = int(2.9 * SR), int(3.4 * SR)
    b_data[lo:hi] += rng.normal(0, 1e-2, (hi - lo, 2))
    a = _id(write_wav(tmp_path / "a.wav", a_data, SR))
    b = _id(write_wav(tmp_path / "b.wav", b_data, SR))
    m = audio.signal_difference(a, b)
    regions = m.result["regions"]
    assert any(r["start"] <= 3.0 and r["end"] >= 3.3 for r in regions), regions


def test_signal_difference_global_gain(tmp_path):
    a_data = sine(SR, 2.0)
    b_data = a_data * 2.0
    a = _id(write_wav(tmp_path / "a.wav", a_data, SR))
    b = _id(write_wav(tmp_path / "b.wav", b_data, SR))
    m = audio.signal_difference(a, b)
    assert m.result["global_change"] is True


def test_signal_difference_rate_mismatch(tmp_path):
    a = _id(write_wav(tmp_path / "a.wav", sine(SR, 0.5), SR))
    b = _id(write_wav(tmp_path / "b.wav", sine(44100, 0.5), 44100))
    m = audio.signal_difference(a, b)
    assert m.result["available"] is False


def test_describe_level_and_onsets(tmp_path):
    sr = SR
    clicks = np.zeros((sr,))
    for k in range(5):
        clicks[int(0.1 * sr + k * 0.2 * sr)] = 0.9
    data = np.stack([clicks, clicks], axis=1)
    path = write_wav(tmp_path / "clicks.wav", data, sr)
    inp = _id(path)
    out = {m.kind: m.result for m in audio.describe(inp, (0.0, 1.0), ("level", "onsets", "stereo"))}
    onsets = out["onsets"]["onsets"]
    detected = [o["t"] for o in onsets]
    for expected in [0.1, 0.3, 0.5, 0.7, 0.9]:
        assert any(abs(d - expected) < 0.02 for d in detected), detected
    assert out["level"]["peak_dbfs"] > -3
    assert out["stereo"]["windows"][0]["corr"] == pytest.approx(1.0)


def test_describe_spectrum_centroid(tmp_path):
    path = write_wav(tmp_path / "sine.wav", sine(SR, 1.0, freq=1000.0), SR)
    inp = _id(path)
    out = {m.kind: m.result for m in audio.describe(inp, (0.0, 1.0), ("spectrum",))}
    centroid = np.mean([w["centroid_hz"] for w in out["spectrum"]["windows"]])
    assert centroid == pytest.approx(1000.0, rel=0.15)


def test_describe_mono_stereo_unavailable(tmp_path):
    path = write_wav(tmp_path / "mono.wav", sine(SR, 0.5)[:, :1], SR)
    inp = _id(path)
    out = {m.kind: m.result for m in audio.describe(inp, None, ("stereo",))}
    assert out["stereo"]["available"] is False


# --- spectral difference (phase-invariant) ---


def _voice(f0, partials, amp, seconds, seed):
    t = np.arange(int(SR * seconds)) / SR
    rng = np.random.default_rng(seed)
    sig = np.zeros_like(t)
    for h, a in enumerate(partials, start=1):
        sig += a * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
    return amp * sig


def _mix(seconds=6.0, v3_seed=3, seed_base=0):
    """Three-voice multi-partial mix; v3 has a partial at 1500 Hz (375*4)."""
    v1 = _voice(110, [1, 0.5, 0.25, 0.12], 0.15, seconds, seed_base * 10 + 1)
    v2 = _voice(220, [1, 0.4, 0.2], 0.12, seconds, seed_base * 10 + 2)
    v3 = _voice(375, [1, 0.45, 0.22, 0.35], 0.10, seconds, v3_seed)
    return np.repeat((v1 + v2 + v3)[:, None], 2, axis=1)


def _v3(seconds=6.0, seed=3, f0=375.0):
    return np.repeat(_voice(f0, [1, 0.45, 0.22, 0.35], 0.10, seconds, seed)[:, None], 2, axis=1)


def _spectral(tmp_path, a_data, b_data):
    a = _id(write_wav(tmp_path / "a.wav", a_data, SR), "A", "before")
    b = _id(write_wav(tmp_path / "b.wav", b_data, SR), "B", "after")
    return audio.signal_difference(a, b).result


def test_spectral_phase_only_no_regions(tmp_path):
    """Same tones, new oscillator start phases: waveform diff is global, spectral is clean."""
    res = _spectral(tmp_path, _mix(), _mix(v3_seed=13, seed_base=1))
    spec = res["spectral"]
    assert spec["available"] is True
    assert spec["regions"] == [], spec["regions"]
    assert spec["global_change"] is False


def test_spectral_tiny_noise_no_regions(tmp_path):
    a = _mix()
    b = a + 1e-5 * np.random.default_rng(0).standard_normal(a.shape)
    res = _spectral(tmp_path, a, b)
    assert res["spectral"]["regions"] == [], res["spectral"]["regions"]


def test_spectral_detuned_voice_region(tmp_path):
    a = _mix()
    b = a.copy()
    lo, hi = int(2.0 * SR), int(4.0 * SR)
    b[lo:hi] += (_v3(f0=375 * 2 ** (50 / 1200), seed=9) - _v3())[lo:hi]
    res = _spectral(tmp_path, a, b)
    regions = res["spectral"]["regions"]
    assert any(r["start"] <= 2.1 and r["end"] >= 3.9 for r in regions), regions


def test_spectral_gain_drop_region(tmp_path):
    a = _mix()
    b = a.copy()
    lo, hi = int(2.0 * SR), int(4.0 * SR)
    b[lo:hi] += (10 ** (-9 / 20) - 1.0) * _v3()[lo:hi]
    res = _spectral(tmp_path, a, b)
    assert res["spectral"]["regions"], res["spectral"]


def _peaking(x, f0, q, gain_db):
    a_lin = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    alpha = np.sin(w0) / (2 * q)
    b0, b1, b2 = 1 + alpha * a_lin, -2 * np.cos(w0), 1 - alpha * a_lin
    a0, a1, a2 = 1 + alpha / a_lin, -2 * np.cos(w0), 1 - alpha / a_lin
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    y = np.zeros_like(x)
    for i in range(2, x.shape[0]):
        y[i] = b[0] * x[i] + b[1] * x[i - 1] + b[2] * x[i - 2] - a[1] * y[i - 1] - a[2] * y[i - 2]
    return y


def test_spectral_eq_boost_region(tmp_path):
    a = _mix()
    b = a.copy()
    lo, hi = int(2.0 * SR), int(4.0 * SR)
    v3 = _v3()
    v3_eq = np.repeat(_peaking(v3[:, 0], 1500, 8, 12)[:, None], 2, axis=1)
    b[lo:hi] += (v3_eq - v3)[lo:hi]
    res = _spectral(tmp_path, a, b)
    assert res["spectral"]["regions"], res["spectral"]


def _pluck_mix(attack_s, seconds=6.0):
    v1 = _voice(110, [1, 0.5, 0.25, 0.12], 0.15, seconds, 1)
    v3 = _voice(375, [1, 0.45, 0.22, 0.35], 0.10, seconds, 3)
    pluck = np.zeros(int(SR * seconds))
    for start in np.arange(0, seconds, 1.0):
        n = int(SR * min(1.0, seconds - start))
        t = np.arange(n) / SR
        env = np.minimum(t / attack_s, 1.0) * np.exp(-3 * t)
        ph = np.sin(2 * np.pi * 220 * t) + 0.4 * np.sin(2 * np.pi * 440 * t)
        pluck[int(start * SR) : int(start * SR) + n] += 0.15 * env * ph
    return np.repeat((v1 + pluck + v3)[:, None], 2, axis=1)


def test_spectral_attack_change_region(tmp_path):
    res = _spectral(tmp_path, _pluck_mix(0.010), _pluck_mix(0.150))
    assert res["spectral"]["regions"], res["spectral"]


def test_spectral_identical_inputs(tmp_path):
    res = _spectral(tmp_path, _mix(), _mix())
    spec = res["spectral"]
    assert spec["available"] is True
    assert spec["regions"] == []
    assert spec["changed_fraction"] == 0
