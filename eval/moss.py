# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2,<3", "soundfile>=0.12,<0.14", "scipy>=1.13,<2", "jsonschema>=4,<5"]
# ///
"""MOSS evaluation only. Run with the external MOSS venv; see moss-results.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import jsonschema
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import corpus
from build import EVAL_DIR, recipe_key, sha256
from run import fingerprint

from agent_listening import audio, evidence, prompts, tasks

MODEL_REVISION = "fce7f8304e96cc2d3398b8106456cbb2ecec3139"
SOURCE_REVISION = "ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e"
GENERATION = {"max_new_tokens": 2048, "do_sample": False, "num_beams": 1, "use_cache": True}


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def parse_response(text: str, schema: dict) -> dict:
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    payload = json.loads(text)
    jsonschema.validate(payload, schema)
    return payload


def chat_prompt(built: prompts.ListeningPrompt, labels: list[str]) -> str:
    media = "\n".join(f"Clip {label}: <|audio_bos|><|AUDIO|><|audio_eos|>" for label in labels)
    return (
        f"<|im_start|>system\n{built.system}<|im_end|>\n<|im_start|>user\n{media}\n"
        f"{built.prompt}\nReturn only JSON matching this schema. Use concise claims.\n"
        f"{json.dumps(built.schema)}<|im_end|>\n<|im_start|>assistant\n"
    )


def finite_logits(input_ids, scores):
    if not scores.isfinite().all().item():
        raise ValueError("NONFINITE_LOGITS: inference is numerically invalid")
    return scores


class Model:
    def __init__(self, source: Path, weights: Path, device: str):
        import torch

        sys.path.insert(0, str(source))
        from src.modeling_moss_music import MossMusicModel
        from src.processing_moss_music import MossMusicProcessor

        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != SOURCE_REVISION:
            raise ValueError("unexpected MOSS source revision")
        if subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--name-only"]):
            raise ValueError("MOSS source has local patches")
        self.torch, self.device = torch, device
        torch.manual_seed(0)
        torch.set_num_threads(8)
        started = time.monotonic()
        self.model = MossMusicModel.from_pretrained(
            str(weights),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",
        ).eval()
        self.processor = MossMusicProcessor.from_pretrained(str(weights), enable_time_marker=True)
        self.load_s = time.monotonic() - started

    def call(self, path: Path, built: prompts.ListeningPrompt, inputs: list, windows: dict) -> dict:
        if path.exists():
            return json.loads(path.read_text())
        import torchaudio

        torch = self.torch
        if self.device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        prompt = chat_prompt(built, [i.label for i in inputs])
        record = {
            "model_revision": MODEL_REVISION,
            "source_revision": SOURCE_REVISION,
            "device": self.device,
            "dtype": "bfloat16",
            "encoder_conv_chunksize": self.model.audio_encoder.conv_chunksize,
            "attention": {
                "requested": "eager",
                "language": self.model.language_model.config._attn_implementation,
                "encoder": self.model.config.audio_config._attn_implementation,
            },
            "generation": GENERATION,
            "finite_logits_checked": True,
            "seed": 0,
            "mps_memory_fraction": "torch default",
            "mps_recommended_bytes": torch.mps.recommended_max_memory() if self.device == "mps" else None,
            "prompt": prompt,
            "model_load_s": self.load_s,
            "inputs": [],
            "status": "failed",
        }
        started = time.monotonic()
        stop = threading.Event()
        peaks = {
            "mps_driver_bytes_sampled": 0,
            "mps_tensor_bytes_sampled": 0,
            "rss_bytes_process_highwater": 0,
        }

        def sample():
            while not stop.is_set():
                peaks["rss_bytes_process_highwater"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if self.device == "mps":
                    peaks["mps_tensor_bytes_sampled"] = max(
                        peaks["mps_tensor_bytes_sampled"], torch.mps.current_allocated_memory()
                    )
                    peaks["mps_driver_bytes_sampled"] = max(
                        peaks["mps_driver_bytes_sampled"], torch.mps.driver_allocated_memory()
                    )
                stop.wait(0.1)

        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        try:
            waveforms = []
            for inp in inputs:
                if sha256(Path(inp.path)) != inp.sha256:
                    raise ValueError(f"input changed: {inp.path}")
                start, end = windows[inp.label]
                x, sr = sf.read(
                    inp.path,
                    start=round(start * inp.format.sample_rate),
                    stop=round(end * inp.format.sample_rate),
                    always_2d=True,
                    dtype="float32",
                )
                x = torchaudio.functional.resample(torch.from_numpy(x.mean(axis=1)), sr, 16000).numpy()
                waveforms.append(x)
                record["inputs"].append(
                    {
                        "path": inp.path,
                        "sha256": inp.sha256,
                        "label": inp.label,
                        "window_s": [start, end],
                        "sample_rate": 16000,
                        "samples": len(x),
                        "mono_float32_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                    }
                )
            encoded = self.processor(text=prompt, audios=waveforms, return_tensors="pt")
            record["input_tokens"] = int(encoded["input_ids"].shape[1])
            if record["input_tokens"] + GENERATION["max_new_tokens"] > 40960:
                raise ValueError("model context budget exceeded; audio was not truncated")
            encoded = encoded.to(self.device)
            encoded["audio_data"] = encoded["audio_data"].to(self.model.dtype)
            encoded["audio_input_mask"] = encoded["input_ids"] == self.processor.audio_token_id
            with torch.inference_mode():
                ids = self.model.generate(**encoded, **GENERATION, logits_processor=[finite_logits])
            if self.device == "mps":
                torch.mps.synchronize()
            output = ids[0, record["input_tokens"] :]
            record["output_tokens"] = len(output)
            record["raw_response"] = self.processor.decode(output, skip_special_tokens=True)
            record["payload"] = parse_response(record["raw_response"], dict(built.schema))
            record["status"] = "complete"
        except Exception:
            record["error"] = traceback.format_exc()
        finally:
            stop.set()
            monitor.join()
            record["wall_s"] = round(time.monotonic() - started, 3)
            record["memory"] = peaks
            write(path, record)
        return record


def execute(case: dict, condition: str, manifest: dict, model: Model, run_dir: Path) -> None:
    out = run_dir / f"{case['id']}__{condition}.json"
    if out.exists():
        return
    paths = [manifest[recipe_key(r)] for r in case["inputs"]]
    inputs = [
        audio.identify(p["path"], "AB"[i], ("before", "after")[i] if case["task"] == "compare" else "subject")
        for i, p in enumerate(paths)
    ]
    if any(i.sha256 != p["sha256"] for i, p in zip(inputs, paths, strict=True)):
        raise ValueError("build-manifest hash mismatch")
    measurements = []
    contexts = []
    if condition == "H+C":
        contexts = [str(Path(i.path).with_suffix(".scd")) for i in inputs]
    if condition == "MC":
        misleading = manifest[recipe_key(case["misleading_context"])]["path"]
        contexts = [str(Path(inputs[0].path).with_suffix(".scd")), str(Path(misleading).with_suffix(".scd"))]
    context_files, context_views = tasks._contexts(contexts)
    windows = {i.label: (0.0, i.format.duration_s) for i in inputs}
    if case["task"] == "investigate":
        windows["A"] = tuple(case["focus"])
        measurements = audio.describe(inputs[0], windows["A"], tasks.MEASURE_KINDS)
    if len(inputs) == 2:
        measurements = [audio.pcm_identity(*inputs), audio.signal_difference(*inputs)]
    identical = len(inputs) == 2 and measurements[0].result["identical"] and condition != "IM"
    measured = (
        {m.kind: tasks._clip_relative(m.result, windows["A"][0]) for m in measurements}
        if case["task"] == "investigate"
        else None
    )
    clip_infos = {
        i.label: evidence.ClipInfo(
            i.label,
            i.label,
            windows[i.label][0],
            windows[i.label][1] - windows[i.label][0],
            i.format.duration_s,
        )
        for i in inputs
    }
    claims, calls, problems = [], [], []
    for order in [inputs, inputs[::-1]] if condition == "CB" else [inputs]:
        if identical:
            break
        views = [
            prompts.ClipView(
                i.label,
                i.role,
                windows[i.label][1] - windows[i.label][0],
                windows[i.label][0],
                i.format.duration_s,
                1,
            )
            for i in order
        ]
        built = prompts.build(
            "compare-focus" if condition == "F" else case["task"],
            views,
            question=case.get("question"),
            contexts=context_views,
            measurements=measured,
            region=tuple(case["truth"]["regions"][0]) if condition == "F" else None,
        )
        raw_path = run_dir / "raw" / f"{case['id']}__{condition}__{''.join(i.label for i in order)}.json"
        record = model.call(raw_path, built, order, windows)
        calls.append(record)
        parsed, issues = evidence.claims_from_model(
            record.get("payload", {}), clip_infos, id_prefix=f"c{len(calls)}_"
        )
        problems.extend(issues)
        if len(inputs) == 2:
            diff = measurements[1]
            regions, global_change, spectral, spectral_global = tasks._difference_regions(diff)
            parsed = evidence.apply_difference_check(
                parsed,
                regions=regions,
                global_change=global_change,
                measurement_id="m2",
                spectral_regions=spectral,
                spectral_global=spectral_global,
            )
        claims.append(parsed)
    if condition == "CB":
        first, second = evidence.apply_order_check(*claims)
        claims = [first, second]
    document = {
        "task": case["task"],
        "status": "failed" if any(c["status"] == "failed" for c in calls) else "complete",
        "inputs": [asdict(i) for i in inputs],
        "context": [asdict(c) for c in context_files],
        "claims": [asdict(c) for group in claims for c in group],
        "measurements": [asdict(replace(m, id=f"m{n + 1}")) for n, m in enumerate(measurements)],
        "answer": calls[0].get("payload", {}).get("answer") if calls else None,
        "summary": calls[0].get("payload", {}).get("summary", "") if calls else "PCM-identical",
        "producer": {
            "interaction_ids": [str(n) for n in range(len(calls))],
            "latency_s": sum(c["wall_s"] for c in calls),
            "usage": {},
        },
        "error": [c.get("error") for c in calls if c["status"] == "failed"] or None,
        "adapter_problems": problems,
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_only": True,
        "condition": condition,
    }
    write(out, document)
    print(f"{out.name}: {document['status']} {document['producer']['latency_s']:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["pilot", "holdout"], required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--source", type=Path, default=EVAL_DIR / "MOSS-Music")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path.home()
        / ".cache/huggingface/hub"
        / "models--OpenMOSS-Team--MOSS-Music-8B-Instruct/snapshots"
        / MODEL_REVISION,
    )
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--match", default="")
    args = parser.parse_args()
    run_dir = EVAL_DIR / "runs" / args.run
    manifest = json.loads((EVAL_DIR / "build-manifest.json").read_text())
    before = fingerprint(manifest)
    snapshot = run_dir / "integrity-before.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != before:
        raise ValueError("corpus fingerprint changed since the run began")
    if not snapshot.exists():
        write(snapshot, before)
    model = Model(args.source, args.weights, args.device)
    selected = [c for c in corpus.cases() if c["split"] == args.split and re.search(args.match, c["id"])]
    try:
        for case in selected:
            for condition in case["conditions"]:
                execute(case, condition, manifest, model, run_dir)
            if case["truth"]["kind"] == "change":
                execute(case, "F", manifest, model, run_dir / "focused")
            if case["truth"]["kind"] == "identical":
                execute(case, "IM", manifest, model, run_dir / "identical-model")
    finally:
        after = fingerprint(manifest)
        changed = [p for p, digest in before.items() if after.get(p) != digest]
        write(run_dir / "integrity-after.json", {"files_checked": len(before), "changed": changed})
        if changed:
            raise ValueError(f"originals changed: {changed}")


if __name__ == "__main__":
    main()
