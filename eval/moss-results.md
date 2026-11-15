# MOSS-Music comparative listening evaluation

## Method fixed before the holdout

This evaluation asks whether MOSS-Music supplies independently checked musical listening gains over
the stored `gemini-4-hydrogen` holdout. The decision set is the existing holdout, including MUSE
(`AGENT_LISTENING_EVAL_PAID_TIER=1` enables those existing cases; inference here is local).
Development uses only pilot cases. The initial method commit preceded the first MOSS holdout attempt. That SDPA attempt was terminated
when numerical invalidity was demonstrated; a second start exposed a nested-backend configuration
error. Both are excluded. The final method is committed before the valid decision run; the invalid attempt is retained and excluded from capability conclusions.
No product dependency, contract, threshold, CLI behaviour or SC Agent IDE code changes.

The runner resolves `corpus.py` recipes against `build-manifest.json`, checks the original file hashes,
and uses the product's unmodified `prompts.build`, `claims_from_model`, deterministic difference checks
and `apply_order_check`. The resulting files are **scorer inputs**, marked `evaluation_only`, not
complete product evidence documents. `eval/score.py` supplies the scoring functions without changes.
Raw responses, exact prompts, original paths/hashes, resampled waveform hashes, generation settings,
device, dtype, elapsed time and sampled memory are saved atomically after each call. Existing response
files are resumed rather than regenerated. Schema/JSON failures are recorded, without model repair.

The conditions are:

- H: the existing task and question, without context. `investigate` uses the same physical focus excerpt
  and deterministic measurements as Hydrogen. Description and multiple-choice questions retain their
  existing tasks. Unprompted description scores measure grounding and coverage, not perceptual truth.
- CB: the same 15 holdout Bach10 edits as Hydrogen. Two fresh calls present A/B and B/A; clip identities
  and reference/after roles stay fixed. The existing checker matches aspect and overlapping original
  loci. Its denominator is **difference claims**, not cases; a case with no differences contributes
  no claim. Report case coverage alongside that metric.
- H+C and MC: the same source files and misleading context as `eval/run.py`. No filenames, construction
  recipe, answer, instrument identity or expected change aspect enter the listening prompt.
- F: all 23 controlled holdout edits, as a separate condition. Both **complete audio files** remain
  unchanged; the existing `compare-focus` text supplies only the construction recipe's known changed
  interval, in seconds. This is oracle location assistance, with no changed aspect or answer supplied.
  It measures naming with attention directed to a region, not independent detection or localization.
  It differs from Hydrogen's automatic, selectively triggered cropped follow-ups and is not claimed
  to be an identical-task comparison against those follow-ups.
- Identical: the same PCM identity shortcut as Hydrogen (zero model calls), plus separate IM calls
  that send the identical pair to MOSS. IM measures model hallucination and is not attributed to the
  zero-call product path. Near-null and MC comparisons always reach the model.

The prompt uses the same system instructions and task text as Hydrogen, with the existing JSON schema
written in the user message because Transformers has no constrained JSON response mode here. It asks
for concise JSON only. Parsing accepts plain JSON or a single `json` code fence, validates the schema,
and otherwise records failure. A logits processor also rejects every generation step with any NaN
or infinity, before argmax can turn invalid scores into repeated tokens. There are no retries,
output repairs or answer extraction from prose.
Greedy decoding on MPS uses BF16, SDPA language prefill and eager cached language decoding
(the encoder retains eager attention), `do_sample=False`, `num_beams=1`, `max_new_tokens=2048`, `use_cache=True`,
and seed 0. The encoder retains its upstream batch size of 64 four-second chunks and default MPS allocation
limits. Unused MPS cache is cleared before each call. This removes sampling randomness; cross-device bitwise determinism is not asserted.

Report `score.py`'s correct-aspect detection, aspect-correct onset Hit@3s, any-aspect interval overlap
with the known region padded by 3 s, claim-based order consistency, raw and checked null differences,
choice accuracy and change-kind breakdown. These distinct location metrics must not be conflated.
Wilson intervals and median bootstrap intervals come from that scorer. Missing/failed calls and parse
failures are reported separately; a successful-only score is never presented as complete coverage.
Focused results use `score_change` directly in a separate group, without adding them to independent H.
No model judges a model. No listening-panel or musical-quality claim follows from schema validity.

## Upstream and machine verification

Source: [OpenMOSS/MOSS-Music](https://github.com/OpenMOSS/MOSS-Music/tree/ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e),
commit `ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e`. Weights:
[OpenMOSS-Team/MOSS-Music-8B-Instruct](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct/tree/fce7f8304e96cc2d3398b8106456cbb2ecec3139),
revision `fce7f8304e96cc2d3398b8106456cbb2ecec3139`. Model licence: Apache-2.0.

[audio_io.py](https://github.com/OpenMOSS/MOSS-Music/blob/ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e/src/audio_io.py)
averages channels and resamples to the processor's 16,000 Hz. The runner decodes with soundfile,
averages channels and uses the same `torchaudio.functional.resample` defaults. This bypasses
`torchaudio.load`'s TorchCodec runtime without patching upstream code. Audio is not normalised.

[processing_moss_music.py](https://github.com/OpenMOSS/MOSS-Music/blob/ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e/src/processing_moss_music.py)
accepts multiple waveforms and explicit repeated audio spans, checks placeholder count, and resets
the two-second time markers for each span. The default convenience prompt creates only one span;
passing two waveforms to that default would fail. Our explicit one-turn chat contains a labelled span
for each file. A real two-waveform MPS generation succeeded before pilot. There is no concatenation
or silence-gap substitution. Inference imports the official GitHub `src` classes, as `infer.py` does;
the checkpoint bundles an older encoder implementation. The tested source revision is therefore part
of the result, not interchangeable with arbitrary `AutoModel` remote code.

The official sources do not state a validated maximum audio duration. The model config has a
40,960-token language context; audio representations use 12.5 tokens/s plus time markers. The encoder
processes four-second mel chunks, so `max_source_positions=1500` is not a whole-recording duration cap.
The runner checks the actual total token count plus output budget and fails rather than truncating.
The longest corpus source is 418.5 s; the longest pair is about 837 s combined. Successful processing
would establish coverage of these inputs, not a general maximum-duration guarantee.

[infer.py](https://github.com/OpenMOSS/MOSS-Music/blob/ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e/infer.py)
samples with temperature 1.0, top-p 0.8 and top-k 50; the
[serving guide](https://github.com/OpenMOSS/MOSS-Music/blob/ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e/moss_music_usage_guide.md)
uses temperature 0.0 examples. The upstream HF wrapper explicitly supports greedy decoding with
`do_sample=False`. Upstream recommends SGLang for best quality. This evaluation covers its Transformers
path on Apple MPS, not SGLang/CUDA performance or published benchmark reproduction.

[pyproject.toml](https://github.com/OpenMOSS/MOSS-Music/blob/ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e/pyproject.toml)
declares base dependencies safetensors, numpy, soundfile, tiktoken, einops, scipy, tqdm, packaging,
gradio and requests; the optional runtime pins torch/torchaudio 2.9.1+cu128, TorchCodec 0.9.*, Transformers
4.57.1 and accelerate. The inference import path needs torch, torchaudio, Transformers, numpy and
einops, plus Transformers' dependencies and the audio decoder. We installed the non-CUDA torch and
torchaudio 2.9.1 wheels, Transformers 4.57.1 and accelerate in Python 3.12.16 outside the product env.
Gradio, TorchCodec, FlashAttention and SGLang are unnecessary for this soundfile-based inference path.
The full resolved environment is pinned in `eval/moss-requirements.txt` and copied to the external feasibility directory.

The first long pilot call with upstream batch size 64 reached 146.48 GB in sampled MPS driver
allocations, including cached allocations. A separate working-set-limit experiment was interrupted
after about 120 s as system swap increased. A further batch-size-8 + working-set-limit trial was stopped after about 200 s without a response;
macOS sampling recorded a peak physical footprint of 108.8G and a thread waiting for Metal work.
The final method restores the already successful upstream defaults and clears unused cache between
calls. These bounded memory experiments are retained as deployment evidence, not holdout results.

Machine: Apple M4 Max, arm64, 128 GB unified memory. MPS is available. The initial smoke test
verified text generation but did not check cached-step logits. On a longer pair, a separate diagnostic
found all 151,936 next-token logits nonfinite at generation step 1 under BF16 MPS SDPA. The invalid
outputs can degenerate into repeated `!`; they cannot be scored as listening ability.

Setting only the outer `from_pretrained(attn_implementation=...)` argument does not reliably configure
the nested Qwen model in this wrapper. The runner explicitly sets and checks `model.language_model`.
Fully eager MPS passed short pilot calls and a 418.452 s single input (155.44 s wall time), but reached
190.45 GB sampled driver allocation and crashed with SIGSEGV before producing logits on two such
inputs (12,567 tokens). The bounded CPU fallback used FP32 + SDPA and completed that full structured
comparison in 316.45 s, 680 finite generation steps, peak RSS 53.96 GB. These are feasibility probes,
not holdout capability observations.

The final MPS policy uses SDPA for prefill and switches the nested language model to eager immediately
after checking the first finite logits. Cached decode then uses eager. The standard Qwen3 cache holds
ordinary RoPE-transformed K/V tensors, independent of attention backend; both implementations use
the same attention scaling, grouping and causal access. Transformers 4.57.1 supports the public setter
and rereads the implementation on each forward. Numerical results need not be bitwise identical.
The runner resets SDPA before every request and restores it in `finally`; calls are sequential.
No upstream source or weights are patched. This policy passed the 12,567-token pilot-source pair:
prefill SDPA and all three cached eager steps were finite, 43.67 s including load, sampled driver peak
85.09 GB and tensor peak 42.48 GB. The final policy also completed all three pilot calls (short comparison, focused comparison and
418.452 s choice input), with finite logits and valid JSON: 51.8 s, 14.4 s and 47.7 s respectively.
These checks preceded the final method commit.

CPU mode loads the same BF16 checkpoint, converts to FP32, and uses SDPA throughout. Memory figures
use decimal GB. Driver allocation includes caches/in-flight allocations; it must not be summed with
RSS or presented as an exact physical footprint. Sampling is every 0.1 s; RSS is process high-water.
There is no HF_TOKEN environment value or saved token, so HF Jobs was not used. External service
charge is zero; electricity and hardware amortisation are unmeasured. SGLang/CUDA remains untested.

`hf download OpenMOSS-Team/MOSS-Music-8B-Instruct` resolved the existing cache. All 24 files passed
remote size and available digest checks (LFS SHA256; Git blob SHA1 for other files), totalling
18,119,304,629 bytes. The standalone venv occupies about 647 MB. Upstream clone and weights are outside
git; the runner rejects a different source commit or tracked local patch.

## Reproduction

From the product checkout, using the isolated interpreter:

```sh
export AGENT_LISTENING_EVAL_DIR="${AGENT_LISTENING_EVAL_DIR:-$HOME/.cache/agent-listening-eval}"
git clone https://github.com/OpenMOSS/MOSS-Music "$AGENT_LISTENING_EVAL_DIR/MOSS-Music"
git -C "$AGENT_LISTENING_EVAL_DIR/MOSS-Music" checkout ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e
uv venv --python 3.12 "$AGENT_LISTENING_EVAL_DIR/moss-venv"
uv pip install --python "$AGENT_LISTENING_EVAL_DIR/moss-venv/bin/python" \
  -r eval/moss-requirements.txt
hf download OpenMOSS-Team/MOSS-Music-8B-Instruct --revision fce7f8304e96cc2d3398b8106456cbb2ecec3139
AGENT_LISTENING_EVAL_PAID_TIER=1 "$AGENT_LISTENING_EVAL_DIR/moss-venv/bin/python" \
  eval/moss.py --split holdout --run moss-holdout-3
AGENT_LISTENING_EVAL_PAID_TIER=1 uv run eval/score.py --run moss-holdout-3 --split holdout
AGENT_LISTENING_EVAL_PAID_TIER=1 uv run eval/moss_score.py --run moss-holdout-3
uv run --with scipy python -m pytest -q eval/test_moss.py
```

Set `AGENT_LISTENING_EVAL_DIR` to `/Users/dmus/.cache/agent-listening-eval` if it is not already set.
Do not use a new holdout run name to tune the method after observing decision results. An interrupted
run may resume the same directory; completed raw calls are reused, including parse failures.

## Results

Historical SDPA pilot (execution/format evidence, not a numerically validated capability result):
15 completed raw calls, zero JSON/schema failures, across 8 selected pilot cases plus a
counterbalance and identical-model control. H detection was 0/3; focused detection was 0/3; choice
accuracy was 2/4. The near-null and identical-model controls returned no differences. Repeating the
same AB prompt in the counterbalance call produced exactly the same text. Prompt and scoring
thresholds were not changed after these observations. The original `runs/moss-holdout-1` attempt produced nonfinite logits (151,936/151,936
at generation step 1) and is invalid for capability comparisons; see `runs/moss-feasibility/logits-mps.json`.
The eager diagnostic is `logits-mps-eager.json`. The second start (`moss-holdout-2`) also used SDPA
inside Qwen despite the outer eager flag; it was rejected by the finite-logits guard and excluded.
Actual eager pilot output is `runs/moss-pilot-5-explicit-eager`; the final phase-switching policy is
checked in `runs/moss-pilot-6-hybrid`. The long-pair probes are `long-pair-mps.json` (native crash),
`long-pair-cpu.json` (complete FP32 fallback) and `long-pair-mps-hybrid.json` (finite phase switch).
The held-out decision results follow below after the revised method commit.
