# MOSS-Music comparative listening evaluation

Status: a preserved negative result. The frozen Transformers/MPS method below does not enter the product.
That is a decision under this method and evidence (research baseline `agent-listening-1.0` at
`a58f45581ff9dad441b137a425fc38484bade6d6`), not a verdict on the model: a new model version, another runtime
or output method, or evidence from a specific task is a reason to evaluate again. The decision and how these
metrics should be read are in [docs/architecture.md §4.2 and §4.3](../docs/architecture.md#42-how-to-read-these-results).
The comparison uses the holdout that development has already seen, so it diagnoses this method and does not
measure either model on unseen material.

MOSS-Music-8B-Instruct does **not demonstrate an independently verifiable listening gain** for Agent Listening in this frozen Transformers/MPS evaluation. It offers three isolated multiple-choice wins and avoids the one misleading-context false positive, but supplies no scored change identification, loses substantially on the choice set, and has material output-format and deployment failures. This conclusion applies to the pinned implementation and method below; it does not establish the model's intrinsic limit under SGLang/CUDA or another prompt.

**1. MOSS vs Hydrogen on identical tasks**

The decision set is the existing holdout, including all 18 MUSE questions. H uses the same audio, questions and product system/task instructions without source context; `investigate` retains Hydrogen's focus crop and measurements. MOSS receives the existing JSON schema as prompt text rather than Hydrogen's constrained response mode. Native multiple-audio spans were verified; no concatenation was needed. Failed outputs are unscored, never treated as correct negatives. Fractions below use valid outputs; planned coverage is explicit.

| Identical task / existing `score.py` metric | Hydrogen `holdout-1` | MOSS recovered frozen run |
|---|---:|---:|
| H correct change aspect | 3/23, 13.0% | 0/19 valid, 0%; 19/23 covered |
| H correct-aspect onset within 3 s | 3/23 | 0/19; 19/23 covered |
| H any-aspect difference locus overlapping true interval ±3 s | 7/23 | 0/19; 19/23 covered |
| CB order-consistent **difference claims** | 0/10 claims, across 6 cases; 15/15 cases valid | Unmeasurable: 0 eligible claims; 6/15 paired cases valid |
| H near-null cases with raw difference claim | 0/5 | 0/3 valid; 3/5 covered, 2 parse failures |
| MC misleading-context cases with raw difference claim | 1/1, 2 false pitch/tuning claims | 0/1, no difference claims |
| Identical PCM shortcut | 4/4, zero model calls and claims | 4/4, zero model calls and claims |
| Choice accuracy | 39/45, 86.7% | 14/43 valid, 32.6%; 43/45 covered |
| Source-context change identification, same three SC cases | H 0/3 → H+C 3/3 | H 0/3 → H+C 0/3 |
| Standalone `hear` | 4/4 valid; 53 claims; 30 musical relations; all four scales in 3/4 | 4/4 valid; 13 claims; 0 musical relations; all claims work-scale |

The Wilson 95% intervals are 4.5–32.1% for Hydrogen H change identification, 0–16.8% for MOSS H, 73.8–93.7% for Hydrogen choices, and 20.5–47.5% for MOSS choices. On the **same 43 valid MOSS choice cases**, Hydrogen is 38/43 and MOSS 14/43: 3 MOSS-only wins, 27 Hydrogen-only wins, 11 both correct, 2 both wrong. On the same 19 valid change cases, Hydrogen is 2/19 and MOSS 0/19. Four failed change responses include Hydrogen's third success, `bach10-05-pitch`.

Correct change type here means the scorer's pre-existing allowed-aspect match on a `differs` claim, not independently confirmed prose semantics. MOSS's 19 valid H comparisons contain 40 claims: 18 `repeats`, 18 `unchanged`, one `transforms`, one `varies`, and two without a relation; **none is `differs`**, despite the explicit comparison instruction. No prose or relation remapping was introduced after observing holdout results. Thus the zero includes failure to express evidence in the required protocol. It does not prove that every sentence lacked musical content. Likewise, a low raw-difference rate cannot establish selective discrimination when change detection is zero. `hear` grounding and scale counts measure formal coverage, not caption truth; neither model's captions received an independent listening-panel correctness grade.

Correct-aspect identification by change kind is below. Each MOSS cell is **correct / valid [planned]**; F is assisted and separate.

| Construction change | Hydrogen H | MOSS H | MOSS F |
|---|---:|---:|---:|
| Resonance | 0/5 | 0/4 [5] | 0/4 [5] |
| Articulation | 0/3 | 0/2 [3] | 0/2 [3] |
| Timbre as a separate construction | N/A | N/A | N/A |
| Pitch | 1/5 | 0/4 [5] | 0/4 [5] |
| Timing | 1/3 | 0/2 [3] | 0/2 [3] |
| Mute | 1/3 | 0/3 [3] | 0/3 [3] |
| Entry | 0/1 | 0/1 [1] | 0/1 [1] |
| Balance | 0/1 | 0/1 [1] | 0/1 [1] |
| Dynamics | 0/1 | 0/1 [1] | 0/1 [1] |
| Removed return | 0/1 | 0/1 [1] | 0/1 [1] |

Resonance recipes accept the corpus's registered timbre/space/production aspects; they do not create an additional timbre test. Choice-family counts use the same scorer grouping:

| Choice family | Hydrogen | MOSS correct / valid [planned] |
|---|---:|---:|
| Instrument count | 3/3 | 0/3 [3] |
| Highest voice | 1/3 | 2/3 [3] |
| Lowest voice | 2/3 | 2/3 [3] |
| Focused articulation (`bach10`) | 1/3 | 0/2 [3] |
| Entries | 1/1 | 0/1 [1] |
| Instrument family | 4/4 | 4/4 [4] |
| MUSE contour | 18/18 | 2/17 [18] |
| Pulse | 2/2 | 1/2 [2] |
| Return form | 1/1 | 0/1 [1] |
| Return time | 1/1 | 0/1 [1] |
| Texture | 2/2 | 0/2 [2] |
| Theme count | 1/1 | 1/1 [1] |
| Tuning | 1/1 | 1/1 [1] |
| Voices | 1/2 | 1/2 [2] |

**2. Which Hydrogen failures MOSS solved**

There are three observed choice wins: `bach10-05-ask-highest`, `bach10-08-ask-highest`, and `aof-canon8-ask-voices`. MOSS also reports no audible changes for the misleading-context `sc-canon-nearnull__MC` pair, where Hydrogen generated two false pitch/tuning differences. These are case-level successes. MOSS solves **none** of Hydrogen's missed controlled change cases under the registered metric, and does not demonstrate an order-consistency improvement.

Hydrogen's stored comparison measurements predate the current spectral checker. A separate, model-free recheck applies the same current checker used for MOSS: both Hydrogen MC claims become contradicted (2 surviving → 0), while its H detection stays 3/23 and CB stays 0/10. MOSS also has zero surviving MC difference claims. This checker effect must not be credited to either model.

**3. Which failures remain**

Independent change identification and event localization remain unsolved. The conditional onset-error median for Hydrogen is 0.20 s across only its three correct-aspect detections; MOSS has no eligible detections, so its median is undefined. Deterministic signal localization is 23/23 for Hydrogen and 19/19 valid H MOSS documents because both use the same signal method; this is not a MOSS perception gain.

MOSS CB produced 30 primary responses but only six complete AB/BA case pairs; 13 of the 30 responses failed validation. The six valid pairs emitted no `differs` claims. Its claim-based order score is undefined, not 0% or 100%. The separate identical-audio **model** control returned no difference claims on 3/3 valid calls; the fourth (`bach10-05-identical__IM`) failed output parsing and its hallucination state is unknown. The four zero-call identity decisions remain separately verified.

All **138 planned raw response slots** are present: 112 schema-valid outputs and **26 JSON/schema failures (18.8%)**, with no runtime failures left in the recovered view. The standard 100 case/condition documents contain 83 scored and 17 failed, with zero missing; the separate focused set has 19/23 valid and the identical-model set 3/4 valid. Of the generated responses, 22 hit the frozen 2,048-token cap and failed JSON syntax, and four failed schema validation: two missing `relation`, one missing `answer`, and one invalid `harmony` aspect. No generated answer was retried, repaired or scored from a guessed letter. These are pipeline failures, not evidence of correct negative perception.

**4. Did measurement-guided focusing improve MOSS?**

No measured improvement: F is 0/19 correct with 19/23 valid, exactly the same valid cases as H. Paired H→F has zero gains and zero losses; assisted onset/overlap metrics also remain 0/19. Both complete audio files are retained and the prompt supplies only the known construction interval, without the changed aspect. This oracle location assistance tests naming after attention is directed, not independent localization. It is separate from Hydrogen's selectively triggered cropped follow-ups, which raised Hydrogen's detection from 3/23 to 4/23; those are different task presentations and cannot supply a direct F comparison.

**5. Does MOSS merit product integration?**

No, on this evidence. It does not meet the registered 50% change-detection or 70% choice targets, its choice lower confidence bound is below mean chance (26.7%), and MUSE contour is 2/17 against a 75% target. Output-format failures and memory recovery add deployment burden without demonstrated listening benefit.

Across all 138 generated responses, MOSS call wall time is median **28.66 s** (bootstrap 95% interval 20.90–37.06 s), maximum **1,460.81 s**, total **11,045.55 s**. This includes failed outputs and audio preparation/generation, excludes model loading and the preceding deterministic measurements. The final near-null call reached 2,048 tokens and failed parsing after 24.35 minutes; it was preserved without retry. For the same 43 successful single-interaction choice tasks, median MOSS call time is **13.91 s**, versus Hydrogen's **13.71 s** provider latency; the timing boundaries differ. Hydrogen's often quoted 35.42 s is a median of **96 document-level sums**, not individual calls. Its saved producer IDs record **144 interactions** (111 primary and 33 automatic follow-ups); individual-call median across all 144 is unavailable from those sums.

The original valid-method batch had **40 response-free MPS OOM attempts**. The recovery policy, committed separately at `8d93940a2f3de702c0123c1053e36032f25b3ef3`, reran exactly those slots in fresh processes with unchanged inference, prompts and settings. It retained all generated failures. Thus there were **178 execution attempts** for 138 response slots, not 178 independent judgments. The 40 recovery workers took **3,714.72 s** including process/model-load and deterministic-check overhead. Their JSON records identify each retry and the immutable original files. The recovery source contained 265 original raw/case JSON files; all remain byte-for-byte unchanged.

Final-view sampled peaks are **193.51 GB MPS driver allocation**, **42.48 GB MPS tensor allocation**, and **15.42 GB process high-water RSS**. The original failed batch reached **216.11 GB driver allocation**. Driver figures include cached/in-flight allocation and overcommit; they are not physical resident usage and must not be added to RSS. These observations explain the fresh-process recovery burden on the 128 GB M4 Max. Verified weight storage is **18,119,304,629 bytes across 24 files**, plus approximately **647 MB** for the isolated venv. MOSS external-service charge is **$0**; energy and hardware amortisation were not measured. Hydrogen's saved token totals imply **$4.594** at `score.py`'s price assumptions, an estimate rather than an invoice; these whole-run totals cover different call counts and follow-up policies. No HF token was available, so HF Jobs was not used.

Numerical feasibility required a patch-free runtime workaround: BF16 MPS SDPA language prefill, then the public Transformers setter switches nested Qwen to eager cached decoding after the first finite logits. Encoder attention stays eager; all generation steps reject nonfinite logits. Pure SDPA produced 151,936/151,936 nonfinite logits at step 1. Fully eager MPS crashed on the longest pilot-source pair. A bounded FP32 CPU/SDPA full-pair probe completed in 316.45 s with 53.96 GB peak RSS. This is a feasibility probe, not a CPU holdout. Invalid `moss-holdout-1` and `-2` attempts are retained and excluded. The final inference method was frozen at `19d2bedc89712734167c77b6e077eef7f22fbd70`; no prompt, generation setting or scoring semantics changed during the valid decision run or resource recovery.

**6. Conclusion to record**

“Do not integrate MOSS-Music-8B-Instruct into Agent Listening on the present evidence. The pinned Transformers/MPS method shows isolated question-level gains but no verified improvement in independent change listening, localization or counterbalanced judgment, and substantially lower choice accuracy with incomplete structured-output coverage. Keep Hydrogen's Issue #13 capability limits explicit; retain deterministic checks and this reproducible negative evaluation. SGLang/CUDA capability and other prompt methods remain untested.”

**7. Code changes, tests and evidence**

Only evaluation files were added: `eval/moss.py` resolves corpus inputs, invokes the isolated model and persists raw calls; `eval/moss_score.py` reuses unchanged `score.py` for focused and matched reporting; `eval/moss_recover.py` isolates only response-free MPS OOM attempts while preserving all generated answers; `eval/test_moss.py` covers parsing, clip order/identity, scoring distinctions, input hash rejection and the recovery selector; `eval/moss-requirements.txt` pins the external runtime; `eval/moss-results.md` records method and results. Product code, dependencies/lockfile, schema, Skill and thresholds are unchanged. The user's untracked `Research/` is preserved.

The full required gate passed: `uv sync --locked`, Ruff lint/format, mypy, Import Linter (3 contracts), and **220 pytest tests**. The separate eval test command passed **9 tests**. The product wheel was not rebuilt because no product behavior changed. Real MOSS audio generation, dual-input processing, hash checks and partial-run recovery were exercised.

Evidence root: `/Users/dmus/.cache/agent-listening-eval/runs/` (all audio, weights and raw outputs remain outside Git).

- Original valid execution: `moss-holdout-3/`, including `method.json`, all 138 raw slot attempts, and the 40 response-free OOM failures.
- Final scored view: `moss-holdout-3-recovered/`: `comparison.json`, `summary-holdout.json`, `rows-holdout.json`, `recovery.json`, `integrity-after.json`, `raw/`, `focused/raw/`, and `identical-model/raw/`. Per-case documents accompany those raw files. Recovery kept original responses byte-for-byte and verifies all 142 corpus/source files unchanged.
- Baseline: `holdout-1/`; common-check view: `hydrogen-moss-common-checks/`. Both scored with unchanged `eval/score.py`; recheck made zero provider calls.
- Representative responses: `raw/bach10-03-mute__H__AB.json` (sameness claim on real edit), `raw/sc-canon-entry__H__AB.json` (`varies`, not required `differs`), `focused/raw/sc-canon-entry__F__AB.json` (focus supplies no detected change), `raw/sc-canon-nearnull__MC__AB.json` (misleading-context control), and `identical-model/raw/bach10-05-identical__IM__AB.json` (unparseable control).
- Numerical/runtime probes: `moss-feasibility/`, including `download-verification.json`, `logits-mps.json`, `long-pair-cpu.json`, `long-pair-mps-hybrid.json`; final pilot: `moss-pilot-6-hybrid/`. The original OOM attempts remain available rather than disappearing from deployment accounting.

Upstream source is pinned to `ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e`; model snapshot to `fce7f8304e96cc2d3398b8106456cbb2ecec3139`. Python 3.12.16, macOS torch/torchaudio 2.9.1, Transformers 4.57.1; exact resolved dependencies are committed. Input is 16 kHz mono, channel mean, no normalisation. The processor supports multiple explicit spans and 12.5 audio tokens/s; language context is 40,960 tokens, with actual input+output budget checked. Upstream gives no validated maximum audio duration; this corpus reaches 418.452 s per file and approximately 837 s per pair. Greedy decoding uses seed 0, no sampling, one beam, 2,048 output tokens. Bitwise cross-device determinism is unverified. The local upstream clone is clean and unpatched.


## Method fixed before the holdout

This evaluation asks whether MOSS-Music supplies independently checked musical listening gains over
the stored `gemini-4-hydrogen` holdout. The decision set is the existing holdout, including MUSE
(`AGENT_LISTENING_EVAL_PAID_TIER=1` enables those existing cases; inference here is local).
Development uses only pilot cases. The initial method commit preceded the first MOSS holdout attempt.
That SDPA attempt was terminated when numerical invalidity was demonstrated; a second start exposed
a nested-backend configuration error. Both are retained and excluded. The final inference method
was committed before the valid decision run. Resource recovery retries only attempts that produced
no response; every generated answer, including invalid JSON, remains final.
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
- H+C and MC: the same source files and misleading context as `eval/run.py`. Apart from that supplied
  context, no filenames, recipe metadata, ground-truth answers or expected change aspects are added.
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
or infinity, before argmax can turn invalid scores into repeated tokens. Generated responses are never
retried; there are no output repairs or answer extraction from prose.
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
The longest corpus source is 418.452 s; the longest pair is about 837 s combined. Processing these
inputs establishes corpus coverage, not a general maximum-duration guarantee.

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
AGENT_LISTENING_EVAL_PAID_TIER=1 "$AGENT_LISTENING_EVAL_DIR/moss-venv/bin/python" \
  eval/moss_recover.py --run moss-holdout-3 --out moss-holdout-3-recovered
AGENT_LISTENING_EVAL_PAID_TIER=1 uv run eval/score.py \
  --run moss-holdout-3-recovered --split holdout
uv run --with-editable . eval/recheck.py --run holdout-1 --out hydrogen-moss-common-checks
AGENT_LISTENING_EVAL_PAID_TIER=1 uv run eval/score.py \
  --run hydrogen-moss-common-checks --split holdout
AGENT_LISTENING_EVAL_PAID_TIER=1 uv run eval/moss_score.py \
  --run moss-holdout-3-recovered --baseline hydrogen-moss-common-checks
uv run --with scipy python -m pytest -q eval/test_moss.py
```

Set `AGENT_LISTENING_EVAL_DIR` to `/Users/dmus/.cache/agent-listening-eval` if it is not already set.
Do not use a new holdout run name to tune the method after observing decision results. An interrupted
run may resume the same directory; completed raw calls are reused, including parse failures.
