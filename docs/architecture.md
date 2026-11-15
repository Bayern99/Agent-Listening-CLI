# Agent Listening 1.0 — Architecture and Acceptance Criteria

This is the single authoritative design document. It was written from first principles before the
0.x implementation was audited; the legacy audit and migration live in the last two sections.
Executable checks (`tests/`, `pyproject.toml` lint and import contracts) outrank this prose.

## 1. Problem

A composing agent working in SuperCollider cannot hear. It needs, about real rendered audio:

1. perceptual descriptions grounded in time (what sounds where, at event, phrase, section and work scale);
2. answers to targeted questions;
3. what changed between two versions, and where;
4. whether a render meets explicit artistic goals and keeps protected musical relationships.

The dominant failure modes are not missing features but untrustworthy evidence: claims about
differences that do not exist, wrong timestamps, claims about properties the model cannot perceive,
analysis of the wrong audio version, and "hearing" what the source code says instead of what the audio
contains (source-context priming).

## 2. Primary-source findings

### 2.1 What the listening model actually receives

From the Gemini audio-understanding documentation (ai.google.dev/gemini-api/docs/audio, read 2026-11-15)
and the `gemini-api` v0.4.5 CLI (`--usage`, `generate --schema`, dry runs):

| Fact | Consequence |
| --- | --- |
| 32 audio tokens per second (≈31 ms per token); a 3 s probe used 91 audio tokens | Temporal resolution floor; cost ≈ 1,920 input tokens per audio minute. |
| Audio "downsampled to 16 Kbps" | High-frequency detail is not reliably available to the model. |
| Multi-channel audio is combined to one channel | The model cannot perceive stereo image; out-of-phase content can cancel. Spatial facts must come from deterministic measurement. |
| Inline requests ≤ ~14 MB media; larger files via `files upload` (48 h TTL) | Transport must choose inline vs upload and clean up uploads. |
| `AudioContent` has no offset/clipping fields | Focused listening requires physically excerpting audio and mapping times back. |
| `analyze` has no structured-output flag; `generate` accepts repeatable `-i` plus `--body` with `response_format` (JSON Schema subset) and `generation_config.thinking_level` (`minimal`/`low`/`medium`/`high` in v0.4.5) | Use `gemini-api generate` for all listening calls. |
| Exit codes 0 ok, 1 runtime/API, 2 usage, 3 auth; `--agent-mode` gives JSON errors on stderr | Adapter maps these without parsing prose. |
| Observed latency ≈ 90 s per short request (low and high thinking alike, 2026-11-15) | Listening is expensive in wall time; never run it implicitly, run evaluation cases concurrently. |

A first probe (440 Hz + 660 Hz sine dyad) was described as "F4 and A4" (actual A4 and E5):
absolute pitch naming must not be trusted without measurement.

### 2.2 Research on audio-language models and music

- **MUSE** (Carone, Román, Ripollés, arXiv:2510.19055; stimuli and human data on GitHub/OSF): 10 relational
  perception tasks with an N=200 human baseline. Gemini Pro matches non-experts on basic perception and
  instrument ID but trails experts on relational tasks (pitch-shift detection, contour, key modulation);
  chain-of-thought prompting is inconsistent. Relational claims need controlled tests, not trust.
- **MusTBench** (Sony, arXiv:2605.29300, MIT code): temporal grounding over full tracks. Published Gemini
  models reach only ≈55–60 % timestamp Hit@3s and ≈25–34 % interval IoU/F1. Model timestamps are approximate
  at the multi-second scale; localization must be measured and cross-checked, not assumed.

### 2.3 What leading experts would require that a naive spec omits

| Requirement | Missing capability it addresses | How it is measured | In 1.0? |
| --- | --- | --- | --- |
| Multi-timescale claims (event, phrase, section, work) | Flat tag lists lose form | Claims carry `scale`; long-form cases ask about returns and transformations | Yes |
| Relations between loci (repeats, varies, imitates, answers, doubles, contrasts, transforms, returns) without Western key/meter/song-section labels | Polyphony and form are relational | Ask cases with score-derived answers (voice entry order, canon, theme return) | Yes |
| Voices as perceptual streams, named by register/timbre rather than score parts | Mixed audio has no part labels | Voice-related multiple-choice questions on real ensembles | Yes |
| Score / performance / audio distinction | Source code describes intent, not sound | Every claim records `basis` (audio, context, both); with/without-context and misleading-context conditions | Yes |
| Free rhythm and non-12-TET tuning | Beat/key extractors impose wrong categories | Gamelan, guqin, shakuhachi, duduk cases; no key/meter fields in the contract | Yes |
| Timbre, articulation, resonance, gesture | Most MIR outputs ignore them | Controlled articulation and resonance edits with known spans | Yes |
| Negative controls and hallucination rates | Models invent differences | Bit-identical and near-null pairs; contradicted-claim accounting | Yes |
| Causal attribution of a perceived change | "Something changed" is not actionable | Controlled single-factor edits; aspect + location scoring | Yes |
| Human perceptual uncertainty and disagreement | A single label hides ambiguity | Protocol for blind human ABX / ITU-R BS.1116-style tests and inter-rater agreement | Protocol only; needs human listeners |
| Usefulness for revision | Correct description ≠ better music | Defect-repair pilot scored by known ground truth, never by the listening model | Pilot |

### 2.4 Original materials

Ground truth must come from original material, in this order of authority: source declarations and
construction recipes (exact), original stems and aligned scores (Bach10), published scores (Bach, *The Art
of Fugue*), recording metadata and established ethnomusicology (instrument, tuning system), then human
listening annotations. Model captions and earlier analysis outputs are never ground truth. Every corpus item
records URL, license, sha256 of the downloaded original, and, for constructed items, the exact recipe.
Audio lives outside the repository (`$AGENT_LISTENING_EVAL_DIR`, default `~/.cache/agent-listening-eval`).

Selected sources: Bach10 (Duan & Pardo; real separately-recorded violin/clarinet/saxophone/bassoon stems
with aligned notes; research use), Kimiko Ishizaka's *The Art of the Fugue* (CC0, archive.org), Wikimedia
Commons recordings of gamelan (slendro/pelog), guqin, shakuhachi and duduk (per-file licenses), MUSE
stimuli (research-only license, no training), and SuperCollider renders built from source snapshots in
`eval/sc/`. URMP, SALAMI, Saraga and MusTBench audio were examined and deferred: their size or access
conditions exceed the value for a bounded 1.0 suite (URMP ≈12.5 GB; Saraga requires large Zenodo/Dunya
downloads; SALAMI targets popular-song section labels; MusTBench audio is Slakh/MTG-Jamendo).

### 2.5 Methods and tools

| Tool / method | Role | Decision | Validation |
| --- | --- | --- | --- |
| `gemini-api` v0.4.5 + `gemini-4-hydrogen` | Native audio listening | Selected (pinned version, ≥7 days old) | Real calls in evaluation; adapter tested against a fake executable |
| numpy + soundfile (libsndfile) | Decode, hash, excerpt, deterministic measurements | Selected (small, BSD/LGPL) | Unit and property tests with synthetic signals |
| Controlled PCM transformations, metamorphic tests | Known-answer evaluation | Selected | Recipes are code in `eval/` |
| A/B order counterbalancing | Detect position bias in comparisons | Evaluation option | Order-swap agreement rate |
| Human ABX / ITU-R BS.1116 | Perceptual ground truth for subtle changes | Protocol only | Requires recruited listeners |
| MUSE, MusTBench metrics (Hit@3s, IoU) | External comparability | MUSE subset run; MusTBench metric definitions reused | Published baselines |
| mir_eval | Onset/segment metrics | Not selected: the needed metrics (absolute error, Hit@k, IoU) are a few lines | — |
| Partitura, music21, Sync Toolbox | Score parsing and score-audio alignment | Deferred: no 1.0 task needs automatic alignment; score facts are hand-derived for a few cases | Revisit when a task needs aligned symbolic evidence |
| Essentia, all-in-one, Basic Pitch, Demucs | Beat/key/section labels, transcription, separation | Not selected: heavy ML stacks, Western-centric categories; no measured benefit for the five tasks | Re-admit only with an evaluation showing improvement |
| Hypothesis | Property and stateful tests | Selected | — |
| Ruff, mypy, Import Linter, AST rule tests | Static enforcement | Selected | CI |
| mutmut | Mutation testing of the evidence core | Selected (manual run) | Mutation score reported |
| Formal model checking | Bounded state machines | Not selected: the only state machine (claim status) is a pure function over a finite domain and is tested exhaustively | — |

## 3. Architecture

### 3.1 Concepts (the whole vocabulary)

- **Input**: an audio file identified by the sha256 of its bytes, plus `pcm_sha256` (hash of decoded
  samples, equal across lossless containers) and format facts. Identity is computed once and re-checked
  immediately before any derived file is made.
- **Transport**: a derived file the model actually hears (lossless FLAC, optionally an excerpt), recorded
  with source hash, own hash, offset and parameters. Originals are opened read-only; derived files live in a
  content-addressed cache.
- **Context**: symbolic or textual material supplied by the caller (SuperCollider source, score text,
  notes), recorded with its hash.
- **Brief**: human goals and protected relationships for `review`.
- **Measurement**: a deterministic result (PCM identity, signal difference regions, level, spectrum,
  stereo, onsets) with method and parameters.
- **Claim**: one statement with `origin` (model, measurement, human), `aspect`, `scale`, `loci`
  (input, start, end, optional voice), optional `relation`, `basis`, `confidence`, deterministic `checks`
  and a derived `status`.
- **Evidence**: the one versioned document every command emits.

Claim `status` is `unchecked`, `consistent` (all applicable deterministic checks passed), `ungrounded`
(no valid locus) or `contradicted` (deterministic evidence disagrees). A model claim is never "verified";
that word is reserved for artifact identity.

### 3.2 Tasks and CLI

One executable, `agent-listening`. Listening commands call the model once (twice with
`--counterbalance`); deterministic commands never do.

| Command | Purpose |
| --- | --- |
| `hear AUDIO [--focus S-E] [--context F]...` | Perceptual description across scales |
| `ask AUDIO QUESTION [--focus S-E] [--context F]...` | Targeted question with supporting claims |
| `compare A B [--question Q] [--context F]... [--counterbalance]` | Differences and preserved relations between versions |
| `investigate AUDIO --focus S-E [--measure level,spectrum,stereo,onsets] [QUESTION]` | Zoomed listening plus deterministic measurements |
| `review AUDIO --brief BRIEF.json [--before AUDIO] [--context F]...` | Goal and protected-relation assessments |
| `identify AUDIO...` | Identity and format only |
| `check EVIDENCE [--audio PATH]...` | Re-hash inputs and validate an evidence file |
| `doctor` | Provider CLI version and credential presence (never values) |

Common options: `--out PATH` (no-clobber; `--overwrite` replaces), `--model` (default `gemini-4-hydrogen`),
`--thinking`, `--dry-run` (identity, transport and measurements; no model call).
Exit codes: 0 complete, 1 failure (identity, input, internal), 2 usage, 3 provider unavailable or failed
(evidence is still emitted with `status: failed`).

`compare` computes PCM identity first; bit-identical inputs are reported as identical without a model call.
When inputs share a sample rate it measures signal-difference regions (cross-correlation alignment,
blockwise residual-to-signal ratio) and contradicts model difference claims that fall outside every measured
region beyond a tolerance. Global changes make the check inapplicable rather than permissive.

Spectral difference (added after a real SC revision, organ pedal bass "pure" to "drawbars", was
heard as identical while the waveform residual was global and therefore inapplicable): the waveform residual
is phase-sensitive, so a re-rendered oscillator with a new start phase "differs" without sounding different.
`signal-difference` therefore also reports phase-invariant spectral regions (log-magnitude on a 1/12-octave
grid). "unchanged" claims are contradicted by spectral regions, including global ones; "differs" claims are
supported by waveform or spectral regions and become inapplicable only when both are global.

Measurement-guided follow-up (added after the pilot, where whole-clip comparison missed audible
articulation and resonance edits and answered "no perceptible difference"): measured difference regions that
no `consistent` difference claim covers are re-heard as matching A/B excerpts with the region pointed out
(`--follow-up N`, default 2 regions, 0 disables). Those claims carry an informational `measurement-guided`
check: their location was primed by measurement, so only their description (what changed, in which voice)
is model evidence.

### 3.3 Modules and dependency rules

```text
cli ──► tasks ──► audio ────┐
             ├──► prompts ──┼──► evidence   (stdlib only)
             └──► gemini    │   (stdlib only, no internal imports)
```

| Module | Owns | May import |
| --- | --- | --- |
| `evidence` | Contract types, validation, time mapping, grounding checks, serialization | stdlib |
| `audio` | Identity, decoding, transport derivation, deterministic measurements | `evidence`, numpy, soundfile |
| `prompts` | Listening instructions and model response schemas | `evidence` |
| `gemini` | The only `gemini-api` subprocess adapter | stdlib |
| `tasks` | Orchestration of the five tasks | all of the above |
| `cli` | Argument parsing, output writing, exit codes | `tasks`, `evidence` (lazy) |

Enforced by Import Linter contracts and AST tests: only `gemini` imports `subprocess`; only `tasks`
imports `gemini`; numpy/soundfile are never imported at module import time by `cli`; no heavyweight ML
packages anywhere; no `global` statements or module-level mutable containers; file writes only in the
cache writer and the evidence writer; no imports of removed 0.x modules; no narrative comments.

### 3.4 Evidence contract `agent-listening/evidence` version 1

Published as `src/agent_listening/evidence-1.schema.json`. Times are seconds in the original input's time
base (excerpt offsets are mapped back by `evidence`). Top-level fields: `schema`, `version`, `task`,
`status`, `tool`, `created_at`, `inputs`, `context`, `brief`, `request`, `transport`, `measurements`,
`claims`, `assessments`, `answer`, `summary`, `producer`, `limitations`, `error`.

Automatic limitations: `MODEL_INPUT_MONO` (multichannel input), `MODEL_INPUT_REDUCED_RESOLUTION`,
`MODEL_TIME_APPROXIMATE`, `CONTEXT_SUPPLIED` (claims may reflect context), `DIFFERENCE_CHECK_UNAVAILABLE`.

## 4. Acceptance criteria and evaluation protocol

Four independent groups. Group 1 and 4 thresholds are fixed now. Group 2 and 3 targets are calibrated on a
pilot split, committed to `eval/thresholds.json`, and only then is the holdout split run. Results report
Wilson 95 % intervals for proportions, bootstrap intervals for medians, and Newcombe intervals for
differences; failures and disagreements are listed, not averaged away.

**1. Transport and identity (deterministic, must all hold)**
- Every evidence input `sha256` equals the bytes of the file analysed (100 %).
- A file modified between identification and transport is rejected (100 % of fault-injection cases).
- `pcm_sha256` is invariant across WAV↔FLAC re-encoding of the same samples (property tests).
- Originals are byte-identical and unmodified after the full evaluation run (100 %).
- Updated SC Agent IDE checker: verified for matching evidence, failed for any hash mismatch (100 %).

**2. Perception and temporal grounding (Hydrogen, holdout split)**
- Bit-identical pairs: reported identical with zero model calls (100 %).
- Near-null pairs (−100 dBFS noise): raw model difference-claim rate reported; difference claims left
  `consistent` after checks = 0.
- Controlled single-factor changes (mute, timing, pitch, articulation, resonance, balance): aspect-correct
  detection rate; localization Hit@3s of the change onset; median absolute onset error; false-positive
  difference claims per case before and after deterministic checks.
- Multiple-choice relational and voice questions (scored by option letter): accuracy against chance.
- MUSE subset accuracy against chance and the published human baseline.
- Source context: with-context vs without-context detection, and misleading-context hallucination rate
  (context declares a change the audio does not contain).

**3. Revision usefulness (pilot)**
- Defect-repair tasks on SuperCollider renders with a known defect and a goal. A reviser (text-only call)
  sees code and goal, with or without `review` evidence. Success = the defect parameter moves into its known
  target range without regressing others, judged by the construction recipe, never by the listening model.
  Reported as a difference in success rates with interval; small N is stated, not hidden.

**4. Software, cost and growth**
- Ruff, mypy (strict on the package), Import Linter, pytest (unit, property, rule, CLI) all pass.
- Mutation score on `evidence` ≥ 80 %.
- `agent-listening --help` loads neither numpy nor soundfile.
- `compare` of two 10-minute 48 kHz stereo files peaks below 400 MB RSS.
- Wheel < 150 KB; tracked files fewer than 0.2.0.
- Per-request latency, input/audio/thought tokens and cost per audio minute recorded from `usage`.

Human blind listening (ABX for difference claims, BS.1116-style hidden-reference grading for subtle
changes, Krippendorff's α across raters) is specified here but not executed in 1.0: it needs recruited
listeners. Engineering tests passing does not show that music got better.

## 5. Legacy audit (0.2.0, performed after sections 1–4)

0.2.0 was an offline pipeline (`analyze`/`build-ir`/`doctor`) that fused Essentia, all-in-one, Demucs and
Basic Pitch outputs into JAMS plus a "Music IR" with key, chords, beats and song sections. It had no
listening model, no comparison, no question answering and no goal review.

| Component | Decision | Reason |
| --- | --- | --- |
| `src/` (`core`, `cli`, `fusion/builder`, adapters, `resources`) | Delete | Different product; heavy ML dependencies; Western-centric categories; package named `src` |
| `schemas/music-ir-*.json`, `profiles/`, JAMS output, demo `jams/` and `music-ir/` | Delete | Replaced by evidence contract v1 |
| `tests/` and fixtures | Replace | Tested the removed adapters |
| `CONTEXT.md`, `docs/adr/`, `docs/field-guide.md`, `docs/review-checklist.md`, `.scratch/`, tracked `Research/`, `README.zh-CN.md` | Delete | Superseded; history keeps them; one README and this document replace them |
| `docs/agents/*` | Replace | Folded into `AGENTS.md` |
| `bin/agent-listening`, `MANIFEST.in` | Delete | Installed entry point and uv build backend replace them |
| `pyproject.toml`, `uv.lock`, CI, `AGENTS.md`, `README.md`, `CREDITS.md`, Skill | Replace | New dependency set and product |
| `LICENSE` (MIT), `CHANGELOG.md` | Keep | Still true; changelog gains 1.0 |
| sha256 source identity, no-clobber output | Re-derived independently in §3, no code reused | — |
| `agent-listening/0.2` receipt + `music-ir/0.2` consumed by SC Agent IDE | External compatibility contract | See §6 |

Untracked local research notes and generated outputs in the working tree are user files and are left in
place, uncommitted.

## 6. SC Agent IDE integration contract

Ownership: SC Agent IDE owns SuperCollider execution, render runs, source revisions, Music Memory, grants
and human acceptance. Agent Listening owns listening and audio evidence. The only coupling is files:
the agent runs `agent-listening` on an artifact from a render run and the IDE's helper checks the result.
No MCP server, runtime or provider dependency is added to the IDE.

Caller audit: the only machine consumer is `skills/sc-agent-ide-grounded/helpers/check-analysis.mjs`
(with `test/check_analysis.test.ts`); `references/revision-loop.md` documents the 0.2 commands for agents.

New flow (SC Agent IDE with the coordinated update):

```bash
agent-listening review "$RUN/artifacts/out.wav" --brief brief.json \
  --before "$PREV_RUN/artifacts/out.wav" \
  --out "$PROJECT/analysis/$RUN_ID/review.evidence.json"
node helpers/check-analysis.mjs --project-root "$PROJECT" --run "$RUN_ID" \
  --artifact "sha256:<hash>" --analysis-root "$PROJECT/analysis/$RUN_ID" \
  --evidence "$PROJECT/analysis/$RUN_ID/review.evidence.json"
```

Checker rules for `--evidence`: `schema == "agent-listening/evidence"`, `version == 1`,
`status != "failed"`; the input whose `role` is `subject` (or `after` for `compare`) must have `sha256`
equal to both the manifest hash and the freshly computed artifact hash, otherwise `SOURCE_HASH_MISMATCH`;
a match only on a `before` input fails with `ARTIFACT_ROLE_MISMATCH`. Evidence `limitations` pass through;
the result keeps the `sc-agent-ide/analysis-check` v1 shape with `versions.evidence`. The `--receipt`
path for 0.2 stays byte-for-byte unchanged.

Migration: released SC Agent IDE ≤ 1.9.1 only understands 0.2 receipts. Users of those releases keep
`agent-listening` 0.2.0 (`uv tool install git+https://github.com/Bayern99/Agent-Listening-CLI@v0.2.0`);
with 1.0 installed the old checker fails loudly (`RECEIPT_VERSION_UNKNOWN`, exit 1), never silently.
The coordinated SC Agent IDE change accepts both formats, so upgrading the IDE first is always safe.
