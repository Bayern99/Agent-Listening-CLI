# Agent Listening 1.0 — Architecture and Acceptance Criteria

This is the single authoritative design document and the place where results are interpreted. It was
written from first principles before the 0.x implementation was audited; the legacy audit and migration
live in the last two sections. Executable checks (`tests/`, `pyproject.toml` lint and import contracts)
outrank this prose.

Each statement here is one of four kinds, and the sections keep them apart: long-term aim (§1),
implemented behaviour (§3), current evidence (§4.1 to §4.3), open question (§4.4).

| Line | State |
| --- | --- |
| Research baseline | branch `agent-listening-1.0` at `a58f45581ff9dad441b137a425fc38484bade6d6` (package version 1.0.0, draft PR #10). All results in §4 were produced on this line. |
| Mainline | `main` at `b3ae2b2d5eccc19b0fc3fc579be4bc8f07918852`, still the 0.2.0 offline pipeline. |
| Published release | `v0.2.0` only. 1.0.0 has no tag and no release. |

## 1. Product and problem

Agent Listening is a standalone music listening tool for agents that compose and analyse music. A caller
puts a question to real audio and gets back auditory observations that carry a time position, can be
followed up, can be compared, and can be checked against other evidence. Their purpose is to help the
caller form and revise musical choices. SC Agent IDE is the first important integration, not the
product boundary.

Long-term scope: sound events, voices and texture, the continuation and transformation of material,
sections and long-range structure. Instrumental, polyphonic, electroacoustic, free-rhythm and non-12-TET
material are all working objects. Audio can be the only input; source code, scores, target material and
neutral text hints are supplied when the question needs them. The model, commands and schema described
below are the current implementation, and a well-founded product finding may change them.

Current focus: make observations and answers usable in concrete musical tasks, and find out how sound
information changes the caller's next decision. The five tasks keep their purposes:

1. `hear`: perceptual description grounded in time (event, phrase, section and work scale);
2. `ask`: an answer to a targeted question;
3. `compare`: what changed between two versions, and where;
4. `investigate`: a close look at one span, with measurements;
5. `review`: whether a render meets explicit artistic goals and keeps protected musical relationships.

Multiple-choice questions, difference localization and controlled-edit evaluation each serve part of this
problem. They are instruments, not the definition of the product, which keeps exploring wider musical
relations and material understanding.

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
| First probes took ≈ 90 s per short request (low and high thinking alike, 2026-11-15); the holdout later measured much shorter calls (§4.1 latency) | Listening costs wall time and varies between runs; never run it implicitly, run evaluation cases concurrently. |

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
| MOSS-Music-8B-Instruct, frozen Transformers/MPS method | Second listening model | Not in the product (§4.3) | `eval/moss-results.md`; runtime pinned in `eval/moss-requirements.txt`, outside product dependencies |
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
- **Brief**: human goals and protected relationships for `review`. The contract records the brief's path
  and hash but no author, so attribution is the caller's job: a brief holds what a person stated, and a
  creative hypothesis proposed by an agent stays labelled as the agent's (as an `ask` or `compare` question,
  or in the caller's own records).
- **Measurement**: a deterministic result (PCM identity, signal difference regions, level, spectrum,
  stereo, onsets) with method and parameters.
- **Claim**: one statement with `origin` (model, measurement, human), `aspect`, `scale`, `loci`
  (input, start, end, optional voice), optional `relation`, `basis`, `confidence`, deterministic `checks`
  and a derived `status`.
- **Evidence**: the one versioned document every command emits.

Claim `status` is `unchecked`, `consistent` (all applicable deterministic checks passed), `ungrounded`
(no valid locus) or `contradicted` (deterministic evidence disagrees). A model claim is never "verified";
that word is reserved for artifact identity.

`consistent` means only that the claim passed the checks that applied to it. The checks test whether a
claimed difference or sameness sits where the signal did or did not change; they do not test what the
claim says about pitch, timbre, voice or form. A `measurement-guided` check means a locating cue was
given to the model before it spoke: the location is the measurement's, and the semantic content is used
to the degree something else supports it (§4.2).

Evidence is chosen by the question, and no kind substitutes for another:

| Question | Evidence that answers it |
| --- | --- |
| How was it built, what was intended? | Source code and construction recipe |
| What was actually rendered, where did the signal change? | The audio itself and deterministic measurements |
| How does it sound, what is perceived? | Listening observations (model claims, and human listening where available) |
| Where should the music go? | The creator |

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
grid). "unchanged" claims are contradicted by spectral regions (including global ones) and by local, non-global
waveform regions; a phase-only re-render shows up as a global waveform difference, which never contradicts.
On the stored holdout claims, spectral-only contradiction caught 174 of 419 "unchanged" claims lying over real
edits, the combined rule is required to keep the waveform rule's coverage. "differs" claims need waveform or
spectral regions covering min(25 % of the locus, 0.5 s), with waveform windows counted only when the residual
exceeds -80 dBFS, and become inapplicable only when both region sets are global.

Goal-blind review (added after the revision pilot's negative control): when the goals were in the
listening prompt, Hydrogen reported every goal as violated on both the defective and the corrected renders
(5/5 defects flagged, 5/5 corrected renders falsely flagged; for example "a clear whistling resonance" on a
render with the resonance removed). `review` therefore listens first without seeing the brief (`hear`, or
`compare` with `--before`), applies the deterministic checks, and then asks a text-only assessment call to
judge each goal from the non-contradicted claims only, answering `unclear` when no claim addresses a goal.
That is the implemented behaviour: goals and protected relationships never reach a listening prompt
(enforced in `tests/test_rules.py`); `--context` files do. Goal-blind listening halved the false alarms and
also the detections (§4.1), so it is a correction of one failure, not a solved review. Any other strategy,
such as goal-derived neutral questions or focused re-listening of the spans a goal names, is a proposal
until its task is implemented and tested against the corrected-render control.

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

Evaluation and use are separate. The holdout split has been exposed: its results were read, then used to
change the checks (`cc04df4`, `5fdb9d6`, re-applied to stored claims with `eval/recheck.py`) and to compare a second model. It now serves reproduction,
diagnosis and regression. A claim about capability on unseen material needs works or sources that
development has not touched. In real composition the caller supplies whatever context the question needs
(§3.1); the no-context conditions below isolate what the audio alone yields and are not a usage rule.

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

### 4.1 Results (2026-11-15, gemini-4-hydrogen via gemini-api 0.4.5)

Thresholds were committed in `eval/thresholds.json` (f5b2aa1) after pilot-3 and before the holdout. The holdout
(100 runs, 96 model calls) used the wheel built from 1b05b5e; "final checks" re-applies the current deterministic
checks to the same stored claims with `eval/recheck.py` (no new model calls). Intervals are Wilson 95 %.

| Measure | Result |
| --- | --- |
| Bit-identical pairs | 4/4 reported identical, 0 model calls |
| Near-null pairs, hallucinated differences | 0/5 cases |
| Misleading source context (claims a detune that is absent) | 2 hallucinated difference claims in 1/1 case; survived the 1b05b5e checks (hard criterion failed), contradicted by the final checks |
| Controlled edits, a `differs` claim whose `aspect` is in the edit's allowed set | 3/23 = 13 % [5, 32]; mute 1/3, timing 1/3, pitch 1/5, articulation 0/3, resonance 0/5, balance, dynamics, entry, removed theme return 0/1 each |
| Controlled edits, any `differs` claim overlapping the true interval ±3 s, any aspect | 7/23 = 30 % [16, 51] |
| Earliest measured difference region starts within 3 s of the true onset | 23/23 |
| With source context (SC cases) | 0/3 without context, 3/3 with (allowed-aspect match), but the misleading-context control shows context alone produces such claims |
| Counterbalanced order, difference claims reproduced after swapping | 0/10 |
| "Unchanged" claims lying over real edits contradicted | 379/419 (1b05b5e), 385/419 final; 0/47 on near-null pairs; 20/20 true difference claims kept |
| Multiple-choice perception questions | 39/45 = 87 % [74, 94], chance 27 %; MUSE contour 18/18 (zero-shot single turn, not the paper's chat protocol); Bach10 instrument count 3/3, highest/lowest instrument 3/6, articulation in a focused passage 1/3; gamelan tuning, pulse and instrument families 7/7; fugue entry order, textures and SC theme returns 6/6; voice count 1/2 |
| `hear` on long-form and non-Western recordings | 4/4 documents valid with every claim grounded; 53 claims, 30 relations (5-9 per document). All four timescales present in 4/4 as first recorded here, 3/4 in the later tabulation in `eval/moss-results.md`; the stored run decides, see §4.4 |
| Revision pilot (5 SC defects × 3) | code only 15/15, code + listening 15/15, difference 0 [-0.20, 0.20]: ceiling, not discriminative |
| Goal review on defective vs corrected renders | goals in the listening prompt: 5/5 flagged, 5/5 corrected falsely flagged; goal-blind: 2/5 flagged, 2/5 falsely flagged |
| Latency | 35.42 s is the median of 96 document-level sums, not of calls: those documents hold 144 interactions (111 primary, 33 automatic follow-ups), and a per-call median over all 144 cannot be recovered from the stored sums. The 43 single-interaction choice tasks have a median of 13.71 s |
| Cost | US$0.027 per audio minute and US$4.59 for the holdout, estimated from saved token counts at `eval/score.py`'s price assumptions, not invoiced |

### 4.2 How to read these results

The numbers above are kept as recorded. What each one supports:

- **3/23 is an allowed-aspect match.** A case counts when the model emitted a `differs` claim whose `aspect`
  label belongs to the set registered for that edit kind in `eval/corpus.py` (`CHANGE_ASPECTS`; for a mute:
  texture, voice, dynamics or timbre). Nobody graded whether the claim's prose describes the edit, so 3/23 is
  neither a semantic accuracy nor a perception rate. With automatic follow-ups counted it is 4/23, and those
  claims are `measurement-guided`.
- **23/23 is onset localization by measurement.** The start of the earliest measured difference region lies
  within 3 s of the recipe's onset. It says nothing about the end of the change, about later regions, or about
  what changed, and it involves no model.
- **The 23 edits are several constructions, not one.** 15 are single-instrument edits inside a four-stem Bach10
  remix (mute, timing, pitch, articulation, resonance, three pieces each); 4 are span edits applied to a whole
  recording (+100 cents and −10 dB on *The Art of Fugue*, +40 cents on duduk, an added resonance on shakuhachi);
  3 are SuperCollider parameter changes (entry delay, filter resonance, balance); 1 removes a theme return from a
  long-form SuperCollider piece. The pooled rate mixes these; per-kind cells have 1 to 5 cases and support no
  per-kind conclusion.
- **15/15 against 15/15 is parameter repair by a text agent.** The reviser is a text-only model call that sees
  the source and the goal and proposes parameters; success is the defect parameter landing in its registered
  range. The defect was recoverable from the code and the goal alone, so the pilot cannot show whether listening
  helps and says nothing about revision of music whose problem is only audible.
- **`hear` validity and coverage are not correctness.** Grounded loci, timescales present and relation counts
  describe the form of the document. No caption was graded against a reference or by listeners.
- **39/45 is letter accuracy on closed questions**, zero-shot, on this corpus, against a mean chance of 27 %.
  Sub-families are small (1 to 18 cases) and uneven: MUSE contour 18/18, highest or lowest instrument 3/6.
- **0/10 is a claim count.** Ten `differs` claims from 6 of the 15 counterbalanced cases were eligible, and none
  reappeared with the clip order swapped. Cases without a difference claim contribute nothing.
- **385/419 and 0/47 measure the checks,** not the model: how often the deterministic rule contradicts an
  "unchanged" claim lying over a real edit, and how often it wrongly does so on near-null pairs.

Supported by this evidence: the measurement locates where a signal changed; closed questions about a whole
recording or a focused span are often answered correctly; whole-recording descriptions are well-formed and
located. Not supported: that the model names what changed in a comparison, that its difference claims are
stable under order, that goal verdicts separate defective from corrected renders, or that listening evidence
changes a caller's decision. The last has not been tested with a task able to show it.

### 4.3 MOSS-Music: decision and negative result

The frozen MOSS-Music-8B-Instruct method (upstream `ad107c7`, weights `fce7f83`, Transformers 4.57.1 on Apple
MPS, greedy decoding, method frozen at `19d2bed`) does not enter the product. This is a decision under this
round's method and evidence, recorded in full in [`eval/moss-results.md`](../eval/moss-results.md):

- Choice accuracy 14/43 valid (33 %, 43/45 covered) against Hydrogen's 38/43 on the same cases; three MOSS-only
  wins and no difference claim on the misleading-context pair.
- Allowed-aspect match 0/19 valid (19/23 covered). No MOSS comparison claim used the `differs` relation, so the
  zero includes failure to use the required protocol and does not show that the prose lacked content.
- 26 of 138 responses failed JSON or schema validation, 40 response-free memory failures needed fresh-process
  recovery, and sampled MPS driver allocation peaked at 193.51 GB on a 128 GB machine.

The result is not the model's intrinsic limit. A new model version, a different runtime (SGLang/CUDA was not
tested), a different prompt or output method, or evidence from a specific task are each a reason to evaluate
again. Research trials stay separate from product dependencies: the MOSS runtime is pinned in
`eval/moss-requirements.txt`, runs in its own interpreter, and nothing under `src/` imports it.

### 4.4 Open questions

Exploration stays open: new models, signal methods, ways of supplying context, task designs and outside music
research may explain a real failure, reveal a musical use nobody planned for, and change the next question or
the current priority. A proposal for further investment states which musical question it helps, what it may
add over the existing method, and what it costs to use.

1. **Does sound information change the caller's next decision?** Untested. The revision pilot was at ceiling
   without listening. This is the current priority; it needs tasks in which the deciding fact is audible and
   absent from the source (interaction of voices, masking, the rendered result of a nonlinear or stochastic
   process, a recording with no source at all), on material development has not touched.
2. **What do free-form observations get right?** `hear` and `investigate` content has no correctness grade.
   Grading needs reference facts from scores and recipes, or listeners.
3. **Which context helps which question?** Context raised the allowed-aspect match on three cases and produced
   two false claims on one control. Target material and neutral text hints have not been tried.
4. **Review beyond goal-blind listening.** See the proposal status in §3.2.
5. **Relations and long-range structure.** Coverage so far is a handful of closed questions (fugue entries,
   theme returns, texture); continuation and transformation of material have no task of their own.
6. **Human listening.** Protocol below, not executed.
7. **Record keeping.** The `hear` timescale count differs between this document and `eval/moss-results.md`
   (4/4 against 3/4); recount from the stored `holdout-1` run before quoting either.

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

Migration: every SC Agent IDE tag up to `v1.12.1-rc.1` and its `main` (package version 1.12.1 at `0aec06b`,
checked 2026-11-21) only understand 0.2 receipts. Users of those versions keep `agent-listening` 0.2.0
(`uv tool install git+https://github.com/Bayern99/Agent-Listening-CLI@v0.2.0`); with 1.0 installed the old
checker fails loudly (`RECEIPT_VERSION_UNKNOWN`, exit 1), never silently. The `--evidence` path exists only
on the unmerged SC Agent IDE branch `agent-listening-1.0-evidence` (`63ab0f8`). That change accepts both
formats, so upgrading the IDE first is always safe.
