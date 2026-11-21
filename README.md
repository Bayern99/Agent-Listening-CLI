# Agent Listening

Agent Listening is a standalone music listening tool for agents that compose and analyse music. An agent
puts a question to real audio and gets auditory observations that carry a time position, can be followed
up, can be compared, and can be checked against other evidence, so that it can form and revise musical
choices. [SC Agent IDE](https://github.com/Bayern99/sc-agent-ide) for SuperCollider composition is the
first important integration.

The current implementation sends rendered or recorded audio to a native audio model (`gemini-4-hydrogen`
via Google's `gemini-api` CLI) and returns one versioned JSON evidence document: time-located claims about
sound events, phrases, voices, sections and whole works, checked against deterministic signal measurements
where a check applies. Audio alone is enough; source code, score text and other notes are added with
`--context` when the question needs them.

## Status

| Line | State |
| --- | --- |
| This branch, `agent-listening-1.0` | 1.0.0, unreleased research baseline (`a58f455`), draft PR #10. Everything below describes it. |
| `main` | 0.2.0, the earlier offline analysis pipeline. |
| Published release | `v0.2.0` only. |

## Install

Requirements: Python ≥ 3.11, [`gemini-api`](https://github.com/google-gemini/gemini-api-cli) 0.4.x on
`PATH`, and a Gemini API key (`GEMINI_API_KEY` or `gemini-api configure`).

```bash
uv tool install git+https://github.com/Bayern99/Agent-Listening-CLI@agent-listening-1.0
agent-listening doctor
```

Without `@agent-listening-1.0` the command installs `main`, which is 0.2.0 and has none of the commands
below. Released SC Agent IDE versions read only 0.2 receipts; see
[docs/architecture.md §6](docs/architecture.md#6-sc-agent-ide-integration-contract).

## Commands

```bash
agent-listening hear render.wav --out hear.json                     # what is in this audio, and where
agent-listening ask render.wav "Which voice carries the melody? A. upper B. middle C. lower"
agent-listening compare before.wav after.wav --out diff.json        # what changed, and where
agent-listening investigate render.wav --focus 1:02-1:10 --question "Is the attack of the low voice clipped?"
agent-listening review render.wav --brief brief.json --before previous.wav   # goals and protected relations
agent-listening identify render.wav          # identity only, no model call
agent-listening check diff.json              # re-hash the audio an evidence file refers to
```

Every listening command accepts `--context FILE` (source or score text), `--model`, `--thinking`,
`--dry-run`, `--out` and `--overwrite`. Exit codes: 0 complete, 1 input/identity failure, 2 usage,
3 provider unavailable or failed (evidence is still written with `status: failed`).

## What the evidence means

- Inputs are identified by sha256 of their bytes and of their decoded samples; the model hears a lossless
  derived copy, never a modified original.
- Claim `status` is derived from deterministic checks: `contradicted` claims disagree with the signal,
  `consistent` claims passed the checks that applied, `unchecked` claims had nothing to check against. The
  checks test where the signal changed, not what a claim says about the music. No model claim is ever
  "verified".
- `compare` measures where the signal changed (waveform and phase-invariant spectral differences) and, by
  default, re-hears up to two measured regions that no checked claim explains (`--follow-up N`). Those
  claims are marked `measurement-guided`: the location came from the measurement, the description is the
  model's.
- `review` listens without seeing the brief, then judges each goal from the checked claims in a separate
  text-only pass, answering `unclear` when no claim addresses a goal.
- The model hears a mono mix at reduced resolution with approximate timing; these limits are listed in each
  document's `limitations`.

## Current evidence (holdout evaluation, 2026-11-15)

- Closed questions: 39/45 option letters correct (MUSE contour 18/18; highest or lowest instrument 3/6).
- Comparison: on 23 controlled edits of several constructions, 3 produced a difference claim whose aspect
  label was in the set allowed for that edit. This is a label match, not graded semantic accuracy. The
  deterministic measurement placed the earliest difference within 3 s of the true onset in all 23.
- Order: none of 10 difference claims reappeared when the two clips were swapped.
- `hear`: documents were valid and fully located. Their content has not been graded for correctness.
- Revision: a text agent repaired a known parameter defect 15/15 times with listening evidence and 15/15
  without it. Whether sound information changes a caller's decision is untested, and is the current focus.
- A second model, MOSS-Music-8B-Instruct under one frozen Transformers/MPS method, was evaluated and is not
  in the product ([eval/moss-results.md](eval/moss-results.md)).

The holdout has been seen during development, so it now serves reproduction and regression; it does not
support claims about unseen material. Treat model claims as hypotheses to check. Interpretation, costs and
open questions are in [docs/architecture.md §4](docs/architecture.md#4-acceptance-criteria-and-evaluation-protocol).

The design, acceptance criteria, evaluation protocol and SC Agent IDE contract are in
[docs/architecture.md](docs/architecture.md). Agent usage lives in the Skill at
[.agents/skills/agent-listening/SKILL.md](.agents/skills/agent-listening/SKILL.md).

## Development

```bash
uv sync --locked
uv run ruff check && uv run ruff format --check && uv run mypy src && uv run lint-imports && uv run pytest -q
```

The evaluation suite (`eval/`) builds its corpus outside the repository; see `eval/corpus.py`.

## License

MIT. Third-party notices: [CREDITS.md](CREDITS.md).
