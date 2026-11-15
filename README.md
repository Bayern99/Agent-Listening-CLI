# Agent Listening

Agent Listening lets a composing agent hear real audio. It sends rendered or recorded audio to a native
audio model (`gemini-4-hydrogen` via Google's `gemini-api` CLI) and returns one versioned JSON evidence
document: time-located claims about sound events, phrases, voices, sections and whole works, each
cross-checked against deterministic signal measurements where possible. Its first integration is
[SC Agent IDE](https://github.com/Bayern99/sc-agent-ide) for SuperCollider composition.

## Install

Requirements: Python ≥ 3.11, [`gemini-api`](https://github.com/google-gemini/gemini-api-cli) 0.4.x on
`PATH`, and a Gemini API key (`GEMINI_API_KEY` or `gemini-api configure`).

```bash
uv tool install git+https://github.com/Bayern99/Agent-Listening-CLI
agent-listening doctor
```

## Commands

```bash
agent-listening hear render.wav --out hear.json
agent-listening ask render.wav "Which voice carries the melody? A. upper B. middle C. lower"
agent-listening compare before.wav after.wav --out diff.json
agent-listening investigate render.wav --focus 1:02-1:10 --question "Is the attack of the low voice clipped?"
agent-listening review render.wav --brief brief.json --before previous.wav
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
  `consistent` claims agree with a check, `unchecked` claims had nothing to check against. No model claim
  is ever "verified".
- `compare` measures where the signal changed (waveform and phase-invariant spectral differences) and, by
  default, re-hears up to two measured regions that no checked claim explains (`--follow-up N`).
- `review` listens without seeing the brief, then judges each goal from the checked claims in a separate
  text-only pass, answering `unclear` when no claim addresses a goal.
- The model hears a mono mix at reduced resolution with approximate timing; these limits are listed in each
  document's `limitations`.

## Measured limits (holdout evaluation, 2026-11-15)

Listening alone is not a reliable change detector: on 23 controlled single-voice edits it named the right kind
of change in 3 (13 %), while the deterministic difference measurement located all 23. Multiple-choice
perception questions were answered correctly 39/45 times (MUSE contour 18/18), and differences reported with
one clip order disappeared when the order was swapped. Treat model claims as hypotheses to check; details are in
[docs/architecture.md](docs/architecture.md).

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
