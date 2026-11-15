# AGENTS.md

Agent Listening turns real audio into time-located, checked listening evidence for composing agents.
SC Agent IDE is the primary consumer. Read [docs/architecture.md](docs/architecture.md) before changing
behaviour, contracts or dependencies; it holds the design, the acceptance criteria and the SC Agent IDE contract.

## Gate

Run before handing back any change:

```bash
uv sync --locked
uv run ruff check && uv run ruff format --check
uv run mypy src
uv run lint-imports
uv run pytest -q
```

A change is done when the gate is green and, for behaviour changes, the installed wheel was exercised
through the real CLI (`uv build`, install into a fresh venv, run the command on real audio).

## Extending the product

- One CLI, one evidence contract (`src/agent_listening/evidence-1.schema.json`), one Skill
  (`.agents/skills/agent-listening/SKILL.md`). A contract change bumps the version and updates the SC Agent
  IDE checker in the same release.
- Module boundaries are enforced by Import Linter (`pyproject.toml`) and `tests/test_rules.py`. When a rule
  blocks you, change the design, not the rule; when an AI-made mistake recurs, add the cheapest permanent
  check that would have caught it (API shape, type, rule test, unit test, then evaluation).
- Perception claims are judged by the evaluation in `eval/`, never by the listening model itself. Ground truth
  comes from original sources and construction recipes; audio stays in `$AGENT_LISTENING_EVAL_DIR`, outside git.
- Comments carry only protocol, math, licensing or platform facts the code cannot express.

## Issues and labels

Issues and verification records live in GitHub Issues/PRs for `Bayern99/Agent-Listening-CLI` (`gh`).
Triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.
Domain vocabulary: [docs/architecture.md §3.1](docs/architecture.md#31-concepts-the-whole-vocabulary).
