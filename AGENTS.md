# AGENTS.md

Agent Listening is a standalone listening tool for agents that compose and analyse music: questions put to
real audio, answered with time-located observations that can be followed up, compared and checked.
SC Agent IDE is the first integration. Read [docs/architecture.md](docs/architecture.md) before changing
behaviour, contracts or dependencies; it is the authority for design and for what results mean.

| For | Read |
| --- | --- |
| Product aim, implemented behaviour, results and their interpretation, open questions, SC Agent IDE contract | `docs/architecture.md` |
| Product identity, commands, current status | `README.md` |
| How a caller picks the question, context and time range | `.agents/skills/agent-listening/SKILL.md` |
| MOSS-Music negative result | `eval/moss-results.md` |

This branch (`agent-listening-1.0`, 1.0.0, unreleased) is the research baseline; `main` and the only release are
0.2.0. State which line a number or command belongs to.

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
- Quote a result with its metric definition (architecture §4.2). The holdout has been exposed: use it for
  reproduction, diagnosis and regression, and use untouched works or sources for claims about unseen material.
- Research is open: models, signal methods, context and task designs may be tried and may change the priority.
  Trials live in `eval/` with their own pinned runtime; a product dependency is added only with task evidence.
  Keep negative results. Write untested behaviour as a proposal, implemented behaviour as fact.
- Comments carry only protocol, math, licensing or platform facts the code cannot express.

## Issues and labels

Issues and verification records live in GitHub Issues/PRs for `Bayern99/Agent-Listening-CLI` (`gh`).
Triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.
Domain vocabulary: [docs/architecture.md §3.1](docs/architecture.md#31-concepts-the-whole-vocabulary).
