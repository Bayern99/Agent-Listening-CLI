# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`docs/architecture.md`**: §1 for the product aim, §3.1 for the concept vocabulary, §4 for the evaluation
  and how its results are read. It is the authority for design and for what results mean.
- **`GLOSSARY.md`** at the repo root and **`docs/adr/`**, if they exist.

This repo is single-context and currently has no separate glossary or ADR directory. **Proceed silently**
without them; don't flag their absence or suggest creating them upfront. The `/domain-modeling` skill creates
them lazily when terms or decisions actually get resolved.

## Use the architecture's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name),
use the term as defined in `docs/architecture.md` §3.1. Don't drift to synonyms.

If the concept you need isn't there yet, that's a signal: either you're inventing language the project
doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag conflicts with recorded decisions

If your output contradicts a decision recorded in `docs/architecture.md` (or a later ADR), surface it
explicitly rather than silently overriding:

> _Contradicts architecture §3.2 (goal-blind review), but worth reopening because…_
