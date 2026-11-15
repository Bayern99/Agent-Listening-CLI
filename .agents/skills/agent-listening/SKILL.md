---
name: agent-listening
description: Listen to real rendered or recorded audio with the agent-listening CLI and get time-located, deterministically checked evidence. Use to hear a render, ask a question about a passage, compare two versions, investigate a sound or voice, or review a render against artistic goals. Not for generating or editing audio.
---

# Agent Listening

`agent-listening` (1.x) sends audio to `gemini-4-hydrogen` through the `gemini-api` CLI and returns one
JSON evidence document (`agent-listening/evidence` v1). Run `agent-listening doctor` once; exit 0 means
the provider CLI and credential are ready.

## Pick the task

| Need | Command |
| --- | --- |
| What does this render sound like? | `agent-listening hear AUDIO [--focus 1:20-1:45]` |
| One specific question | `agent-listening ask AUDIO "QUESTION" [--focus S-E]` |
| What changed between two versions? | `agent-listening compare BEFORE AFTER [--question Q]` |
| Close look at one sound, voice or passage | `agent-listening investigate AUDIO --focus S-E [--question Q]` |
| Does the render meet the goals? | `agent-listening review AUDIO --brief brief.json [--before PREV]` |

Add `--out FILE.json` to save the evidence (no-clobber). Each listening call takes roughly 15–120 s, so
listen only when the answer can change your next decision. Ask multiple-choice questions when you can;
lettered options make answers comparable across revisions.

A brief is human intent, written by the composer, never invented by you:

```json
{"goals": ["The second voice answers the first in imitation"],
 "protect": [{"text": "The low drone stays continuous", "loci": [{"start": 0, "end": 30}]}]}
```

## Source code as context

`--context FILE` adds SuperCollider source or score text. Listen without context first: context can make
the model report what the code intends instead of what the audio contains. Claims whose `basis` is
`context` or `audio+context` reflect that text; weigh `audio` claims for perceptual questions.

## Read the evidence

1. `status`: `complete`, `partial` (dry run), or `failed` (see `error`; identity and measurements remain).
2. `summary` / `answer`, then `claims`. Each claim has `loci` (input label, start, end in seconds of that
   input), `aspect`, `scale` (event, phrase, section, work), optional `relation`, `basis`, `confidence`.
3. Claim `status` comes from deterministic checks, never from the model:
   - `contradicted`: the signal disagrees (a difference claimed on identical audio, or outside every measured
     difference region). Discard it.
   - `ungrounded`: no valid time locus. Treat as an unlocated impression.
   - `consistent`: a deterministic check agrees. Still a model claim, not ground truth.
   - `unchecked`: no deterministic check applies (most `hear`/`ask` claims).
4. `limitations`: `MODEL_INPUT_MONO` means the model heard a mono mix (no stereo judgements);
   `MODEL_TIME_APPROXIMATE` means times are approximate at the seconds scale; `CONTEXT_SUPPLIED` flags
   possible priming.
5. In `compare`, bit-identical inputs return zero claims without a model call; the
   `signal-difference` measurement lists where the signal actually changed (waveform `regions` and
   phase-invariant `spectral.regions`). Claims carrying a `measurement-guided` check were heard after the
   region was pointed out: trust their description, not their location.
6. `review` listens without the brief; its `assessments` come from a text-only pass over the checked claims.
   `unclear` means no claim addressed the goal, not that the goal failed.

Measured on the holdout corpus, the model named the right kind of change in only 3 of 23 controlled edits
while the signal measurement located all 23, and differences it reported often vanished when the clip order was
swapped. When a measured region has no consistent claim, a change happened that the model did not describe:
investigate that span or ask the human.

Model timestamps and pitch names are approximate. Before a change depends on a claim, confirm it with a
focused `investigate`, a `compare` against the previous render, or the human.

## SC Agent IDE

Write evidence into the run's analysis directory and bind it to the render before reading conclusions:
`node helpers/check-analysis.mjs ... --evidence <file>` (exit 0 verified, 2 unverified, 1 failed).
Machine evidence never replaces the human Accept/Revise decision.
