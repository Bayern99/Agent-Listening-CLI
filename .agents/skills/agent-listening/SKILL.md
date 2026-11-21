---
name: agent-listening
description: Put a question to real rendered or recorded audio with the agent-listening CLI and get time-located observations with deterministic checks. Use to hear a render, ask about a passage, compare two versions, investigate a sound or voice, or review a render against stated goals. Not for generating or editing audio.
---

# Agent Listening

`agent-listening` 1.x sends audio to `gemini-4-hydrogen` through the `gemini-api` CLI and returns one
JSON evidence document (`agent-listening/evidence` v1). Run `agent-listening doctor` once; exit 0 means
the provider CLI and credential are ready. If `agent-listening --version` prints 0.2.0, the installed tool
is the older pipeline and has none of these commands.

Listen when the answer can change your next musical decision. Start from that decision, then choose the
question, the context and the time range.

## Choose the question

| The decision depends on | Command |
| --- | --- |
| What is in this audio, and where | `agent-listening hear AUDIO [--focus S-E]` |
| One fact you can state as a question | `agent-listening ask AUDIO "QUESTION" [--focus S-E]` |
| What changed between two versions | `agent-listening compare BEFORE AFTER [--question Q]` |
| One sound, voice or passage in detail | `agent-listening investigate AUDIO --focus S-E [--question Q]` |
| Whether a render meets stated goals | `agent-listening review AUDIO --brief brief.json [--before PREV]` |

- Ask about one thing per call: a voice, a relation between two moments, a property of a passage. Any
  musical material is a valid subject, including free rhythm, electroacoustic sound and non-12-TET tuning.
- Lettered options (`A. ... B. ...`) suit facts with a few possible answers and make answers comparable
  across revisions. Use an open question when you cannot enumerate the answers; closed options hide
  whatever you did not think to list.
- Phrase the question neutrally. "Does the low voice stay continuous?" asks; "Notice the break in the low
  voice" tells.
- Follow up. Take a claim's locus from one document as the `--focus` of the next call.

Add `--out FILE.json` to save the evidence (no-clobber). A single call usually returns in tens of seconds
(median 14 s on single-question tasks in the evaluation); long audio, `compare` follow-ups and provider
retries can take minutes.

## Choose the context

Evidence is chosen by the question:

| You need to know | Use |
| --- | --- |
| How it was built or what it was meant to do | Source code, recipe, score |
| What was rendered and where the signal changed | The audio and the `measurements` in the evidence |
| How it sounds | Listening claims |
| Where the music should go | The creator |

`--context FILE` (repeatable) adds source, score text or notes to the listening call. Supply it when the
question needs it: naming voices the way the source does, relating sound to a score, hearing against target
material you describe. In the evaluation, context made the model name three changes it had missed, and on a
control it produced two claims of a change the audio did not contain. So when the question is "did this
actually happen in the sound", listen without context, or listen both ways and compare. Claims carry
`basis`: `audio`, `context` or `audio+context`.

## Choose the time range

- Whole file for form, returns, long-range relations and overall character.
- `--focus S-E` (seconds, `M:SS` or `H:MM:SS`) for anything local: one entry, an attack, a transition. The
  model hears only that excerpt; times in the evidence are mapped back to the original file.
- After `compare`, focus on the measured regions in `signal-difference`, widened by a few seconds so the
  change has surroundings.
- Model times are approximate at the scale of seconds. Use measured region boundaries when timing matters.

## Read the evidence

1. `status`: `complete`, `partial` (dry run), or `failed` (see `error`; identity and measurements remain).
2. `summary` / `answer`, then `claims`. Each claim has `loci` (input label, start, end in seconds of that
   input), `aspect`, `scale` (event, phrase, section, work), optional `relation`, `basis`, `confidence`.
3. Claim `status` comes from deterministic checks, never from the model:
   - `contradicted`: the signal disagrees (a difference claimed on identical audio or outside every
     measured region, or "unchanged" claimed over a measured change). Discard it.
   - `ungrounded`: no valid time locus. An unlocated impression.
   - `consistent`: the claim passed the checks that applied. The checks test where the signal changed, not
     what the claim says about the music, so the content is still the model's.
   - `unchecked`: no check applies (most `hear` and `ask` claims).
4. A `measurement-guided` check means the region was pointed out to the model before it described it. The
   location is the measurement's; use the description as far as something else supports it.
5. `limitations`: `MODEL_INPUT_MONO` (the model heard a mono mix; take stereo facts from
   `investigate --measure stereo`), `MODEL_INPUT_REDUCED_RESOLUTION`, `MODEL_TIME_APPROXIMATE`,
   `CONTEXT_SUPPLIED`.
6. In `compare`, bit-identical inputs return zero claims without a model call. `signal-difference` lists
   where the signal changed (waveform `regions` and phase-invariant `spectral.regions`). A measured region
   with no claim is a change the model did not describe: investigate that span.
7. `review` listens without seeing the brief, then a text-only pass judges each goal from the checked
   claims. `unclear` means no claim addressed the goal. `--context` files do reach the listening pass.

## What the evidence has shown

- Closed questions about a whole recording or a focused span were answered correctly 39 of 45 times.
- In `compare`, the measurement found the start of the change in all 23 controlled edits; the model's own
  difference claims carried a fitting aspect label in 3, and none of 10 survived swapping the clip order.
  Rely on the measurement for where, and on a focused follow-up question for what.
- `review` verdicts did not separate defective from corrected renders (2 of 5 flagged in each).
- `hear` descriptions are well-formed and located; their content has not been graded.

Before a change depends on a claim, confirm it another way: a focused `investigate` or `ask`, a
measurement, the source, or the person. Pitch names are approximate.

## Briefs and attribution

A brief records what a person stated:

```json
{"goals": ["The second voice answers the first in imitation"],
 "protect": [{"text": "The low drone stays continuous", "loci": [{"start": 0, "end": 30}]}]}
```

Keep your own creative hypotheses labelled as yours: test them with `ask` or `compare --question`, and
report them as proposals. The evidence file records a brief's hash, not its author.

## SC Agent IDE

Write evidence into the run's analysis directory and bind it to the render before reading conclusions:
`node helpers/check-analysis.mjs ... --evidence <file>` (exit 0 verified, 2 unverified, 1 failed).
`--evidence` needs the SC Agent IDE branch `agent-listening-1.0-evidence`; released IDE versions accept only
0.2 receipts. Machine evidence never replaces the human Accept/Revise decision.
