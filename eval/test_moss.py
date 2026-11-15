from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import moss
from moss_recover import recoverable


def test_response_validation_preserves_failure_and_choice():
    built = moss.prompts.build("ask", [], question="A or B?")
    payload = {"summary": "uncertain", "answer": "B", "claims": []}
    assert moss.parse_response(json.dumps(payload), dict(built.schema)) == payload
    assert moss.parse_response(f"```json\n{json.dumps(payload)}\n```", dict(built.schema)) == payload
    with pytest.raises(json.JSONDecodeError):
        moss.parse_response('{"claims": [', dict(built.schema))
    with pytest.raises(jsonschema.ValidationError):
        moss.parse_response('{"summary": "ok", "claims": []}', dict(built.schema))


def test_counterbalance_keeps_labels_and_reference_roles():
    clips = [
        moss.prompts.ClipView("A", "before", 30, 0, 30, 1),
        moss.prompts.ClipView("B", "after", 30, 0, 30, 1),
    ]
    built = moss.prompts.build("compare", clips[::-1])
    text = moss.chat_prompt(built, ["B", "A"])
    assert text.index("Clip B: <|audio_bos|>") < text.index("Clip A: <|audio_bos|>")
    assert "Clip A (before)" in text and "Clip B (after)" in text
    assert text.count("<|audio_bos|>") == 2


def test_scorer_counts_wrong_aspect_separately_from_location():
    claim = {
        "relation": "differs",
        "aspect": "timbre",
        "checks": [],
        "loci": [{"start": 10, "end": 12}],
        "status": "consistent",
    }
    from score import score_change

    result = score_change({"claims": [claim]}, {"regions": [[10, 12]], "aspects": ["pitch"]})
    assert result["located_any_aspect"]
    assert not result["detected"] and not result["onset_hit3"]


def test_runner_rejects_manifest_mismatch_before_inference(tmp_path):
    import numpy as np
    import soundfile as sf

    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(1600), 16000)
    recipe = {"op": "test"}
    manifest = {moss.recipe_key(recipe): {"path": str(path), "sha256": "0" * 64}}
    case = {"id": "hash-test", "task": "hear", "inputs": [recipe]}
    with pytest.raises(ValueError, match="build-manifest hash mismatch"):
        moss.execute(case, "H", manifest, None, tmp_path / "run")


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"status": "failed", "error": "RuntimeError: MPS backend out of memory"}, True),
        ({"status": "failed", "error": "JSONDecodeError", "raw_response": "{"}, False),
        ({"status": "failed", "error": "MPS backend out of memory", "raw_response": ""}, False),
        ({"status": "failed", "error": "NONFINITE_LOGITS"}, False),
        ({"status": "complete", "error": "MPS backend out of memory"}, False),
    ],
)
def test_recovery_never_retries_a_generated_answer(record, expected):
    assert recoverable(record) is expected
