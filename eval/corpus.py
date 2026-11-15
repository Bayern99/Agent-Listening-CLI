"""Evaluation corpus for Agent Listening 1.0.

Every case names its original sources (hash-pinned in source_hashes.json), the exact construction
recipe for derived audio, the question, and the ground truth with its authority. Ground truth never
comes from a model. Split "pilot" calibrates thresholds; "holdout" is run only after
eval/thresholds.json is committed.
"""

from __future__ import annotations

SOURCES = {
    "bach10": {
        "url": "https://github.com/flippy-fyp/Bach10_v1.1 (commit 2a53cdc6495bb82f03f943b77ca3b11ddf7f5a31)",
        "license": "Research use; cite Duan, Pardo & Zhang 2010 / Duan & Pardo 2011",
        "authority": "Separately recorded real instrument stems; note list aligned to the audio (audio onset ms, score onset ms, MIDI pitch, voice 1-4 = violin, clarinet, saxophone, bassoon)",
    },
    "aof": {
        "url": "https://archive.org/details/pandacd-715-js-bach-the-art-of-the-fugue-kunst-der-fuge-bwv-1080",
        "license": "CC0 1.0 (Kimiko Ishizaka, The Art of the Fugue)",
        "authority": "Real piano recording; answers derived from the published score of BWV 1080",
    },
    "commons": {
        "url": "https://commons.wikimedia.org (per-file pages)",
        "license": "Slendro/Pelog: CC BY-SA 4.0 (Karawitan Marta Yogiswara); guqin: CC BY-SA 3.0 (CharlieHuang); "
        "Shika no Tone: public domain (Araki Kodo III, Victor 13029); duduk: CC BY 3.0 (SERGO.TEL)",
        "authority": "Real recordings; answers from file metadata and established organology/ethnomusicology",
    },
    "muse": {
        "url": "https://github.com/brandoncarone/MUSE_music_benchmark (commit 2ff5f017413a7776de5283f750349ea1fa0be61d)",
        "license": "Research/benchmarking only, no training (LICENSE_DATA.md)",
        "authority": "Contour label encoded in the stimulus file name by the benchmark authors",
    },
    "sc": {
        "url": "eval/sc/*.scd in this repository",
        "license": "Project (MIT)",
        "authority": "Constructed: the parameter difference between two source snapshots is the exact answer",
    },
}

BACH10 = {
    "01": "01-AchGottundHerr",
    "03": "03-ChristederdubistTagundLicht",
    "05": "05-DieNacht",
    "08": "08-FuerDeinenThron",
}
INSTRUMENTS = ("violin", "clarinet", "saxphone", "bassoon")

CHANGE_ASPECTS = {
    "mute": ("texture", "voice", "dynamics", "timbre"),
    "timing": ("rhythm", "texture", "voice"),
    "pitch": ("pitch", "tuning"),
    "articulation": ("articulation", "rhythm"),
    "resonance": ("timbre", "space", "production"),
    "dynamics": ("dynamics",),
    "entry": ("rhythm", "form", "voice", "texture", "motif"),
    "attack": ("articulation", "timbre", "dynamics"),
    "balance": ("dynamics", "texture", "voice", "production"),
    "removed-return": ("motif", "form", "voice", "texture", "pitch"),
}

BACH10_PLAN = {
    "01": (
        "pilot",
        {
            "mute": ("violin", 8, 14),
            "timing": ("bassoon", 12, 18),
            "pitch": ("clarinet", 6, 12),
            "articulation": ("saxphone", 10, 17),
            "resonance": ("violin", 14, 20),
        },
    ),
    "03": (
        "holdout",
        {
            "mute": ("clarinet", 10, 16),
            "timing": ("violin", 6, 12),
            "pitch": ("bassoon", 13, 19),
            "articulation": ("violin", 8, 15),
            "resonance": ("saxphone", 5, 11),
        },
    ),
    "05": (
        "holdout",
        {
            "mute": ("saxphone", 20, 27),
            "timing": ("clarinet", 14, 21),
            "pitch": ("violin", 25, 31),
            "articulation": ("bassoon", 6, 13),
            "resonance": ("clarinet", 9, 15),
        },
    ),
    "08": (
        "holdout",
        {
            "mute": ("bassoon", 15, 22),
            "timing": ("saxphone", 22, 28),
            "pitch": ("saxphone", 9, 15),
            "articulation": ("clarinet", 18, 25),
            "resonance": ("bassoon", 24, 30),
        },
    ),
}

EDIT_PARAMS = {
    "timing": {"delay_s": 0.25},
    "pitch": {"cents": 50},
    "articulation": {"keep_fraction": 0.4, "min_keep_s": 0.08},
    "resonance": {"f0_hz": 1500, "q": 8, "gain_db": 15},
    "crossfade_s": 0.03,
    "mix_gain": 0.5,
    "near_null_dbfs": -100,
}

Q_INSTRUMENT_COUNT = "How many distinct instruments are playing? A. 2 B. 3 C. 4 D. 5"
Q_LOWEST = "Which instrument plays the lowest line? A. Violin B. Clarinet C. Saxophone D. Bassoon"
Q_HIGHEST = "Which instrument plays the highest line? A. Violin B. Clarinet C. Saxophone D. Bassoon"
Q_ARTICULATION = (
    "In this passage one instrument plays noticeably shorter, more detached notes than written legato "
    "chorale style. Which one? A. Violin B. Clarinet C. Saxophone D. Bassoon"
)
Q_FAMILY = (
    "Which best describes the main sound source? A. Plucked string zither B. End-blown flute "
    "C. Ensemble of tuned gongs and metallophones D. Double-reed wind instrument"
)
Q_TUNING = (
    "Is the tuning compatible with 12-tone equal temperament (as on a modern piano)? "
    "A. Yes B. No, it uses a different tuning system"
)
Q_PULSE = (
    "Does this music have a steady, regular pulse? A. Yes, a steady regular pulse "
    "B. No, free rhythm without a steady pulse"
)
Q_FUGUE_ENTRIES = (
    "The piece opens with a single unaccompanied line and further voices enter one at a time. Which "
    "describes the register order of the first three entries? A. middle, high, low B. low, middle, high "
    "C. high, middle, low D. all voices begin together"
)
Q_VOICE_COUNT = "How many independent voices are present at the fullest point? A. 2 B. 3 C. 4 D. 5"
Q_TEXTURE = (
    "Which best describes the texture? A. Melody with chordal accompaniment B. Imitative counterpoint in "
    "which voices take up the same theme C. A single unaccompanied melody D. Repeated block chords"
)
Q_RETURN_TIME = (
    "When does the opening melodic theme first come back after the opening statement? A. around 0:25 "
    "B. around 0:45 C. around 1:10 D. it never returns"
)
Q_RETURN_FORM = (
    "The theme is heard again starting around 1:32. How is it changed? A. unchanged B. transposed higher "
    "C. melodically inverted (its contour turned upside down) D. played backwards"
)
Q_THEME_COUNT = (
    "How many times in total is the opening theme heard, counting the opening statement and every return "
    "in any transformation? A. 2 B. 3 C. 4 D. 6"
)
Q_CONTOUR = (
    "Which option best describes the overall pitch contour of the notes you hear? A. Arch (ascending and "
    "then descending) B. Inverted arch (descending and then ascending) C. Ascending (pitch rises over time) "
    "D. Descending (pitch falls over time)"
)

SC_BASE = {
    "v1Amp": 0.2,
    "v1Attack": 0.01,
    "v2Delay": 2.0,
    "v2Amp": 0.2,
    "v2Res": 0,
    "v3Delay": 4.0,
    "v3Amp": 0.2,
    "v3Detune": 0,
}
SC_VARIANTS = {
    "entry": ("holdout", {"v2Delay": 4.0}, (2.1, 22.5)),
    "detune": ("pilot", {"v3Detune": 35}, (4.1, 22.5)),
    "attack": ("pilot", {"v1Attack": 0.15}, (0.1, 18.1)),
    "resonance": ("holdout", {"v2Res": 2.0}, (2.1, 20.6)),
    "balance": ("holdout", {"v1Amp": 0.07}, (0.1, 18.1)),
}
SC_VARIANT_KIND = {
    "entry": "entry",
    "detune": "pitch",
    "attack": "attack",
    "resonance": "resonance",
    "balance": "balance",
}

MUSE_CONTOUR = {"Arch": "A", "InvArch": "B", "Asc": "C", "Desc": "D"}


def _choice(answer: str, chance: float) -> dict:
    return {"kind": "choice", "answer": answer, "chance": chance}


def _change(kind: str, start: float, end: float, **extra: object) -> dict:
    return {
        "kind": "change",
        "change": kind,
        "aspects": list(CHANGE_ASPECTS[kind]),
        "regions": [[start, end]],
        **extra,
    }


def cases() -> list[dict]:
    out: list[dict] = []
    for num, (split, edits) in BACH10_PLAN.items():
        piece = BACH10[num]
        base = {"op": "bach10-mix", "piece": piece}
        for edit, (inst, start, end) in edits.items():
            conditions = ["H", "CB"] if split == "holdout" else ["H"]
            out.append(
                {
                    "id": f"bach10-{num}-{edit}",
                    "split": split,
                    "task": "compare",
                    "inputs": [
                        base,
                        {
                            "op": "bach10-edit",
                            "piece": piece,
                            "edit": edit,
                            "instrument": inst,
                            "span": [start, end],
                        },
                    ],
                    "truth": _change(edit, start, end, instrument=inst),
                    "conditions": conditions,
                }
            )
        out.append(
            {
                "id": f"bach10-{num}-nearnull",
                "split": split,
                "task": "compare",
                "inputs": [base, {"op": "bach10-nearnull", "piece": piece}],
                "truth": {"kind": "null"},
                "conditions": ["H"],
            }
        )
        out.append(
            {
                "id": f"bach10-{num}-identical",
                "split": split,
                "task": "compare",
                "inputs": [base, {"op": "bach10-identical-flac", "piece": piece}],
                "truth": {"kind": "identical"},
                "conditions": ["H"],
            }
        )
        for qid, question, answer in (
            ("count", Q_INSTRUMENT_COUNT, "C"),
            ("lowest", Q_LOWEST, "D"),
            ("highest", Q_HIGHEST, "A"),
        ):
            out.append(
                {
                    "id": f"bach10-{num}-ask-{qid}",
                    "split": split,
                    "task": "ask",
                    "inputs": [base],
                    "question": question,
                    "truth": _choice(answer, 0.25),
                    "conditions": ["H"],
                }
            )
        inst, start, end = edits["articulation"]
        out.append(
            {
                "id": f"bach10-{num}-investigate-articulation",
                "split": split,
                "task": "investigate",
                "inputs": [
                    {
                        "op": "bach10-edit",
                        "piece": piece,
                        "edit": "articulation",
                        "instrument": inst,
                        "span": [start, end],
                    }
                ],
                "focus": [start - 2, end + 2],
                "question": Q_ARTICULATION,
                "truth": _choice("ABCD"[INSTRUMENTS.index(inst)], 0.25),
                "conditions": ["H"],
            }
        )

    cp1 = {"op": "decode", "source": "aof/01 Contrapunctus 1.flac"}
    canon8 = {"op": "decode", "source": "aof/17 Canon Alla Ottava.flac"}
    for qid, question, answer in (
        ("entries", Q_FUGUE_ENTRIES, "A"),
        ("voices", Q_VOICE_COUNT, "C"),
        ("texture", Q_TEXTURE, "B"),
    ):
        out.append(
            {
                "id": f"aof-cp1-ask-{qid}",
                "split": "holdout",
                "task": "ask",
                "inputs": [cp1],
                "question": question,
                "truth": _choice(answer, 0.25),
                "conditions": ["H"],
            }
        )
    for qid, question, answer in (("voices", Q_VOICE_COUNT, "A"), ("texture", Q_TEXTURE, "B")):
        out.append(
            {
                "id": f"aof-canon8-ask-{qid}",
                "split": "holdout",
                "task": "ask",
                "inputs": [canon8],
                "question": question,
                "truth": _choice(answer, 0.25),
                "conditions": ["H"],
            }
        )
    out.append(
        {
            "id": "aof-cp1-pitch",
            "split": "holdout",
            "task": "compare",
            "inputs": [
                cp1,
                {
                    "op": "span-pitch",
                    "source": "aof/01 Contrapunctus 1.flac",
                    "span": [92, 102],
                    "cents": 100,
                },
            ],
            "truth": _change("pitch", 92, 102),
            "conditions": ["H"],
        }
    )
    out.append(
        {
            "id": "aof-cp1-dynamics",
            "split": "holdout",
            "task": "compare",
            "inputs": [
                cp1,
                {
                    "op": "span-gain",
                    "source": "aof/01 Contrapunctus 1.flac",
                    "span": [141, 149],
                    "gain_db": -10,
                },
            ],
            "truth": _change("dynamics", 141, 149),
            "conditions": ["H"],
        }
    )
    out.append(
        {
            "id": "aof-cp1-hear",
            "split": "holdout",
            "task": "hear",
            "inputs": [cp1],
            "truth": {"kind": "descriptive"},
            "conditions": ["H"],
        }
    )

    world = {
        "slendro": (
            "commons/Ketawang_Sekar_Teja_Slendro_Manyura.oga",
            "pilot",
            (("family", Q_FAMILY, "C"), ("tuning", Q_TUNING, "B"), ("pulse", Q_PULSE, "A")),
        ),
        "pelog": (
            "commons/Ketawang_Mijil_Wigaringtyas_Pelog_Nem.oga",
            "holdout",
            (("family", Q_FAMILY, "C"), ("tuning", Q_TUNING, "B"), ("pulse", Q_PULSE, "A")),
        ),
        "guqin": ("commons/Guqin-Yangguan_Sandie.ogg", "holdout", (("family", Q_FAMILY, "A"),)),
        "shakuhachi": (
            "commons/Shikanotoone.ogg",
            "holdout",
            (("family", Q_FAMILY, "B"), ("pulse", Q_PULSE, "B")),
        ),
        "duduk": (
            "commons/SERGO.TEL_-_April_-_duduk-improvisation2.ogg",
            "holdout",
            (("family", Q_FAMILY, "D"),),
        ),
    }
    for name, (source, split, questions) in world.items():
        decoded = {"op": "decode", "source": source}
        for qid, question, answer in questions:
            chance = 0.5 if qid in ("tuning", "pulse") else 0.25
            out.append(
                {
                    "id": f"world-{name}-ask-{qid}",
                    "split": split,
                    "task": "ask",
                    "inputs": [decoded],
                    "question": question,
                    "truth": _choice(answer, chance),
                    "conditions": ["H"],
                }
            )
    out.append(
        {
            "id": "world-duduk-pitch",
            "split": "holdout",
            "task": "compare",
            "inputs": [
                {"op": "decode", "source": world["duduk"][0]},
                {"op": "span-pitch", "source": world["duduk"][0], "span": [60, 70], "cents": 40},
            ],
            "truth": _change("pitch", 60, 70),
            "conditions": ["H"],
        }
    )
    out.append(
        {
            "id": "world-shakuhachi-resonance",
            "split": "holdout",
            "task": "compare",
            "inputs": [
                {"op": "decode", "source": world["shakuhachi"][0]},
                {
                    "op": "span-resonance",
                    "source": world["shakuhachi"][0],
                    "span": [200, 210],
                    "f0_hz": 2000,
                    "q": 6,
                    "gain_db": 15,
                },
            ],
            "truth": _change("resonance", 200, 210),
            "conditions": ["H"],
        }
    )
    out.append(
        {
            "id": "world-slendro-nearnull",
            "split": "holdout",
            "task": "compare",
            "inputs": [
                {"op": "decode", "source": world["slendro"][0]},
                {"op": "near-null", "source": world["slendro"][0]},
            ],
            "truth": {"kind": "null"},
            "conditions": ["H"],
        }
    )
    for name in ("slendro", "shakuhachi"):
        out.append(
            {
                "id": f"world-{name}-hear",
                "split": "holdout",
                "task": "hear",
                "inputs": [{"op": "decode", "source": world[name][0]}],
                "truth": {"kind": "descriptive"},
                "conditions": ["H"],
            }
        )

    sc_base = {"op": "sc-canon", "params": SC_BASE}
    for name, (split, change, (start, end)) in SC_VARIANTS.items():
        out.append(
            {
                "id": f"sc-canon-{name}",
                "split": split,
                "task": "compare",
                "inputs": [sc_base, {"op": "sc-canon", "params": {**SC_BASE, **change}}],
                "truth": _change(SC_VARIANT_KIND[name], start, end),
                "conditions": ["H", "H+C"],
            }
        )
    out.append(
        {
            "id": "sc-canon-identical",
            "split": "holdout",
            "task": "compare",
            "inputs": [sc_base, {"op": "sc-canon", "params": SC_BASE, "render": "second"}],
            "truth": {"kind": "identical"},
            "conditions": ["H"],
        }
    )
    out.append(
        {
            "id": "sc-canon-nearnull",
            "split": "holdout",
            "task": "compare",
            "inputs": [sc_base, {"op": "sc-canon-nearnull", "params": SC_BASE}],
            "truth": {"kind": "null", "misleading_aspects": ["pitch", "tuning"]},
            "misleading_context": {"op": "sc-canon", "params": {**SC_BASE, "v3Detune": 35}},
            "conditions": ["H", "MC"],
        }
    )

    long_base = {"op": "sc-longform", "params": {"returnAt92": True}}
    for qid, question, answer in (
        ("return-time", Q_RETURN_TIME, "B"),
        ("return-form", Q_RETURN_FORM, "C"),
        ("theme-count", Q_THEME_COUNT, "C"),
    ):
        out.append(
            {
                "id": f"sc-long-ask-{qid}",
                "split": "holdout",
                "task": "ask",
                "inputs": [long_base],
                "question": question,
                "truth": _choice(answer, 0.25),
                "conditions": ["H"],
            }
        )
    out.append(
        {
            "id": "sc-long-removed-return",
            "split": "holdout",
            "task": "compare",
            "inputs": [long_base, {"op": "sc-longform", "params": {"returnAt92": False}}],
            "truth": _change("removed-return", 92.5, 101),
            "conditions": ["H"],
        }
    )
    out.append(
        {
            "id": "sc-long-hear",
            "split": "holdout",
            "task": "hear",
            "inputs": [long_base],
            "truth": {"kind": "descriptive"},
            "conditions": ["H"],
        }
    )

    for index, name in enumerate(sorted(MUSE_STIMULI)):
        label = name.removesuffix(".wav").split("_")[2]
        out.append(
            {
                "id": f"muse-contour-{name.removesuffix('.wav')}",
                "split": "pilot" if index % 4 == 0 else "holdout",
                "task": "ask",
                "inputs": [{"op": "copy", "source": f"muse/{name}"}],
                "question": Q_CONTOUR,
                "truth": _choice(MUSE_CONTOUR[label], 0.25),
                "conditions": ["H"],
            }
        )
    return out


MUSE_STIMULI = (
    "AMaj_Arp_Asc_Piano.wav",
    "AbMaj_Arp_Asc_Bass.wav",
    "Abmin_Arp_Asc_Bass.wav",
    "Amin_Arp_Asc_Piano.wav",
    "BbMaj_Scale_Arch_Bass.wav",
    "Bbmin_Scale_Arch_Bass.wav",
    "Bmaj_Scale_InvArch_Piano.wav",
    "Bmin_Scale_InvArch_Piano.wav",
    "CMaj_Scale_Arch.wav",
    "Cmin_Scale_Arch.wav",
    "DMaj_Arp_Desc.wav",
    "Dbmaj_Scale_InvArch_Bass.wav",
    "Dbmin_Scale_InvArch_Bass.wav",
    "Dm_Arp_Desc.wav",
    "EMaj_Arp_Desc_Bass.wav",
    "Ebmaj_Scale_Arch_Piano.wav",
    "Ebmin_Scale_Arch_Piano.wav",
    "Emin_Arp_Desc_Bass.wav",
    "FMaj_Scale_InvArch.wav",
    "Fm_Scale_InvArch.wav",
    "GMaj_Arp_Asc.wav",
    "GbMaj_Arp_Desc_Piano.wav",
    "Gbmin_Arp_Desc_Piano.wav",
    "Gm_Arp_Asc.wav",
)
