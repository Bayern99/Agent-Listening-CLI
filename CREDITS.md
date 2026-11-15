# Credits and third-party notices

Agent Listening is released under the MIT License ([LICENSE](LICENSE)). It does not relicense the
components below; each keeps its own terms.

## Runtime

| Component | Role | License |
| --- | --- | --- |
| [numpy](https://numpy.org) | Deterministic signal measurements | BSD-3-Clause |
| [soundfile](https://github.com/bastibe/python-soundfile) / libsndfile | Audio decoding, hashing, lossless transport files | BSD-3-Clause / LGPL-2.1 |
| [`gemini-api`](https://github.com/google-gemini/gemini-api-cli) (external executable, not bundled) | Calls `gemini-4-hydrogen` | Apache-2.0; use of the Gemini API is subject to Google's terms |

## Evaluation corpus (not distributed; downloaded into `$AGENT_LISTENING_EVAL_DIR`)

| Source | License / terms |
| --- | --- |
| Bach10 (Duan & Pardo) | Research use; cite Duan, Pardo & Zhang 2010 and Duan & Pardo 2011 |
| Kimiko Ishizaka, *The Art of the Fugue* (archive.org) | CC0 1.0 |
| Wikimedia Commons: Karawitan Marta Yogiswara (slendro, pelog) | CC BY-SA 4.0 |
| Wikimedia Commons: CharlieHuang, guqin *Yangguan Sandie* | CC BY-SA 3.0 |
| Wikimedia Commons: Araki Kodō III, *Shika no Tōne* (Victor 13029) | Public domain |
| Wikimedia Commons: SERGO.TEL, duduk improvisation | CC BY 3.0 |
| MUSE benchmark stimuli (Carone, Román, Ripollés) | Research and benchmarking only, no training |
