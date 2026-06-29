# Gatekeeper

> A self-improving RAG agent with a regression-proof learning gate.
> Improvement gets in. Regression doesn't.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Vector_Store-ChromaDB-orange)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-412991?logo=openai)
![MCP](https://img.shields.io/badge/Protocol-MCP-green)
![Status](https://img.shields.io/badge/Status-In_Progress-yellow)

---

## Knowledge base

The agent answers questions about the **solar system's 8 planets**, grounded entirely in 4 source documents:

| File | Covers |
|------|--------|
| `mercury_venus.txt` | Diameter, orbital period, temperature, moons, spacecraft visits |
| `earth_mars.txt` | Diameter, rotation, moons, surface features, rover missions |
| `jupiter_saturn.txt` | Size, moons (Galilean + Titan), rings, storm systems, missions |
| `uranus_neptune.txt` | Axial tilt, moons, ring count, temperature, Voyager 2 flyby |

Two planets per document — this is deliberate. It forces the retriever to surface the right document *and* the model to pick out the correct planet's data from it, creating a realistic retrieval challenge.

---

## Evaluation dataset

30 question→answer pairs split across two files:

| File | Questions | Purpose |
|------|-----------|---------|
| `data/train.json` | 20 | The agent learns on these — failures trigger reflection |
| `data/heldout.json` | 10 | Sealed — never learned on, only used by the gate and harness |

All answers are short exact strings (a name, number, or single phrase) verifiable directly from the source documents — no outside knowledge required.

**Overlapping skills across both sets** — both files contain questions about moon counts, temperatures, spacecraft years, orbital periods, and diameters. A lesson learned on a train failure can plausibly affect a held-out answer, which is the condition that makes regression catchable.

---

## Project structure

```
regression-rag/
├── data/
│   └── docs/           ← 4 knowledge base documents (planets)
├── src/
│   ├── retriever.py    ← chunk, embed, retrieve top-k
│   ├── agent.py        ← answer() + reflect()
│   └── harness.py      ← exact-match eval on any dataset
└── requirements.txt
```

---

## Setup

```bash
pip install -r regression-rag/requirements.txt
export OPENAI_API_KEY=sk-...
```
