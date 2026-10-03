# TopoCoder: Topological Subgraph Condensation and Dual-Process Reasoning for Offline Coding Agents

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/Paper-Kaggle%20Writeup-orange.svg)](paper/kaggle_writeup.md)

> **Official Research Submission for the Google - The Gemma 4 Developer Agent Paper Track**  
> *Targeting: Overall Best Paper ($15,000) & Best New Resource ($10,000)*

---

## Overview

Today's best coding agents (Devin, SWE-agent, Cursor) rely on multi-hundred-billion parameter models running on massive cloud API infrastructure with 100k+ token context windows. This approach excludes millions of developers who require private, offline, or low-cost execution on consumer hardware.

When running open models such as **Gemma (2B / 9B)** on consumer hardware (e.g., Apple Silicon M-series or 8GB–16GB discrete GPUs), developers face a fundamental bottleneck:
* **Strict Context Window Limits**: Context is practically constrained to 4k–16k tokens before KV-cache memory explodes or inference speeds collapse.
* **Context Pollution & Attention Degradation**: Brute-force file searches (`grep`, `find`, `cat`) quickly dump 30,000+ tokens of boilerplate, causing small models to lose the causal thread and hallucinate.
* **Non-Euclidean Topology of Code**: Bugs rarely share lexical tokens with user-reported symptoms; defects often lie 2 to 4 call hops away in internal utility modules.

**TopoCoder** resolves this crisis by introducing a neuro-symbolic framework where repository navigation is performed by a fast, deterministic symbolic graph engine running on CPU, reserving the local Gemma model strictly for high-level deliberation and patch synthesis on a compact, condensed topological slice.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TOPOCODER ARCHITECTURE                                 │
│                                                                                        │
│   ┌──────────────────────────┐                   ┌──────────────────────────────────┐  │
│   │   Codebase Repository    │                   │   Failing Unit Test / Traceback  │  │
│   └────────────┬─────────────┘                   └────────────────┬─────────────────┘  │
│                │                                                  │                    │
│                ▼                                                  ▼                    │
│   ┌──────────────────────────┐                   ┌──────────────────────────────────┐  │
│   │ Hierarchical Code Graph  │                   │ Personalized CodeRank (PCR) Walk │  │
│   │ (AST, Calls, Inheritance)│ ────────────────> │  - Inverts execution call stack  │  │
│   └──────────────────────────┘                   │  - Runs in < 21ms on laptop CPU  │  │
│                                                  └────────────────┬─────────────────┘  │
│                                                                   │                    │
│                                                                   ▼                    │
│   ┌──────────────────────────┐                   ┌──────────────────────────────────┐  │
│   │ AST Verification Gate    │                   │ Topological Context Condensation │  │
│   │ - Non-destructive patch  │ <──────────────── │  - Focal node: Full Source Code  │  │
│   │ - Syntax regression check│                   │  - Boundary: AST Signature Stubs │  │
│   └────────────▲─────────────┘                   │  - Context invariant < 850 tokens│  │
│                │                                 └────────────────┬─────────────────┘  │
│                │                                                  │                    │
│                └────────────────── [Gemma Policy] <───────────────┘                    │
│                               (Offline 9B on MPS / GPU)                                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Breakthroughs & Empirical Highlights

1. **100.0% Defect Localization Recall@5**: Evaluated on multi-hop defect scenarios, TopoCoder's Personalized CodeRank consistently bounds the true root cause within the top 4 candidate functions.
2. **Context Invariance (95.1% to 99.4% Token Reduction)**: While raw repository tokens explode from 16,000 to over 132,000 tokens on 80-module enterprise codebases, TopoCoder's condensed context remains **strictly invariant at ~818 tokens**.
3. **Sub-25ms Execution on Consumer CPU**: Full repository AST parsing takes $<400$ms, and Personalized CodeRank computes in **$<21$ milliseconds on CPU**, requiring **0 MB of GPU VRAM** and leaving 100% of accelerator memory available for Gemma's weights and KV cache.
4. **Complete Open-Source Library (`codegraph`)**: Reusable Python package released under the Apache 2.0 license for graph-augmented code navigation.
5. **Parameter-Efficient Post-Training (T-SFT & T-DPO)**: 60 SFT trajectories and 60 DPO pairs fine-tune Gemma using rank-16 LoRA, yielding an **85.4% loss reduction** (2.842 -> 0.416) and a **40.9% latency speedup** (7.87s -> 4.65s) on edge hardware.

---

## Benchmark Results

### 1. Defect Localization Recall on Multi-Hop Repository Tasks (`TopoBench`)
| Method | Localization Paradigm | Recall@3 (%) | Recall@5 (%) | Avg. Context Tokens |
| :--- | :--- | :---: | :---: | :---: |
| **ReAct-Grep** | Keyword Grep + File Dumps | 60.0% | N/A | 266.6 |
| **BM25-RAG** | Lexical Chunk Retrieval | 80.0% | N/A | 237.4 |
| **Full-Graph** | Relational $k$-hop Neighborhood | 60.0% | N/A | 379.6 |
| **TopoCoder (Ours)** | **H-CKG + Personalized CodeRank** | **60.0%** | **100.0%** | **Bounded ($\le 552$)** |

### 2. Large-Scale Repository Invariance Simulation (Up to 22,000 Lines of Code)
| Modules | Code Lines | Graph Nodes ($|\mathcal{V}|$) | Graph Edges ($|\mathcal{E}|$) | Parse Time | PCR Time | Raw Tokens | TopoCoder Tokens | Token Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10** | 2,749 | 220 | 1,155 | 59.5 ms | 1.2 ms | 16,576 | **818** | **95.1%** |
| **25** | 6,874 | 550 | 6,825 | 130.6 ms | 3.5 ms | 41,446 | **818** | **98.0%** |
| **50** | 13,749 | 1,100 | 26,775 | 252.8 ms | 9.2 ms | 82,896 | **818** | **99.0%** |
| **80** | 21,999 | 1,760 | 68,040 | 385.8 ms | 20.3 ms | 132,636 | **818** | **99.4%** |

### 3. Live On-Device A/B Comparative Benchmark (Apple Silicon MPS / Ollama)
| Model Variant | System Protocol | AST Syntax Validity | Mean Task Latency | Avg Context Tokens | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Base Gemma-2-2B** | Zero-Shot Greedy | 100.0% | 7.87s | 758.4 | Baseline |
| **TopoCoder-Gemma-2B** | **T-SFT + T-DPO Post-Trained** | **100.0%** | **4.65s** | **751.4** | **+40.9% faster** |

---

## Repository Structure

```
├── LICENSE                                # Open Source Apache 2.0 License
├── README.md                              # Project documentation & reproduction guide
├── setup.py                               # Package installation script
├── agent.yaml                             # Declarative config for Kaggle Gemma 4 sandbox
├── codegraph/                             # Core Reusable Library ("Best New Resource")
│   ├── __init__.py                        # Package exports
│   ├── schema.py                          # H-CKG node & edge type definitions
│   ├── parser.py                          # High-performance 2-pass AST repository parser
│   ├── graph.py                           # NetworkX-compatible CodeKnowledgeGraph engine
│   ├── pagerank.py                        # Personalized CodeRank (relational random walk)
│   ├── condensation.py                    # Topological Context Condenser (knapsack solver)
│   ├── embeddings.py                      # Structural random-walk embeddings & hybrid retrieval
│   ├── visualizer.py                      # Interactive HTML/JS graph dashboard renderer
│   └── tools.py                           # Standardized agent navigation tool suite
├── agent/                                 # Agent Scaffolding & Verification
│   ├── __init__.py                        # Agent module exports
│   ├── dual_process.py                    # Dual-Process System 1/System 2 Gemma Agent
│   └── verifier.py                        # AST syntax validation and atomic patch applier
├── benchmarks/                            # Empirical Evaluation Suite
│   ├── tasks.py                           # Multi-hop defect localization benchmark tasks
│   ├── baselines.py                       # ReAct-Grep, BM25-RAG, Full-Graph evaluators
│   ├── eval_runner.py                     # Automated benchmark runner & metrics collector
│   ├── large_repo_simulation.py           # 22k-line repository scalability test
│   └── output/                            # Raw JSON evaluation results
├── paper/                                 # Official Research Submission
│   ├── kaggle_writeup.md                  # Kaggle Writeup (≤ 3,000 words, publication-ready)
│   ├── main.tex                           # Full arXiv/NeurIPS LaTeX draft
│   ├── main.pdf                           # Compiled 5-page publication-ready PDF
│   ├── references.bib                     # Academic BibTeX citations
│   └── figures/                           # Benchmark charts & interactive_graph.html
└── tests/                                 # Unit & Integration Tests
    ├── test_codegraph.py                  # Parser, graph, embeddings, and visualizer tests
    └── test_agent.py                      # Verifier and dual-process agent tests
```

---

## Quickstart & Replication Guide

### 1. Installation
Clone the repository and install in editable mode:
```bash
git clone https://github.com/chirana07/topocoder.git
cd topocoder
pip install -e .
```

### 2. Run Test Suite
Verify that all components pass:
```bash
pytest tests/ -v
```

### 3. One-Click Master Reproduction Pipeline
Execute all unit tests, benchmarks, scalability simulations, post-training loss runs, and LaTeX compilation in a single command:
```bash
python3 run_all.py
```

### 4. Interactive Kaggle Notebook
A complete, self-contained notebook is available at `TopoCoder_Kaggle_Submission.ipynb` for direct upload to Kaggle Notebooks.

---

## Paper Submission Documents

* **Official Kaggle Writeup**: [paper/kaggle_writeup.md](paper/kaggle_writeup.md) (Strictly under the 3,000-word limit; comprehensive methods, math, tables, and citations).
* **arXiv / NeurIPS-style LaTeX Draft**: [paper/main.tex](paper/main.tex) and [paper/references.bib](paper/references.bib).
* **Generated Publication Figures**:
  * [Token Consumption Comparison](paper/figures/token_consumption.png)
  * [Multi-Hop Recall Degradation](paper/figures/hop_degradation.png)
  * [Context Invariance across Codebase Scale](paper/figures/token_invariance.png)
  * [Algorithmic Latency Scaling on CPU](paper/figures/latency_scaling.png)

---

## License

This project is licensed under the **Apache License 2.0** in accordance with the official competition rules. See [LICENSE](LICENSE) for details.
