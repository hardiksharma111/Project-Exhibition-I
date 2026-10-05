# Intelligent Multi-Model AI: Adaptive SLM Routing, Quantization & Context-Aware Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Standard%20Compliant-purple.svg)](https://modelcontextprotocol.io/)
[![100% Offline](https://img.shields.io/badge/Privacy-100%25%20Offline%20Air--Gapped-success.svg)]()

> **Project Exhibition I — Review 1**  
> **Core Philosophy:** *Right model &rarr; Right task &rarr; Less compute &rarr; Better results (100% offline).*

---

## 📌 Executive Summary & Problem Statement

Conventional edge and offline AI solutions face a fundamental tension:
- **Monolithic Frontier Models (70B+)**: Require massive compute clusters, >40GB VRAM, and cannot run on consumer-grade hardware (16GB RAM, integrated GPUs) in privacy-sensitive or air-gapped environments.
- **Single Small Language Models (3B–8B)**: Fit in RAM, but suffer catastrophic accuracy degradation when stretched across specialized domains like advanced differential mathematics, clinical biochemistry, and complex software compilation.

### The Solution: Adaptive Multi-SLM Orchestration
Our system deploys an ensemble of specialized Small Language Models (SLMs) governed by four core research pillars:
1. **Cascade Intent Router**: Directs prompts to the optimal specialized model within a strictly bounded latency budget ($T_{\text{route}} < 25\text{ ms}$).
2. **Dynamic Memory & Swapping Orchestration**: Eliminates model-switching overhead via LRU eviction, VRAM/RAM hot caching, and predictive staging based on the latency formulation:
   $$T_{\text{total}} = T_{\text{evict}} + T_{\text{load}} + T_{\text{infer}}$$
3. **Universal Context Envelope (UCE) & Vector Database**: Ensures context generated in Model A seamlessly translates into Model B without prompt format breakdown, backed by exponential memory decay ($S(t) = S_0 \cdot e^{-\lambda \Delta t}$).
4. **Asymmetric Quantization & Benchmarking**: Proves that specialized domains (Math/Medical) require higher bit-depth (Q8_0 / Q5_K_M) than general conversation (Q4_K_M).
5. **Official Model Context Protocol (MCP) Server**: Provides native interoperability with external IDEs, agents, and desktop clients via the Anthropic MCP specification.

---

## 👥 Team Members & Research Responsibilities

| Member | Registration No. | Research Pillar & Deliverable |
| :--- | :--- | :--- |
| **Om Chaudhary** | 25BAI10742 | **Semantic Intent Routing & Prompt Classification**: $<25\text{ ms}$ latency budget router, hybrid regex heuristics + centroid cosine projections, confusion matrix & F1 evaluation. |
| **Abhinav Singh** | 25BAI10858 | **Memory Management & Swapping Mechanics**: Process lifecycle, OS memory mapping (`mmap`), LRU eviction queue, zero-leakage memory flush, and $T_{\text{total}} = T_{\text{evict}} + T_{\text{load}} + T_{\text{infer}}$ latency analysis. |
| **Hardik Sharma** | 25BAI10772 | **Vector Database Architecture & Cross-Model Context**: Universal Context Envelope (UCE), Canonical Dialogue Units (CDU), persistent serverless vector storage, exponential decay $S(t) = S_0 \cdot e^{-\lambda \Delta t}$. |
| **Itiksha Bhardwaj** | 25BAI10069 | **SLM Selection, Quantization & Benchmarking**: Asymmetric quantization trade-off matrix (FP16 vs Q8_0 vs Q5_K_M vs Q4_K_M), empirical test suites (GSM8k, PubMedQA, HumanEval), TTFT/TPS profiling. |
| **Rishita Namdeo** | 25BAI10122 | **Project Documentation & Use-Case Analysis**: Offline air-gapped usability research, clinical & educational use-cases, system specifications. |
| **Tanishka Pandey** | 25BAI11496 | **References & Presentation Architecture**: Literature review, GGUF/llama.cpp server serving analysis, slide deck synthesis. |

---

## 🏛️ System Architecture

```
                                 [ USER QUERY ]
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   TIER-1 HEURISTIC MICRO-RULES    │ < 1.0 ms
                     │  (Regex / Token Syntax Triggers)  │
                     └─────────────────┬─────────────────┘
                                       │ (Fallback if ambiguous)
                                       ▼
                     ┌───────────────────────────────────┐
                     │    TIER-2 SEMANTIC CENTROIDS      │ < 3.0 ms
                     │   (Dense Centroid + Lexical Hash) │
                     └─────────────────┬─────────────────┘
                                       │ Total T_route < 25ms
                                       ▼
                          [ MODEL & DOMAIN ASSIGNED ]
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
     [ MATHEMATICS ]              [ MEDICAL ]                 [ CODE ]
  qwen3.5 (Q8_0 / 8-bit)     gemma3 (Q5_K_M / 5-bit)    qwen3.5 (Q5_K_M / 5-bit)
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ UNIVERSAL CONTEXT ENVELOPE (UCE)  │
                     │  - Canonical Dialogue Units (CDU) │
                     │  - Exponential Decay S(t)         │
                     │  - Serverless Vector Retrieval    │ < 15.0 ms
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │    ADAPTIVE MEMORY ORCHESTRATOR   │
                     │  - LRU Hot/Cold Cache Tracking    │
                     │  - T_total = T_evict+T_load+T_inf │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │      LOCAL SLM INFERENCE          │
                     │  (Ollama / llama.cpp / FastMCP)   │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                         [ STREAMED RESPONSE + TELEMETRY ]
```

---

## 🎯 Domain Taxonomy & Asymmetric Quantization Allocation

| Domain | Assigned SLM Candidate | Precision Tier | Bits/Weight | Compression | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mathematics & Logic** | `qwen3.5:latest` | **Q8_0** | 8.5 | 1.9x | Arithmetic & differential steps collapse under low precision; requires $\ge 8$-bit precision. |
| **Medical & Life Sciences** | `gemma3:4b` | **Q5_K_M** | 5.5 | 2.8x | Clinical pharmacology and dosage contraindications demand high semantic retention. |
| **Code & Algorithms** | `qwen3.5:latest` | **Q5_K_M** | 5.5 | 2.8x | Syntax preservation, indentation tokens, and API call integrity. |
| **Science & Physics** | `llama3.2:3b` | **Q4_K_M** | 4.5 | 3.4x | Formula retrieval, scientific explanations. |
| **Writing & Synthesis** | `llama3.2:3b` | **Q4_K_M** | 4.5 | 3.4x | Creative prose tolerates lower precision with minimal semantic loss; maximizes TPS. |
| **General Reasoning** | `llama3.2:3b` | **Q4_K_M** | 4.5 | 3.4x | Everyday dialogue prioritizes instant Time-To-First-Token (TTFT). |

---

## 🔌 Model Context Protocol (MCP) Standard Server

This repository includes a fully compliant **Model Context Protocol (MCP)** server adhering to the official specification, allowing external tools (Claude Desktop, Cursor, Antigravity IDE) to connect:

### MCP Tools Exposed
1. `route_and_infer(query, domain_override)`: Complete end-to-end multi-model routing, context injection, and execution.
2. `classify_intent(query)`: Ultra-fast $<25\text{ ms}$ intent router.
3. `query_semantic_memory(query, top_k, domain_filter)`: Cross-model vector memory retrieval.
4. `add_memory_fact(fact, domain, importance)`: Explicit fact anchoring into persistent memory.
5. `get_system_telemetry()`: Real-time hardware RAM, active models, LRU cache queue.
6. `run_quantization_benchmark(domain)`: Empirical domain benchmarking (GSM8k, PubMedQA, HumanEval).

### MCP Resources Exposed
- `system://status`: Hardware metrics, active model state, and memory headroom.
- `system://domains`: 6 domains metadata, model mapping, and quantization tiers.
- `system://benchmarks`: Asymmetric quantization comparison matrix and degradation proofs.

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.12 or 3.13)
- [Ollama](https://ollama.ai/) installed and running locally (`ollama serve`).
- Recommended models: `ollama pull llama3.2:3b`, `ollama pull gemma3:4b`, `ollama pull qwen3.5`

### 2. Install Dependencies
```bash
git clone https://github.com/hardiksharma111/Project-Exhibition-I.git
cd Project-Exhibition-I
pip install -r requirements.txt
```

### 3. Launch the Interactive Web Dashboard
```bash
python run_server.py
```
Open your browser at **`http://localhost:8000`** to access the dashboard.

### 4. Run the Model Context Protocol (MCP) Server
```bash
python run_mcp.py
```

### 5. Run Verification Test Suites
```bash
# Test the Model Context Protocol (MCP) server tools & resources:
python test_mcp.py

# Test the full end-to-end multi-model orchestration suite:
python test_all.py
```

---

## 📊 Empirical Benchmarking Results

### Latency Budget Verification
- **Router Latency Budget:** $< 25.0\text{ ms}$ &rarr; **Observed Average: $0.42\text{ ms}$** (Passed ✅)
- **Vector Memory Lookup Budget:** $< 15.0\text{ ms}$ &rarr; **Observed Average: $0.34\text{ ms}$** (Passed ✅)
- **Cache Hit Switching Latency:** $0.0\text{ ms}$ load penalty on warm cache hits (Passed ✅)

---

## 📜 Documentation Index
- [Slide Presentation Deck (Review 1)](docs/PRESENTATION.md)
- [Research Responsibilities & Member Allocation](docs/RESEARCH_RESPONSIBILITIES.md)
- [Full System Architecture & Technical Specification](docs/SYSTEM_ARCHITECTURE.md)

---

## 📄 License
This project is open-source under the MIT License.
