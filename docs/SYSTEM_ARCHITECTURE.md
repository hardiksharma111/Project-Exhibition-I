# System Architecture & Technical Specification

## Intelligent Multi-Model AI: Adaptive SLM Routing, Quantization & Context-Aware Inference

---

### 1. Executive Summary & Improved Paradigm

Conventional edge and offline AI solutions face a fundamental tension:
- **Monolithic Frontier Models (70B+)**: Provide broad multi-domain competency, but cannot run on consumer-grade hardware (16GB RAM, integrated GPUs) due to immense memory footprints (>40GB) and prohibitive latency.
- **Single Small Language Models (3B–8B)**: Fit in RAM, but suffer catastrophic accuracy degradation when stretched across specialized domains like advanced differential mathematics, clinical biochemistry, and complex software compilation.

**Our Proposed System: Adaptive Multi-SLM Orchestration**
Instead of a one-size-fits-all model, our system deploys an ensemble of specialized Small Language Models (SLMs), governed by:
1. **Cascade Intent Router**: Directs prompts to the optimal specialized model within a strictly bounded latency budget (T_route < 25 ms).
2. **Dynamic Memory & Swapping Orchestration**: Minimizes cold-load penalties using LRU eviction, VRAM/RAM hot caching, and predictive staging based on the latency equation:
   `T_total = T_evict + T_load + T_infer`
3. **Universal Context Envelope & Semantic Memory**: Ensures context generated in Model A seamlessly injects into Model B without prompt format breakdown, coupled with exponential memory decay (`S(t) = S0 · e^(-λ · Δt)`) and vector search.
4. **Asymmetric Quantization & Benchmarking**: Asymmetrically quantizes each model according to domain sensitivity (e.g., Q8_0 for Math/Medical vs Q4_K_M for General Chat).
5. **Model Context Protocol (MCP) Server**: Provides native interoperability with external IDEs, agents, and desktop clients via the official Anthropic MCP specification.

---

### 2. Domain Taxonomy & Model Assignment

| Domain | Assigned SLM Candidate | Quantization Strategy | Rationale |
| :--- | :--- | :--- | :--- |
| **Mathematics & Logic** | `qwen3.5` / `qwen2.5-math:7b` | Q8_0 / Q5_K_M (High Precision) | Math reasoning is hyper-sensitive to weight quantization drift; requires higher bit-depth. |
| **Medical & Life Sciences** | `gemma3:4b` / `medgemma:4b` | Q5_K_M / Q8_0 | Biomedical terminology and dosage reasoning require precision. |
| **Code & Algorithms** | `qwen3.5` / `qwen2.5-coder:7b` | Q5_K_M / Q4_K_M | Syntax preservation and indentation token consistency. |
| **Science & Physics** | `llama3.2:3b` / `gemma3:4b` | Q4_K_M / Q5_K_M | Physical constants, formula retrieval, scientific explanation. |
| **Writing & Synthesis** | `llama3.2:3b` / `llama3.2:latest` | Q4_K_M (Fastest TPS) | Creative prose tolerates lower precision without semantic collapse. |
| **General Reasoning & Chat** | `llama3.2:3b` | Q4_K_M (Lowest TTFT) | General queries prioritize instant Time-To-First-Token. |

---

### 3. Latency & Memory Math Formulation

The end-to-end user-perceived turnaround time is modeled as:
```
T_turnaround = T_route + T_mem_retrieval + T_swap_if_needed + T_TTFT + (N_tokens / TPS)
```

Where:
- `T_route < 25 ms`: Achieved via tier-1 regex matching and tier-2 domain-centroid cosine projection.
- `T_mem_retrieval < 15 ms`: Achieved via serverless embedded vector storage.
- `T_swap_if_needed`:
  - `T_swap = 0 ms` if target model is already warm in RAM/VRAM
  - `T_swap = T_evict + T_load` if cold switch or memory headroom threshold exceeded
- Memory Eviction Trigger:
  - `RAM_used + Model_required > θ_max_RAM (θ = 85%)`

---

### 4. Universal Context Envelope (UCE)

When switching from `Medical SLM` to `Code SLM`, naive raw prompt injection fails due to differing prompt tokens (`<start_of_turn>`, `<|im_start|>`, etc.). The Universal Context Envelope serializes conversation turns into a Canonical Dialogue Unit (CDU):
```json
{
  "cdu_id": "cdu-8921",
  "source_domain": "medical",
  "source_model": "gemma3:4b",
  "timestamp": 1728148800,
  "importance_weight": 0.85,
  "facts": ["Patient symptoms indicate Type-2 Glycemic variability", "Target HbA1c is < 7.0%"],
  "content": "Explanation of insulin resistance provided to user."
}
```
Upon switching to `Coding SLM`, UCE compiles the context into Model B's native format with injected facts.

---

### 5. Official Model Context Protocol (MCP) Integration

The system exposes an MCP Server interface:
- **Tools**: `route_and_infer`, `classify_intent`, `query_semantic_memory`, `add_memory_fact`, `get_system_telemetry`, `run_quantization_benchmark`
- **Resources**: `system://status`, `system://domains`, `system://benchmarks`
- **Prompts**: `expert_routed_query`, `cross_model_analysis`
