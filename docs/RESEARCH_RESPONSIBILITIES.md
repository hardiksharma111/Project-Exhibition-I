# Project Exhibition Research Specifications & Subsystem Architecture

---

## Research Pillar 1: Systems Engineering, Memory Management & Swapping Mechanics

**Primary Focus:** Low-level resource optimization, process lifecycle management, and eliminating model-switching overhead.

### Key Research Responsibilities
* **OS & Memory Eviction Literature:** Survey operating system memory mapping (`mmap`), page caching, and process eviction strategies (LRU, FIFO, dynamic priority queues) applied to neural network weights.
* **Runtime De-allocation Benchmarks:** Investigate how local runtimes (like `llama.cpp` / Ollama / C++ bindings) allocate and deallocate tensors in system RAM and VRAM. Determine how to force an absolute zero-leakage memory flush.
* **Latency Bottleneck Analysis:** Research disk-to-RAM loading bottlenecks across storage interfaces (NVMe SSD throughput vs. RAM bus speeds) to quantify baseline model switch delays.
* **Theoretical Contribution:** Formulate the mathematical model for total latency:
  `T_total = T_evict + T_load + T_infer`
  and design a theoretical pre-fetching algorithm (predictive staging).
* **Paper/Thesis Deliverable:** Drafts the *System Architecture & Dynamic Memory Management* methodology section, memory consumption profiles, and hardware allocation proofs.

---

## Research Pillar 2: Semantic Intent Routing & Prompt Classification

**Primary Focus:** Developing an ultra-fast, zero-cloud intent classifier that directs prompts to the correct specialized model with minimal compute overhead.

### Key Research Responsibilities
* **Routing Algorithm Comparison:** Review and evaluate different routing approaches:
  * **Embedding Cosine Similarity:** Fast sentence transformer embeddings compared against domain centroids.
  * **Lightweight Discriminative Models:** Small classifiers (e.g., SetFit, MobileBERT, or a 30M parameter classifier).
  * **Keyword/Heuristic Hybrid:** Deterministic regex + probabilistic fallback.
* **Domain Taxonomy & Dataset Compilation:** Create or curate a classification dataset mapping prompt types to target models across 6 domains:
  * Medical / Biomedical
  * Mathematics / Algorithmic
  * Code Generation & Debugging
  * Science / Physics / Chemistry / Biology
  * Writing / Content Generation
  * General Reasoning / Dialogue
* **Latency Budgeting:** Set an upper latency limit for the router (< 25 ms) so the routing step does not visibly delay inference.
* **Paper/Thesis Deliverable:** Drafts the *Intelligent Routing Mechanism* section, confusion matrices, router accuracy metrics (F1-score across domains), and router inference latency tables.

---

## Research Pillar 3: Vector Database Architecture & Cross-Model Context Management

**Primary Focus:** Designing the persistent offline memory layer and context-injection pipeline across model switches.

### Key Research Responsibilities
* **Embedded Vector Storage Review:** Evaluate local, serverless vector databases (e.g., ChromaDB, LanceDB, Qdrant Embedded, SQLite-VSS) for disk footprint, query speed, and CPU indexing performance.
* **Local Embedding Models:** Benchmark tiny, offline embedding models (e.g., `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `nomic-embed-text`) on CPU inference speed and retrieval accuracy.
* **Cross-Model Context Normalization:** Research prompt templates to ensure context retrieved from a dialogue with Model A can be parsed and understood when injected into Model B without format breakdown (Universal Context Envelope).
* **Memory Pruning & Decay Strategies:** Investigate strategies for episodic vs. long-term memory: semantic chunking, dynamic relevance decay over time (`S(t) = S0 · e^(-λ · Δt)`), and fact-summarization pipelines during system idle states.
* **Paper/Thesis Deliverable:** Drafts the *Persistent Knowledge Retrieval & Memory Synthesis* section, retrieval precision/recall benchmarks, and context-window efficiency analyses.

---

## Research Pillar 4: Model Specialization, Quantization & Evaluation Benchmarks

**Primary Focus:** Selecting the optimal small language models (SLMs), testing asymmetric quantization strategies, and defining the empirical benchmarking suite.

### Key Research Responsibilities
* **Specialized Model Curation:** Select and evaluate candidates in the 3B–8B parameter range:
  * **Medical:** BioMistral-7B, MedGemma, Gemma-3-4B
  * **Math & Logic:** DeepSeek-Math-7B, Qwen2.5-Math-7B
  * **Code:** Qwen2.5-Coder-7B, DeepSeek-Coder
  * **General/Chat:** Llama-3.2-3B, Mistral-7B-Instruct, Qwen-3.5
* **Asymmetric Quantization Trade-offs:** Research the degradation impact of quantization levels (Q4_K_M vs. Q5_K_M vs. Q8_0 vs. FP16). Prove why certain domains (Math/Medical) require higher precision than casual chat.
* **Evaluation Framework Design:** Establish standard test datasets to measure domain capability:
  * **Math:** GSM8k / MATH subsets
  * **Medical:** MedQA / PubMedQA subsets
  * **Code:** HumanEval / MBPP subsets
  * **System:** Tokens Per Second (TPS), Time To First Token (TTFT), Peak VRAM/RAM curves.
* **Paper/Thesis Deliverable:** Drafts the *Experimental Setup, Model Selection & Performance Evaluation* section, comparative benchmark graphs, and accuracy-vs-quantization tradeoff tables.
