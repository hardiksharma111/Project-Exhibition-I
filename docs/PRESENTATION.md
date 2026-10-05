# Project Exhibition: Intelligent Multi-Model AI
**Adaptive SLM Routing, Quantization and Context-Aware Inference**  

---

## Slide 2: Index
*Project map.*

| Section | Content | Section | Content |
| :--- | :--- | :--- | :--- |
| **01** | Title of the project | **06** | Research Pillars & Subsystem Specifications |
| **02** | Problem Statement | **07** | References |
| **03** | Software Requirement | | |
| **04** | Process Timeline | | |
| **05** | Usability | | |

---

## Slide 3: Problem Statement
Using a single large AI model for diverse tasks leads to high memory, compute and inference costs. An offline system needs to intelligently select specialized SLMs while managing model switching, memory and context efficiently.

### Scope
**6 Domains:**  
Medical | Mathematics | Code | General Reasoning | Writing | Science

---

## Slide 4: Project Objectives

1. **Privacy**  
   Run the AI locally so sensitive data does not need to be sent to the cloud.
2. **Intelligent Routing**  
   Select the most suitable small AI model based on the type and complexity of the task.
3. **Resource Efficiency**  
   Minimize computational and memory requirements by using lightweight models instead of larger ones.
4. **Switching Optimization**  
   Reduce model-switching overhead by considering factors such as model loading time, memory availability, and whether a model is already loaded.
5. **Performance Evaluation**  
   Evaluate the performance of different routing strategies using response time, memory usage, computational cost, and response quality.
6. **Demonstrate Feasibility**  
   Show the feasibility of efficient offline AI on resource-constrained devices through a working prototype.

---

## Slide 5: Software Requirement

* **Python / C++**  
  Used for implementing the AI pipeline, routing logic, memory management, and system integration.
* **Ollama**  
  Provides the selected open-weight specialized SLMs for medical, mathematical, coding, and general tasks.
* **Hugging Face Models**  
  Enables efficient local/offline inference of quantized SLMs with optimized CPU/GPU memory usage.
* **Local Embedding Model**  
  Converts prompts and stored information into vectors for semantic similarity, routing, and offline memory retrieval.

---

## Slide 6: Process Timeline & Architecture Overview

### Project Timeline
* **Week 0 - 1:** Research
* **Week 0 - 1:** Define
* **Week 1 - 2:** Design
* **Week 1 - 2:** Test
* **Week 2:** Launch

### Adaptive Offline AI: Intelligent Multi-Model Routing and Resource-Efficient SLM Inference
*Core Philosophy:* Right model → Right task → Less compute → Better results (offline)

#### Architecture Pipeline
1. **User Query:**  
   You ask a question or provide a request.  
   *Example:* `"Explain the symptoms of diabetes."`
2. **Intent Router:**  
   Understands the intent and classifies the query using lightweight methods (embeddings, classifier or keyword rules).
   * Embedding Similarity
   * Lightweight Classifier
   * Keyword / Heuristic Rules
3. **Model Selection:**  
   Routes the query to the best specialized SLM based on the domain:
   * **Medical:** MedGemma 4B / Gemma 3 4B
   * **Mathematics:** Qwen2.5-Math 7B / DeepSeek-Math
   * **Coding:** Qwen2.5-Coder 7B / DeepSeek-Coder
   * **General:** Llama 3.2 3B / Qwen 3.5
4. **Context & Memory Layer:**  
   Retrieves relevant information from the vector database and manages context across model switches.
   * Vector Database (semantic search)
   * Context Normalization (model-invariant format)
   * Memory Management (pruning & decay)
5. **Response:**  
   The selected model generates the answer with the right context.  
   *Example Output:* `"Diabetes is a chronic condition characterized by high blood sugar levels..."`
6. **Quantization & Resource Optimization (for all models):**  
   Each model is tested with different quantization levels to find the best balance between accuracy and resource usage.
   * **Tested Levels:** FP16 (Full precision) → Q4_K_M (4-bit) → Q5_K_M (5-bit) → Q8_0 (8-bit)
   * **Metrics Measured:**
     * Accuracy (domain-wise)
     * Speed (TTFT / TPS)
     * Memory (RAM / VRAM)
     * Model Size

#### Key Highlights
* **100% Offline:** No cloud APIs, complete privacy and data control.
* **Resource Efficient:** Smaller models, optimized memory and compute usage.
* **Specialized Performance:** Right model for the right task (better accuracy).
* **Seamless Model Switching:** Maintains context across multiple models.
* **Real-World Applications:** Healthcare, education, coding research and more.

---

## Slide 7: Usability

* **Offline & Privacy-Preserving AI:** Enables AI inference without relying on cloud APIs or internet connectivity.
* **Education & Research:** Routes mathematical and reasoning problems to specialized models.
* **Resource-Constrained Devices:** Quantized SLMs reduce memory and computational requirements.
* **Healthcare:** Specialized medical SLMs can assist with medical knowledge and biomedical queries.
* **Coding Assistance:** Provides local code generation, debugging, and explanation.
* **Air-Gapped Environments:** Suitable for organizations where sensitive data cannot leave the local system.

---

## Slide 8: Research Pillars & Subsystem Specifications

| Pillar | Subsystem | Technical Deliverable & Architecture |
| :--- | :--- | :--- |
| **01** | Model Specialization & Quantization | Asymmetric quantization matrix (FP16, Q8_0, Q5_K_M, Q4_K_M) & empirical test suites |
| **02** | Use-Case Engineering & Air-Gap Specs | Project documentation, clinical & educational use-case analysis |
| **03** | Memory Management & Model Swapping | Hardware-aware memory management, OS memory mapping (`mmap`), LRU cache queue |
| **04** | Vector Database & Cross-Model Context | Serverless vector database, Universal Context Envelope (UCE), exponential decay |
| **05** | Serving Infrastructure & References | Literature review, GGUF/llama.cpp serving analysis, slide deck synthesis |
| **06** | Intent Classification & Model Routing | Dual-tier heuristic regex + semantic centroid routing (< 25 ms) |

---

## Slide 9: References

* **llama.cpp - GitHub Repository:** Official documentation for local inference, model serving, and hardware support. *(GitHub: llama.cpp)*
* **Hugging Face - GGUF usage with llama.cpp:** Explains GGUF models, downloading, and running them with llama.cpp. *(Hugging Face GGUF documentation)*
* **llama.cpp - Model Documentation:** Covers obtaining and quantizing models for local inference. *(Model documentation)*
* **llama.cpp - Running a Server:** Useful for explaining how your local models can be served and how routing can load/unload models. *(Server documentation)*
* **llama.cpp - CLI Documentation:** Useful for understanding model execution and performance-related settings. *(CLI documentation)*

---

## Slide 10: Ending
*Thank you.*
