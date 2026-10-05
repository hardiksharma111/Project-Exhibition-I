"""
Core Configuration Module
Intelligent Multi-Model AI: Adaptive SLM Routing, Quantization & Context-Aware Inference
"""

import os
from typing import Dict, List, Any
from pydantic import BaseModel

# Base Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MEMORY_STORE_DIR = os.path.join(DATA_DIR, "memory")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MEMORY_STORE_DIR, exist_ok=True)

# System Latency Budgets (in milliseconds)
MAX_ROUTER_LATENCY_MS = 25.0
MAX_MEMORY_LOOKUP_LATENCY_MS = 15.0

# Memory Eviction & Swapping Thresholds
MAX_RAM_THRESHOLD_PERCENT = 85.0
TARGET_UNLOAD_PERCENT = 70.0
LRU_CACHE_CAPACITY = 3  # Max models warm in memory simultaneously

# Supported Research Domains
DOMAINS = {
    "mathematics": {
        "id": "mathematics",
        "name": "Mathematics & Logic",
        "description": "Calculus, linear algebra, arithmetic, discrete math, proofs, and symbolic reasoning",
        "primary_model": "qwen3.5:latest",
        "fallback_model": "llama3.2:3b",
        "recommended_quant": "Q8_0",
        "min_precision_bits": 8,
        "color": "#38bdf8",  # Sky blue
        "icon": "∑"
    },
    "medical": {
        "id": "medical",
        "name": "Medical & Life Sciences",
        "description": "Clinical diagnosis, pharmacology, anatomy, biochemistry, pathology, and therapeutics",
        "primary_model": "gemma3:4b",
        "fallback_model": "llama3.2:3b",
        "recommended_quant": "Q5_K_M",
        "min_precision_bits": 5,
        "color": "#34d399",  # Emerald green
        "icon": "⚕"
    },
    "code": {
        "id": "code",
        "name": "Code & Algorithms",
        "description": "Software engineering, debugging, syntax generation, git, system architecture, refactoring",
        "primary_model": "qwen3.5:latest",
        "fallback_model": "llama3.2:3b",
        "recommended_quant": "Q5_K_M",
        "min_precision_bits": 5,
        "color": "#a78bfa",  # Violet
        "icon": "</>"
    },
    "science": {
        "id": "science",
        "name": "Science & Physics",
        "description": "Classical and quantum physics, inorganic/organic chemistry, astronomy, thermodynamics",
        "primary_model": "llama3.2:3b",
        "fallback_model": "gemma3:4b",
        "recommended_quant": "Q4_K_M",
        "min_precision_bits": 4,
        "color": "#f59e0b",  # Amber
        "icon": "⚛"
    },
    "writing": {
        "id": "writing",
        "name": "Writing & Synthesis",
        "description": "Creative writing, essay structure, professional emails, copywriting, critique, rhetoric",
        "primary_model": "llama3.2:3b",
        "fallback_model": "llama3.2:latest",
        "recommended_quant": "Q4_K_M",
        "min_precision_bits": 4,
        "color": "#ec4899",  # Pink
        "icon": "✍"
    },
    "general": {
        "id": "general",
        "name": "General Reasoning & Dialogue",
        "description": "Everyday queries, open-ended discussion, conversational logic, multi-domain synthesis",
        "primary_model": "llama3.2:3b",
        "fallback_model": "llama3.2:latest",
        "recommended_quant": "Q4_K_M",
        "min_precision_bits": 4,
        "color": "#94a3b8",  # Slate
        "icon": "💬"
    }
}

# Ollama Endpoint Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# Core Architecture & System Specifications
SYSTEM_SPECIFICATIONS = [
    {
        "title": "Semantic Intent Router",
        "subsystem": "Intent Classification & Intelligent Routing",
        "latency_budget": "T_route < 25.0 ms",
        "description": "Ultra-fast zero-cloud intent classifier using a hybrid cascade of deterministic regex heuristics and semantic centroid cosine projections across 6 specialist domains."
    },
    {
        "title": "Adaptive Memory Orchestrator",
        "subsystem": "Memory Management & Swapping Mechanics",
        "latency_budget": "T_total = T_evict + T_load + T_infer",
        "description": "Dynamic OS memory orchestration, LRU eviction queue, zero-leakage flushing, and warm cache bypass eliminating model loading bottlenecks."
    },
    {
        "title": "Universal Context Envelope",
        "subsystem": "Vector Database & Cross-Model Context",
        "latency_budget": "T_lookup < 15.0 ms",
        "description": "Model-invariant Canonical Dialogue Units (CDU) preserving semantic context across model switches, backed by exponential relevance decay S(t) = S0 * e^(-lambda * t)."
    },
    {
        "title": "Asymmetric Quantization Suite",
        "subsystem": "Model Specialization & Quantization Benchmarks",
        "latency_budget": "Pareto Frontier Optimization",
        "description": "Asymmetric quantization matrix allocating higher precision (Q8_0 / Q5_K_M) to sensitive domains (Math, Medical) while maximizing throughput on Q4_K_M for General chat."
    },
    {
        "title": "Model Context Protocol Interop",
        "subsystem": "Standard MCP 2.0 Server",
        "latency_budget": "Zero Cloud Dependency",
        "description": "Official Model Context Protocol specification server exposing tools, resources, and prompt templates for seamless integration with external IDEs and agents."
    },
    {
        "title": "Air-Gapped Privacy Architecture",
        "subsystem": "System Integrity & Local Execution",
        "latency_budget": "100% Offline Local",
        "description": "Fully self-contained offline architecture running on consumer hardware without telemetry egress or external API dependencies."
    }
]
