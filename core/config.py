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

# Team Members Metadata (from presentation & research responsibilities)
TEAM_MEMBERS = [
    {
        "name": "Om Chaudhary",
        "reg_no": "25BAI10742",
        "role": "Intent Classification & Intelligent Model Routing",
        "deliverable": "Ultra-fast zero-cloud intent router (<25ms budget), domain taxonomy, confusion matrix, hybrid cascade"
    },
    {
        "name": "Abhinav Singh",
        "reg_no": "25BAI10858",
        "role": "Memory Management, Model Swapping & System Optimization",
        "deliverable": "Dynamic memory orchestration, LRU eviction, process lifecycle, swapping latency formulation (T_total = T_evict + T_load + T_infer)"
    },
    {
        "name": "Hardik Sharma",
        "reg_no": "25BAI10772",
        "role": "Vector Database, Context Management & Cross-Model Memory",
        "deliverable": "Universal Context Envelope (UCE), canonical dialogue units, persistent serverless vector storage, exponential decay S(t)=S0*e^(-lambda*t)"
    },
    {
        "name": "Itiksha Bhardwaj",
        "reg_no": "25BAI10069",
        "role": "SLM Selection, Quantization & Performance Benchmarking",
        "deliverable": "Asymmetric quantization matrix (FP16 vs Q8_0 vs Q5_K_M vs Q4_K_M), empirical benchmark suite (GSM8k, MedQA, HumanEval), TPS/TTFT profiling"
    },
    {
        "name": "Rishita Namdeo",
        "reg_no": "25BAI10122",
        "role": "Project Documentation & Use-Case Analysis",
        "deliverable": "Comprehensive system documentation, offline air-gapped usability analysis, domain requirements"
    },
    {
        "name": "Tanishka Pandey",
        "reg_no": "25BAI11496",
        "role": "References & Presentation Architecture",
        "deliverable": "Academic literature citations, llama.cpp & GGUF benchmarks, slide deck synthesis"
    }
]
