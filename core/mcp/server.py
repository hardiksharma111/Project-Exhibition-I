"""
Official Model Context Protocol (MCP) Server
Intelligent Multi-Model AI: Adaptive SLM Routing & Context-Aware Inference
Standard: FastMCP / Model Context Protocol Spec
Provides Tools, Resources, and Prompts to external AI agents & IDEs.
"""

import json
from typing import Dict, List, Optional, Any
from fastmcp import FastMCP

from core.config import DOMAINS, TEAM_MEMBERS
from core.router.intent_classifier import IntentRouter
from core.memory.vector_store import VectorMemoryStore
from core.memory.context_normalizer import UniversalContextEnvelope
from core.orchestrator.memory_manager import AdaptiveMemoryManager
from core.orchestrator.model_switcher import ModelSwitcher
from core.benchmarks.quantization_matrix import get_quantization_comparison
from core.benchmarks.benchmark_suite import BenchmarkSuite


# Initialize singletons for the MCP server
router = IntentRouter()
memory_store = VectorMemoryStore()
memory_manager = AdaptiveMemoryManager()
model_switcher = ModelSwitcher(memory_manager=memory_manager)
benchmark_suite = BenchmarkSuite(model_switcher=model_switcher)

# Create the FastMCP Server instance
mcp = FastMCP(
    "Intelligent-Multi-Model-Router",
    instructions=(
        "You are connected to the Adaptive Multi-SLM Routing Engine (Project Exhibition). "
        "This MCP server intelligently routes prompts across 6 domains (Math, Medical, Code, "
        "Science, Writing, General) to specialized local SLMs, manages cross-model memory, "
        "minimizes swapping latency (T_total = T_evict + T_load + T_infer), and enforces asymmetric quantization."
    )
)


# ==========================================
# MCP TOOLS
# ==========================================

@mcp.tool(
    name="route_and_infer",
    description="Routes a query to the optimal domain-specialist SLM, injects cross-model context, and executes inference with rich telemetry."
)
def route_and_infer(
    query: str,
    domain_override: Optional[str] = None
) -> str:
    """
    Full pipeline execution: Intent Classification (<25ms) -> Vector Retrieval (<15ms)
    -> Memory Staging -> Inference -> Response with Telemetry.
    """
    # 1. Route Intent
    route_res = router.route(query, forced_domain=domain_override)

    # 2. Retrieve Cross-Model Memory
    retrieved, mem_latency = memory_store.search(query, top_k=2, min_similarity=0.15)
    injected_facts = [r["text"] for r in retrieved]

    # 3. Create Canonical Dialogue Unit and Compile
    cdu = UniversalContextEnvelope.create_cdu(
        role="user",
        content=query,
        source_model="client",
        domain=route_res.domain
    )
    compiled_messages = UniversalContextEnvelope.compile_for_model(
        cdus=[cdu],
        target_model=route_res.target_model,
        target_domain=route_res.domain,
        injected_facts=injected_facts
    )

    # 4. Execute Inference with Memory Manager Staging
    response_text, telemetry = model_switcher.generate_chat_response(
        messages=compiled_messages,
        target_model=route_res.target_model,
        domain=route_res.domain
    )

    # 5. Store Interaction in Vector Memory
    memory_store.insert(
        text=f"User: {query[:100]} | Assistant: {response_text[:120]}",
        domain=route_res.domain,
        source_model=telemetry["model_used"],
        importance=0.8
    )

    payload = {
        "status": "success",
        "output": response_text,
        "routing": {
            "domain": route_res.domain,
            "domain_name": route_res.domain_name,
            "confidence": route_res.confidence,
            "target_model": route_res.target_model,
            "recommended_quant": route_res.recommended_quant,
            "latency_ms": route_res.latency_ms,
            "decision_tier": route_res.decision_tier,
            "decision_rationale": route_res.decision_rationale
        },
        "telemetry": {
            "model_used": telemetry["model_used"],
            "ttft_ms": telemetry["ttft_ms"],
            "tps": telemetry["tps"],
            "swap_decomposition": telemetry["swap_decomposition"],
            "retrieved_memory_items": len(retrieved),
            "memory_lookup_ms": mem_latency
        }
    }

    return json.dumps(payload, indent=2)


@mcp.tool(
    name="classify_intent",
    description="Classifies a user query across 6 domains within <25ms, returning optimal target model, quantization level, and confidence distribution."
)
def classify_intent(query: str) -> str:
    res = router.route(query)
    return json.dumps(res.model_dump(), indent=2)


@mcp.tool(
    name="query_semantic_memory",
    description="Searches persistent cross-model vector storage for relevant facts, prior dialogue units, and contextual anchors."
)
def query_semantic_memory(
    query: str,
    top_k: int = 3,
    domain_filter: Optional[str] = None
) -> str:
    results, latency_ms = memory_store.search(
        query=query,
        top_k=top_k,
        domain_filter=domain_filter
    )
    return json.dumps({
        "query": query,
        "retrieval_latency_ms": latency_ms,
        "results_count": len(results),
        "results": results
    }, indent=2)


@mcp.tool(
    name="add_memory_fact",
    description="Explicitly stores a factual assertion or user preference in the cross-model persistent vector memory."
)
def add_memory_fact(
    fact: str,
    domain: str = "general",
    importance: float = 1.0
) -> str:
    item_id = memory_store.insert(
        text=fact,
        domain=domain,
        source_model="mcp_user",
        importance=importance
    )
    return json.dumps({
        "status": "stored",
        "item_id": item_id,
        "domain": domain,
        "importance": importance
    })


@mcp.tool(
    name="get_system_telemetry",
    description="Returns real-time OS memory status (psutil), loaded models, LRU cache queue, and swap overhead metrics."
)
def get_system_telemetry() -> str:
    mem_stats = memory_manager.get_hardware_memory_stats()
    return json.dumps({
        "memory": mem_stats.model_dump(),
        "active_model": memory_manager.active_model_name,
        "models_catalog": {k: v.model_dump() for k, v in memory_manager.models.items()},
        "total_swaps_recorded": len(memory_manager.swap_history)
    }, indent=2)


@mcp.tool(
    name="run_quantization_benchmark",
    description="Executes empirical benchmarks on a specific domain (GSM8k, PubMedQA, HumanEval) to evaluate TTFT, TPS, and accuracy."
)
def run_quantization_benchmark(domain: str = "mathematics") -> str:
    res = benchmark_suite.run_domain_benchmark(domain=domain, sample_limit=1)
    return json.dumps(res, indent=2)


# ==========================================
# MCP RESOURCES
# ==========================================

@mcp.resource("system://status")
def get_system_resource() -> str:
    """Returns live hardware metrics and active model state."""
    stats = memory_manager.get_hardware_memory_stats()
    return json.dumps({
        "system_status": "ONLINE",
        "engine": "Adaptive Multi-SLM Orchestrator",
        "active_model": memory_manager.active_model_name,
        "hardware_ram": stats.model_dump()
    }, indent=2)


@mcp.resource("system://domains")
def get_domains_resource() -> str:
    """Returns the 6 research domains, assigned models, and quantization tiers."""
    return json.dumps(DOMAINS, indent=2)


@mcp.resource("system://benchmarks")
def get_benchmarks_resource() -> str:
    """Returns the asymmetric quantization trade-off matrix and findings."""
    return json.dumps(get_quantization_comparison(), indent=2)


# ==========================================
# MCP PROMPTS
# ==========================================

@mcp.prompt(
    name="expert_routed_query",
    description="Formats an expert query with domain specialization framing and memory grounding."
)
def expert_routed_query_prompt(prompt: str) -> str:
    res = router.route(prompt)
    return (
        f"[SYSTEM]: You are the designated {res.domain_name} specialist.\n"
        f"[MODEL ASSIGNED]: {res.target_model} (Precision: {res.recommended_quant})\n"
        f"[ROUTING CONFIDENCE]: {res.confidence:.1%}\n\n"
        f"[USER PROMPT]:\n{prompt}\n\n"
        "Provide your rigorous domain-specialized response:"
    )


def create_mcp_server():
    return mcp


if __name__ == "__main__":
    mcp.run()
