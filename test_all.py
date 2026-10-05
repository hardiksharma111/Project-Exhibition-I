"""
Comprehensive End-to-End System Test Suite
Verifies all 4 Research Pillars + MCP Server + REST API
"""

import sys
import time
import json
from fastapi.testclient import TestClient

def run_all_tests():
    print("=" * 75)
    print("RUNNING COMPREHENSIVE MULTI-SLM ORCHESTRATION & MCP VERIFICATION SUITE")
    print("=" * 75)

    # 1. Test Router (Om's Pillar)
    print("\n--- [TEST 1] Om's Intent Router (<25ms budget) ---")
    from core.router import IntentRouter
    router = IntentRouter()
    
    math_res = router.route("Calculate the integral of sin(2x) dx from 0 to pi")
    print(f"  Math Query -> Domain: {math_res.domain}, Latency: {math_res.latency_ms} ms, Budget OK: {math_res.budget_adhered}")
    assert math_res.domain == "mathematics", f"Expected mathematics, got {math_res.domain}"
    assert math_res.latency_ms <= 25.0, f"Router latency {math_res.latency_ms}ms exceeded 25ms"

    med_res = router.route("What are the differential diagnoses for acute epigastric pain and elevated lipase?")
    print(f"  Medical Query -> Domain: {med_res.domain}, Latency: {med_res.latency_ms} ms, Target SLM: {med_res.target_model}")
    assert med_res.domain == "medical", f"Expected medical, got {med_res.domain}"

    code_res = router.route("def quicksort(arr): return arr if len(arr) <= 1 else ...")
    print(f"  Code Query -> Domain: {code_res.domain}, Decision Tier: {code_res.decision_tier}")
    assert code_res.domain == "code", f"Expected code, got {code_res.domain}"

    bench_rep = router.get_benchmark_report()
    print(f"  Router Benchmark Accuracy: {bench_rep['accuracy'] * 100}% | Avg Latency: {bench_rep['average_latency_ms']} ms")
    assert bench_rep["accuracy"] >= 0.90, "Router accuracy below 90%"

    # 2. Test Vector Memory & Context Envelope (Hardik's Pillar)
    print("\n--- [TEST 2] Hardik's Vector Store & Universal Context Envelope ---")
    from core.memory import VectorMemoryStore, UniversalContextEnvelope, MemoryDecayManager
    v_store = VectorMemoryStore()
    v_store.insert("Patient is taking Metformin 500mg twice daily for diabetes control.", "medical", "gemma3:4b", 0.95)
    
    results, search_lat = v_store.search("metformin diabetes dosage", top_k=1)
    print(f"  Vector Search Latency: {search_lat} ms (< 15ms target) | Match: {results[0]['text'][:50]}...")
    assert len(results) > 0, "Vector search returned no results"
    assert search_lat <= 15.0, f"Vector search latency {search_lat}ms exceeded 15ms"

    # Context compilation
    cdu = UniversalContextEnvelope.create_cdu("user", "What is my current medication?", "client", "medical")
    compiled = UniversalContextEnvelope.compile_for_model([cdu], "llama3.2:3b", "medical", injected_facts=[results[0]["text"]])
    print(f"  Compiled Context Messages Count: {len(compiled)} | System prompt contains injected facts: {'METFORMIN' in compiled[0]['content'].upper()}")
    assert len(compiled) == 2, "Compiled messages structure invalid"

    # 3. Test Memory & Swapping Orchestrator (Abhinav's Pillar)
    print("\n--- [TEST 3] Abhinav's Memory & Swapping Latency Formulation ---")
    from core.orchestrator import AdaptiveMemoryManager
    mem_mgr = AdaptiveMemoryManager(lru_capacity=3)
    
    # Predict swap cost
    decomp = mem_mgr.predict_swap_cost("gemma3:4b")
    print(f"  Cold Switch Cost -> T_evict: {decomp.t_evict_ms}ms, T_load: {decomp.t_load_ms}ms, T_infer: {decomp.t_infer_ms}ms | T_total: {decomp.t_total_ms}ms")
    assert decomp.t_total_ms == decomp.t_evict_ms + decomp.t_load_ms + decomp.t_infer_ms, "Latency equation mismatch"

    # Stage model
    success, decomp2, evicted = mem_mgr.stage_model("gemma3:4b")
    print(f"  Staged gemma3:4b -> Warm in RAM: {'gemma3:4b' in mem_mgr.lru_queue}")
    assert "gemma3:4b" in mem_mgr.lru_queue, "Model not added to LRU queue"

    # Check hot switch
    decomp_hot = mem_mgr.predict_swap_cost("gemma3:4b")
    print(f"  Hot Switch Check -> Is Hot: {decomp_hot.is_hot_switch}, T_load: {decomp_hot.t_load_ms}ms")
    assert decomp_hot.is_hot_switch, "Model should be hot in RAM"
    assert decomp_hot.t_load_ms == 0.0, "Hot switch must have 0ms load penalty"

    # 4. Test Quantization Matrix (Itiksha's Pillar)
    print("\n--- [TEST 4] Itiksha's Asymmetric Quantization Matrix ---")
    from core.benchmarks import get_quantization_comparison
    quant_matrix = get_quantization_comparison()
    print(f"  Quantization Profiles Available: {quant_matrix['levels']}")
    assert len(quant_matrix["profiles"]) == 4, "Expected 4 quantization profiles"

    # 5. Test MCP Server (Model Context Protocol)
    print("\n--- [TEST 5] Model Context Protocol (MCP) Server Tools & Resources ---")
    from core.mcp.server import (
        classify_intent,
        add_memory_fact,
        query_semantic_memory,
        get_system_telemetry,
        get_system_resource,
        get_domains_resource,
        get_benchmarks_resource
    )
    
    mcp_class = json.loads(classify_intent("Calculate the derivative of f(x) = 3x^2"))
    print(f"  MCP Tool 'classify_intent' -> Domain: {mcp_class['domain']}, Target: {mcp_class['target_model']}")
    assert mcp_class["domain"] == "mathematics", "MCP tool failed classification"

    mcp_fact = json.loads(add_memory_fact("Speed of light c = 299,792,458 m/s", "science", 1.0))
    print(f"  MCP Tool 'add_memory_fact' -> Item ID: {mcp_fact['item_id']}")
    assert mcp_fact["status"] == "stored", "MCP tool failed storing fact"

    mcp_mem = json.loads(query_semantic_memory("speed of light"))
    print(f"  MCP Tool 'query_semantic_memory' -> Found: {mcp_mem['results_count']} results")
    assert mcp_mem["results_count"] > 0, "MCP memory query failed"

    sys_res = json.loads(get_system_resource())
    print(f"  MCP Resource 'system://status' -> Status: {sys_res['system_status']}")
    assert sys_res["system_status"] == "ONLINE", "MCP resource status invalid"

    # 6. Test FastAPI Web Application & REST Endpoints
    print("\n--- [TEST 6] FastAPI REST & Static Endpoints ---")
    from api.app import app
    client = TestClient(app)

    res_index = client.get("/")
    print(f"  GET / -> Status: {res_index.status_code}")
    assert res_index.status_code == 200, "Index page failed to load"

    res_stats = client.get("/api/system/stats")
    print(f"  GET /api/system/stats -> Status: {res_stats.status_code}, RAM %: {res_stats.json()['memory']['ram_usage_percent']}%")
    assert res_stats.status_code == 200, "Stats endpoint failed"

    res_route = client.post("/api/route", json={"query": "Explain how CRISPR-Cas9 genome editing works"})
    print(f"  POST /api/route -> Status: {res_route.status_code}, Domain: {res_route.json()['domain']}")
    assert res_route.status_code == 200, "Route endpoint failed"

    res_mcp_inv = client.post("/api/mcp/invoke", json={
        "tool_name": "classify_intent",
        "arguments": {"query": "Write a binary search algorithm in Rust"}
    })
    print(f"  POST /api/mcp/invoke -> Status: {res_mcp_inv.status_code}, Result Domain: {res_mcp_inv.json()['result']['domain']}")
    assert res_mcp_inv.status_code == 200, "MCP invoke endpoint failed"

    print("\n" + "=" * 75)
    print("ALL 6 TEST SUITES PASSED FLAWLESSLY! 100% OPERATIONAL READY!")
    print("=" * 75)


if __name__ == "__main__":
    run_all_tests()
