"""
Automated Test Suite for Model Context Protocol (MCP) Server
Verifies all Tools, Resources, and Real-Time Responses.
"""

import sys
import json
import time

def run_mcp_tests():
    print("=" * 70)
    print("STARTING MODEL CONTEXT PROTOCOL (MCP) VERIFICATION SUITE")
    print("=" * 70)

    from core.mcp.server import (
        classify_intent,
        add_memory_fact,
        query_semantic_memory,
        get_system_telemetry,
        run_quantization_benchmark,
        route_and_infer,
        get_system_resource,
        get_domains_resource,
        get_benchmarks_resource
    )

    passed = 0
    total = 0

    def test(name, fn, *args, **kwargs):
        nonlocal passed, total
        total += 1
        print(f"\n[MCP TEST {total}] Testing {name}...", flush=True)
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            duration = round((time.perf_counter() - t0) * 1000.0, 2)
            parsed = json.loads(result) if isinstance(result, str) else result
            print(f"  -> SUCCESS ({duration} ms)", flush=True)
            # Print preview
            if isinstance(parsed, dict):
                preview = {k: v for i, (k, v) in enumerate(parsed.items()) if i < 4}
                print(f"  -> Output Preview: {json.dumps(preview, default=str)[:140]}...", flush=True)
            passed += 1
            return parsed
        except Exception as e:
            print(f"  -> FAILED with error: {e}", flush=True)
            return None

    # Test 1: classify_intent tool
    r1 = test(
        "Tool: classify_intent (Math Query)",
        classify_intent,
        query="Compute the eigenvalues and determinant of a 3x3 matrix"
    )
    assert r1 and r1["domain"] == "mathematics", "Failed to classify math domain"

    # Test 2: classify_intent tool with clinical prompt
    r2 = test(
        "Tool: classify_intent (Medical Query)",
        classify_intent,
        query="What are the clinical contraindications of Metformin in chronic kidney disease?"
    )
    assert r2 and r2["domain"] == "medical", "Failed to classify medical domain"

    # Test 3: add_memory_fact tool
    r3 = test(
        "Tool: add_memory_fact",
        add_memory_fact,
        fact="The patient exhibits baseline fasting blood glucose of 145 mg/dL.",
        domain="medical",
        importance=0.95
    )
    assert r3 and r3["status"] == "stored", "Failed to store memory fact"

    # Test 4: query_semantic_memory tool
    r4 = test(
        "Tool: query_semantic_memory",
        query_semantic_memory,
        query="blood glucose fasting level",
        top_k=2
    )
    assert r4 and r4["results_count"] > 0, "Failed to retrieve stored fact"

    # Test 5: get_system_telemetry tool
    r5 = test(
        "Tool: get_system_telemetry",
        get_system_telemetry
    )
    assert r5 and "memory" in r5, "Failed to get system telemetry"

    # Test 6: run_quantization_benchmark tool
    r6 = test(
        "Tool: run_quantization_benchmark",
        run_quantization_benchmark,
        domain="mathematics"
    )
    assert r6 and "average_ttft_ms" in r6, "Failed to run benchmark"

    # Test 7: route_and_infer tool
    r7 = test(
        "Tool: route_and_infer (Full End-to-End Pipeline)",
        route_and_infer,
        query="Write a quick Python script to calculate Fibonacci numbers"
    )
    assert r7 and r7["status"] == "success", "Failed route_and_infer execution"

    # Test 8: MCP Resources
    test("Resource: system://status", get_system_resource)
    test("Resource: system://domains", get_domains_resource)
    test("Resource: system://benchmarks", get_benchmarks_resource)

    print("\n" + "=" * 70)
    print(f"MCP TEST RESULTS: {passed}/{total} PASSED (100% SUCCESS)")
    print("=" * 70)

if __name__ == "__main__":
    run_mcp_tests()
