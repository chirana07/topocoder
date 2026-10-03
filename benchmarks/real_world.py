"""
Real-World Production Codebase Benchmark for TopoCoder.
Evaluates AST parsing throughput, graph extraction fidelity, and Personalized CodeRank
on real-world production codebases from the Python runtime environment.
"""

import json
import os
import sys
import time
from typing import Dict, List

from codegraph.condensation import TopologicalCondenser
from codegraph.graph import CodeKnowledgeGraph
from codegraph.pagerank import PersonalizedCodeRank
from codegraph.parser import RepositoryParser


def benchmark_production_codebase(target_dir: str, package_name: str) -> Dict[str, any]:
    print(f"\nEvaluating real-world package: {package_name} ({target_dir})")
    
    t0 = time.perf_counter()
    parser = RepositoryParser(target_dir)
    nodes, edges = parser.parse_repository()
    parse_lat = (time.perf_counter() - t0) * 1000

    graph = CodeKnowledgeGraph()
    for n in nodes.values():
        graph.add_node(n)
    for e in edges:
        graph.add_edge(e)

    stats = graph.stats()
    raw_tokens = sum(n.token_count for n in graph.nodes.values())

    # Personalized CodeRank from central class/function
    classes = [nid for nid, n in graph.nodes.items() if n.node_type.value == "class"]
    seed_node = classes[0] if classes else list(graph.nodes.keys())[0]

    t0 = time.perf_counter()
    pcr = PersonalizedCodeRank(graph)
    ranked = pcr.rank_nodes({seed_node: 1.0}, top_k=5, exclude_seeds=True)
    pcr_lat = (time.perf_counter() - t0) * 1000

    # Topological Context Condensation under 2,500 token ceiling
    condenser = TopologicalCondenser(graph, default_budget_tokens=2500)
    condensed = condenser.condense({seed_node: 1.0}, budget_tokens=2500)
    cond_tokens = condensed["total_tokens"]
    token_red = ((raw_tokens - cond_tokens) / max(1, raw_tokens)) * 100.0

    print(f"  Nodes: {stats['num_nodes']} | Edges: {stats['num_edges']}")
    print(f"  Parse Latency: {parse_lat:.1f} ms | PCR Latency: {pcr_lat:.1f} ms")
    print(f"  Raw Tokens: {raw_tokens:,} -> TopoCoder: {cond_tokens:,} ({token_red:.1f}% reduction)")
    print(f"  Top Ranked Call Candidates from {seed_node}:")
    for nid, score in ranked[:3]:
        print(f"    - {nid}: {score:.4f}")

    return {
        "package": package_name,
        "nodes": stats["num_nodes"],
        "edges": stats["num_edges"],
        "node_types": stats["node_types"],
        "edge_types": stats["edge_types"],
        "parse_ms": parse_lat,
        "pcr_ms": pcr_lat,
        "raw_tokens": raw_tokens,
        "condensed_tokens": cond_tokens,
        "token_reduction_pct": token_red,
        "top_ranked": [(nid, round(s, 4)) for nid, s in ranked[:5]]
    }


def run_all_real_world_benchmarks():
    print("=" * 70)
    print("STARTING TOPOCODER REAL-WORLD PRODUCTION CODEBASE BENCHMARK")
    print("=" * 70)

    py_lib = os.path.dirname(os.__file__)
    packages = [
        ("http", os.path.join(py_lib, "http")),
        ("urllib", os.path.join(py_lib, "urllib")),
        ("logging", os.path.join(py_lib, "logging")),
        ("email", os.path.join(py_lib, "email")),
    ]

    all_results = []
    for pkg_name, pkg_path in packages:
        if os.path.exists(pkg_path):
            res = benchmark_production_codebase(pkg_path, pkg_name)
            all_results.append(res)

    os.makedirs("benchmarks/output", exist_ok=True)
    out_file = "benchmarks/output/real_world_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Saved real-world benchmark metrics to {out_file}")
    print("=" * 70)
    return all_results


if __name__ == "__main__":
    run_all_real_world_benchmarks()
