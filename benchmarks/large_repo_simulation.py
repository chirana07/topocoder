"""
Large-Scale Repository Scaling Simulation for TopoCoder.
Evaluates graph construction throughput, Personalized CodeRank latency,
and Token Budget invariance on a multi-module repository containing 10,000+ lines of code.
"""

import os
import time
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from codegraph.condensation import TopologicalCondenser
from codegraph.graph import CodeKnowledgeGraph
from codegraph.pagerank import PersonalizedCodeRank
from codegraph.parser import RepositoryParser


def generate_synthetic_repository(num_modules: int = 40, classes_per_mod: int = 3, methods_per_class: int = 5) -> Dict[str, str]:
    """Generates a realistic multi-module codebase with multi-hop dependencies."""
    files = {}
    for m in range(num_modules):
        mod_name = f"service_{m:02d}"
        lines = [f'"""Module {mod_name} implementing enterprise service layer {m}."""']
        
        # Cross module import
        if m > 0:
            lines.append(f"from service_{m-1:02d} import ServiceClass_{m-1:02d}_0")
            
        for c in range(classes_per_mod):
            cls_name = f"ServiceClass_{m:02d}_{c}"
            lines.append(f"\nclass {cls_name}:")
            lines.append(f'    """Service controller {cls_name}."""')
            lines.append("    def __init__(self, config: dict = None):")
            lines.append("        self.config = config or {}")
            lines.append("        self.initialized = True")
            
            for f in range(methods_per_class):
                fn_name = f"execute_step_{f}"
                callee = f"ServiceClass_{m-1:02d}_0" if m > 0 and f == 0 else f"execute_step_{max(0, f-1)}"
                lines.append(f"\n    def {fn_name}(self, payload: dict, step_id: int = {f}) -> dict:")
                lines.append(f'        """Executes processing step {f} with schema validation."""')
                lines.append("        # Complex business logic with 30 lines of surrounding code")
                for pad in range(12):
                    lines.append(f"        val_{pad} = payload.get('k_{pad}', {pad}) * 1.5")
                lines.append(f"        return {{'status': 'ok', 'step': step_id, 'ref': '{callee}'}}")
                
        files[f"pkg/{mod_name}.py"] = "\n".join(lines)
    return files


def run_scaling_experiment():
    print("=" * 70)
    print("RUNNING LARGE-SCALE REPOSITORY SCALABILITY EXPERIMENT")
    print("=" * 70)

    repo_sizes = [10, 25, 50, 80]
    results = {
        "num_files": [],
        "num_nodes": [],
        "num_edges": [],
        "parse_time_ms": [],
        "pcr_time_ms": [],
        "uncompressed_tokens": [],
        "topocoder_tokens": []
    }

    for n_mods in repo_sizes:
        print(f"\nSynthesizing repository with {n_mods} modules...")
        files = generate_synthetic_repository(num_modules=n_mods)
        total_code_lines = sum(len(c.splitlines()) for c in files.values())
        raw_tokens = sum(len(c.split()) for c in files.values())
        print(f"  Total code lines: {total_code_lines:,} lines | Raw tokens: {raw_tokens:,}")

        # 1. Parsing & Graph Construction
        t0 = time.perf_counter()
        parser = RepositoryParser(root_path=".")
        graph = CodeKnowledgeGraph()
        for fpath, code in files.items():
            nodes, edges = parser.parse_code_string(fpath, code)
            for n in nodes.values():
                graph.add_node(n)
            for e in edges:
                graph.add_edge(e)
        parse_lat = (time.perf_counter() - t0) * 1000

        # 2. Personalized CodeRank Latency
        t0 = time.perf_counter()
        pcr = PersonalizedCodeRank(graph)
        first_node = list(graph.nodes.keys())[0]
        ranked = pcr.compute(seeds={first_node: 1.0})
        pcr_lat = (time.perf_counter() - t0) * 1000

        # 3. Context Condensation under 2,500 token budget
        condenser = TopologicalCondenser(graph, default_budget_tokens=2500)
        condensed = condenser.condense(seeds={first_node: 1.0}, budget_tokens=2500)
        tc_tokens = condensed["total_tokens"]

        print(f"  Graph Stats: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        print(f"  Parse Time: {parse_lat:.1f} ms | PCR Time: {pcr_lat:.1f} ms")
        print(f"  Raw Tokens: {raw_tokens:,} -> TopoCoder Condensed: {tc_tokens:,} tokens ({((raw_tokens - tc_tokens) / raw_tokens) * 100:.1f}% reduction)")

        results["num_files"].append(n_mods)
        results["num_nodes"].append(len(graph.nodes))
        results["num_edges"].append(len(graph.edges))
        results["parse_time_ms"].append(parse_lat)
        results["pcr_time_ms"].append(pcr_lat)
        results["uncompressed_tokens"].append(raw_tokens)
        results["topocoder_tokens"].append(tc_tokens)

    # Plot scalability curves
    generate_scaling_plots(results)
    return results


def generate_scaling_plots(results: Dict[str, List]):
    os.makedirs("paper/figures", exist_ok=True)

    # Figure: Token Economy vs Codebase Size
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    files = results["num_files"]
    ax.plot(files, results["uncompressed_tokens"], marker="s", color="#d62728", linestyle="--", linewidth=2.2, label="Raw Uncompressed Codebase")
    ax.plot(files, results["topocoder_tokens"], marker="o", color="#1f77b4", linewidth=2.5, label="TopoCoder (TCC Invariant Window)")

    ax.axhline(y=3072, color="gray", linestyle=":", label="Gemma Edge Context Budget (3k)")
    ax.set_xlabel("Number of Modules in Repository", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tokens Required for Navigation", fontsize=11, fontweight="bold")
    ax.set_title("Context Invariance: Raw Codebase vs. TopoCoder", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig.savefig("paper/figures/token_invariance.png")
    plt.close(fig)

    # Figure: Latency Overhead on Edge CPU
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(results["num_nodes"], results["parse_time_ms"], marker="^", color="#2ca02c", linewidth=2.2, label="AST Graph Extraction (ms)")
    ax.plot(results["num_nodes"], results["pcr_time_ms"], marker="o", color="#ff7f0e", linewidth=2.2, label="Personalized CodeRank (ms)")
    ax.set_xlabel("Total Code Knowledge Graph Nodes (|V|)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Latency (milliseconds)", fontsize=11, fontweight="bold")
    ax.set_title("Offline Algorithmic Latency on Consumer CPU", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig.savefig("paper/figures/latency_scaling.png")
    plt.close(fig)
    print("Saved scaling figures to paper/figures/token_invariance.png and latency_scaling.png")


run_scalability_simulation = run_scaling_experiment


if __name__ == "__main__":
    run_scaling_experiment()
