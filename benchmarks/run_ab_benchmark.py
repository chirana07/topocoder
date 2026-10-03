"""
A/B Comparative Benchmark Harness: Base Gemma-2-2B vs. Post-Trained TopoCoder-Gemma-2B.
Compares inference latency, token economy, AST syntax validity, and format adherence
on real multi-hop SWE tasks running locally on Apple Silicon MPS via Ollama.
"""

import json
import os
import time
from typing import Dict, List
from agent.dual_process import DualProcessGemmaAgent
from agent.gemma_backend import OllamaGemmaBackend
from benchmarks.tasks import TASKS
from codegraph.graph import CodeKnowledgeGraph
from codegraph.parser import RepositoryParser


def evaluate_model_on_tasks(model_name: str, tasks: List, max_tasks: int = 5) -> Dict:
    print(f"\nEvaluating Model: {model_name}")
    backend = OllamaGemmaBackend(model_name=model_name)
    if not backend.is_available():
        raise RuntimeError(f"Model '{model_name}' is not accessible in Ollama.")

    results = []
    for idx, task in enumerate(tasks[:max_tasks], 1):
        parser = RepositoryParser(root_path=".")
        graph = CodeKnowledgeGraph()
        for fpath, code in task.files.items():
            nodes, edges = parser.parse_code_string(fpath, code)
            for n in nodes.values():
                graph.add_node(n)
            for e in edges:
                graph.add_edge(e)

        agent = DualProcessGemmaAgent(graph=graph, token_budget=2500, model_name=model_name)
        t0 = time.perf_counter()
        res = agent.solve_task(
            issue_description=task.issue_description,
            failing_test_name=task.failing_test_name,
            llm_backend=backend
        )
        duration = time.perf_counter() - t0

        # Check format adherence (clean python code block without conversational rambling)
        clean_format = bool(res.patch_code and ("def " in res.patch_code or "class " in res.patch_code))

        results.append({
            "task_id": task.task_id,
            "duration_sec": duration,
            "tokens": res.total_tokens,
            "ast_valid": res.success,
            "clean_format": clean_format
        })
        print(f"  [{idx}/{max_tasks}] {task.task_id} -> Valid: {res.success} | Latency: {duration:.2f}s | Tokens: {res.total_tokens}")

    n = len(results)
    return {
        "model": model_name,
        "tasks_count": n,
        "ast_valid_rate": (sum(1 for r in results if r["ast_valid"]) / n) * 100.0,
        "clean_format_rate": (sum(1 for r in results if r["clean_format"]) / n) * 100.0,
        "avg_duration_sec": round(sum(r["duration_sec"] for r in results) / n, 2),
        "avg_tokens": round(sum(r["tokens"] for r in results) / n, 1),
        "task_details": results
    }


def run_ab_comparison():
    print("=" * 75)
    print("A/B COMPARATIVE BENCHMARK: BASE GEMMA-2-2B vs. TOPOCODER-GEMMA-2B")
    print("=" * 75)

    base_summary = evaluate_model_on_tasks("gemma2:2b", TASKS)
    topocoder_summary = evaluate_model_on_tasks("topocoder-gemma:2b", TASKS)

    comparison = {
        "base_model": base_summary,
        "post_trained_model": topocoder_summary,
        "comparison_delta": {
            "ast_valid_gain": topocoder_summary["ast_valid_rate"] - base_summary["ast_valid_rate"],
            "latency_delta_sec": round(topocoder_summary["avg_duration_sec"] - base_summary["avg_duration_sec"], 2),
            "token_delta": round(topocoder_summary["avg_tokens"] - base_summary["avg_tokens"], 1),
            "format_adherence_gain": topocoder_summary["clean_format_rate"] - base_summary["clean_format_rate"]
        }
    }

    os.makedirs("benchmarks/output", exist_ok=True)
    out_file = "benchmarks/output/ab_comparison_results.json"
    with open(out_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 75)
    print("A/B BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"Base Gemma-2-2B:        AST Valid: {base_summary['ast_valid_rate']:.1f}% | Avg Latency: {base_summary['avg_duration_sec']}s | Tokens: {base_summary['avg_tokens']}")
    print(f"Post-Trained TopoCoder: AST Valid: {topocoder_summary['ast_valid_rate']:.1f}% | Avg Latency: {topocoder_summary['avg_duration_sec']}s | Tokens: {topocoder_summary['avg_tokens']}")
    print(f"Results saved to: {out_file}")
    print("=" * 75)


if __name__ == "__main__":
    run_ab_comparison()
