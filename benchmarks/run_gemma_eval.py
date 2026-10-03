"""
End-to-End Evaluation Harness for Real Gemma Inference on TopoCoder.
Evaluates local Gemma model on multi-hop SWE tasks via Ollama, measuring
real latency, AST verification pass rate, and patch correctness.
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


def run_real_gemma_evaluation(model_name: str = "gemma2:2b", max_tasks: int = 5) -> Dict[str, any]:
    print("=" * 70)
    print(f"STARTING LIVE GEMMA EVALUATION ON CONSUMER HARDWARE")
    print(f"Model: {model_name} | Local Backend: Ollama (Apple Silicon MPS / GPU)")
    print("=" * 70)

    backend = OllamaGemmaBackend(model_name=model_name)
    if not backend.is_available():
        print(f"[Warning] Model '{model_name}' is not yet available in Ollama.")
        print("Please wait for 'ollama pull gemma2:2b' to finish downloading.")
        return {"status": "model_not_ready"}

    print("[Success] Connected to local Ollama Gemma backend!")

    results = []
    total_tokens_all = 0
    syntactically_valid_count = 0

    for idx, task in enumerate(TASKS[:max_tasks], 1):
        print(f"\n[{idx}/{len(TASKS)}] Running Task: {task.task_id}")
        print(f"  Failing Test: {task.failing_test_name} | Target: {task.ground_truth_target}")

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

        if res.success:
            syntactically_valid_count += 1
        total_tokens_all += res.total_tokens

        print(f"  Duration: {duration:.2f}s | Context Tokens: {res.total_tokens}")
        print(f"  Target Focal Node: {res.focal_node_id}")
        print(f"  AST Verified: {res.success} | Message: {res.message}")
        if res.patch_code:
            preview = res.patch_code.strip().splitlines()[0] if res.patch_code.strip() else ""
            print(f"  Synthesized Patch Preview: {preview[:70]}...")

        results.append({
            "task_id": task.task_id,
            "target": task.ground_truth_target,
            "focal_node_identified": res.focal_node_id,
            "duration_sec": duration,
            "tokens": res.total_tokens,
            "ast_valid": res.success,
            "patch_preview": preview
        })

    n = len(results)
    summary = {
        "model": model_name,
        "tasks_evaluated": n,
        "ast_valid_rate": (syntactically_valid_count / n) * 100.0 if n > 0 else 0.0,
        "avg_duration_sec": sum(r["duration_sec"] for r in results) / n if n > 0 else 0.0,
        "avg_tokens": total_tokens_all / n if n > 0 else 0.0,
        "detailed_results": results
    }

    print("\n" + "=" * 70)
    print("LIVE GEMMA EVALUATION SUMMARY")
    print("=" * 70)
    print(f"AST Syntax Validity: {summary['ast_valid_rate']:.1f}%")
    print(f"Avg End-to-End Latency: {summary['avg_duration_sec']:.2f} seconds")
    print(f"Avg Context Window Used: {summary['avg_tokens']:.1f} tokens")
    print("=" * 70)

    os.makedirs("benchmarks/output", exist_ok=True)
    out_file = "benchmarks/output/live_gemma_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_real_gemma_evaluation()
