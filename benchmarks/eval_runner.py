"""
Automated empirical evaluation runner comparing TopoCoder against baselines on multi-hop SWE tasks.
Computes Recall@1, Recall@3, Token Economy, and generates publication tables and figures.
"""

import json
import os
import time
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.tasks import TASKS
from benchmarks.baselines import ReActGrepBaseline, BM25RAGBaseline, FullGraphBaseline, TopoCoderEvaluator


def run_all_evaluations() -> Dict[str, any]:
    print("=" * 70)
    print("STARTING TOPOCODER EMPIRICAL BENCHMARK EVALUATION")
    print(f"Total Multi-Hop Repository Tasks: {len(TASKS)}")
    print("=" * 70)

    results = {
        "ReAct-Grep": {"hits": 0, "total_tokens": 0, "latencies": []},
        "BM25-RAG": {"hits": 0, "total_tokens": 0, "latencies": []},
        "Full-Graph": {"hits": 0, "total_tokens": 0, "latencies": []},
        "TopoCoder": {"hits_top1": 0, "hits_top3": 0, "hits_top5": 0, "total_tokens": 0, "latencies": []}
    }

    per_task_logs = []

    for task in TASKS:
        print(f"\nEvaluating {task.task_id}: {task.title}")
        print(f"  Hop distance to defect: {task.hop_distance} hops | Target: {task.ground_truth_target}")

        # 1. ReAct-Grep
        t0 = time.perf_counter()
        grep_res = ReActGrepBaseline(task).run()
        grep_lat = (time.perf_counter() - t0) * 1000
        if grep_res["target_found"]:
            results["ReAct-Grep"]["hits"] += 1
        results["ReAct-Grep"]["total_tokens"] += grep_res["total_tokens"]
        results["ReAct-Grep"]["latencies"].append(grep_lat)

        # 2. BM25-RAG
        t0 = time.perf_counter()
        rag_res = BM25RAGBaseline(task).run(top_k=5)
        rag_lat = (time.perf_counter() - t0) * 1000
        if rag_res["target_found"]:
            results["BM25-RAG"]["hits"] += 1
        results["BM25-RAG"]["total_tokens"] += rag_res["total_tokens"]
        results["BM25-RAG"]["latencies"].append(rag_lat)

        # 3. Full-Graph (uncompressed)
        t0 = time.perf_counter()
        fg_res = FullGraphBaseline(task).run(k=task.hop_distance)
        fg_lat = (time.perf_counter() - t0) * 1000
        if fg_res["target_found"]:
            results["Full-Graph"]["hits"] += 1
        results["Full-Graph"]["total_tokens"] += fg_res["total_tokens"]
        results["Full-Graph"]["latencies"].append(fg_lat)

        # 4. TopoCoder
        t0 = time.perf_counter()
        tc_res = TopoCoderEvaluator(task, token_budget=1500).run()
        tc_lat = (time.perf_counter() - t0) * 1000
        if tc_res["target_found_top1"]:
            results["TopoCoder"]["hits_top1"] += 1
        if tc_res["target_found_top3"]:
            results["TopoCoder"]["hits_top3"] += 1
        if tc_res["target_found_top5"]:
            results["TopoCoder"]["hits_top5"] += 1
        results["TopoCoder"]["total_tokens"] += tc_res["total_tokens"]
        results["TopoCoder"]["latencies"].append(tc_lat)

        per_task_logs.append({
            "task_id": task.task_id,
            "hop_distance": task.hop_distance,
            "grep_hit": grep_res["target_found"],
            "rag_hit": rag_res["target_found"],
            "full_graph_hit": fg_res["target_found"],
            "topocoder_hit_top1": tc_res["target_found_top1"],
            "topocoder_hit_top3": tc_res["target_found_top3"],
            "topocoder_hit_top5": tc_res["target_found_top5"],
            "grep_tokens": grep_res["total_tokens"],
            "rag_tokens": rag_res["total_tokens"],
            "full_graph_tokens": fg_res["total_tokens"],
            "topocoder_tokens": tc_res["total_tokens"]
        })

    n = len(TASKS)
    avg_tokens = {
        "ReAct-Grep": results["ReAct-Grep"]["total_tokens"] / n,
        "BM25-RAG": results["BM25-RAG"]["total_tokens"] / n,
        "Full-Graph": results["Full-Graph"]["total_tokens"] / n,
        "TopoCoder": results["TopoCoder"]["total_tokens"] / n,
    }

    acc = {
        "ReAct-Grep": (results["ReAct-Grep"]["hits"] / n) * 100.0,
        "BM25-RAG": (results["BM25-RAG"]["hits"] / n) * 100.0,
        "Full-Graph": (results["Full-Graph"]["hits"] / n) * 100.0,
        "TopoCoder (Top-1)": (results["TopoCoder"]["hits_top1"] / n) * 100.0,
        "TopoCoder (Top-3)": (results["TopoCoder"]["hits_top3"] / n) * 100.0,
        "TopoCoder (Top-5)": (results["TopoCoder"]["hits_top5"] / n) * 100.0,
    }

    token_reduction = ((avg_tokens["Full-Graph"] - avg_tokens["TopoCoder"]) / avg_tokens["Full-Graph"]) * 100.0

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 70)
    print(f"{'Method':<20} | {'Recall@3 (%)':<15} | {'Recall@5 (%)':<15} | {'Avg Tokens/Task':<16}")
    print("-" * 75)
    print(f"{'ReAct-Grep':<20} | {acc['ReAct-Grep']:<15.1f} | {'N/A':<15} | {avg_tokens['ReAct-Grep']:<16.1f}")
    print(f"{'BM25-RAG':<20} | {acc['BM25-RAG']:<15.1f} | {'N/A':<15} | {avg_tokens['BM25-RAG']:<16.1f}")
    print(f"{'Full-Graph':<20} | {acc['Full-Graph']:<15.1f} | {'N/A':<15} | {avg_tokens['Full-Graph']:<16.1f}")
    print(f"{'TopoCoder (Ours)':<20} | {acc['TopoCoder (Top-3)']:<15.1f} | {acc['TopoCoder (Top-5)']:<15.1f} | {avg_tokens['TopoCoder']:<16.1f}")
    print("=" * 75)

    summary_data = {
        "accuracy": acc,
        "avg_tokens": avg_tokens,
        "token_reduction_pct": token_reduction,
        "task_logs": per_task_logs
    }

    # Save to JSON
    os.makedirs("benchmarks/output", exist_ok=True)
    with open("benchmarks/output/results.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # Generate Figures
    generate_figures(summary_data)
    return summary_data


def generate_figures(summary_data: Dict[str, any]):
    """Generates publication-quality charts for the paper and writeup."""
    os.makedirs("paper/figures", exist_ok=True)
    
    # 1. Token Consumption Bar Chart
    methods = ["ReAct-Grep", "BM25-RAG", "Full-Graph", "TopoCoder (Ours)"]
    tokens = [
        summary_data["avg_tokens"]["ReAct-Grep"],
        summary_data["avg_tokens"]["BM25-RAG"],
        summary_data["avg_tokens"]["Full-Graph"],
        summary_data["avg_tokens"]["TopoCoder"]
    ]
    colors = ["#7f7f7f", "#bcbd22", "#17becf", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    bars = ax.bar(methods, tokens, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Average Prompt Tokens per Task", fontsize=11, fontweight="bold")
    ax.set_title("Context Window Footprint on Consumer Hardware", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 10, f"{int(yval)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig("paper/figures/token_consumption.png")
    plt.close(fig)

    # 2. Multi-Hop Recall by Distance
    task_logs = summary_data["task_logs"]
    hop_groups = {}
    for log in task_logs:
        h = log["hop_distance"]
        if h not in hop_groups:
            hop_groups[h] = {"grep": [], "rag": [], "topocoder": []}
        hop_groups[h]["grep"].append(1 if log["grep_hit"] else 0)
        hop_groups[h]["rag"].append(1 if log["rag_hit"] else 0)
        hop_groups[h]["topocoder"].append(1 if log["topocoder_hit_top3"] else 0)

    hops = sorted(list(hop_groups.keys()))
    grep_rec = [np.mean(hop_groups[h]["grep"]) * 100 for h in hops]
    rag_rec = [np.mean(hop_groups[h]["rag"]) * 100 for h in hops]
    tc_rec = [np.mean(hop_groups[h]["topocoder"]) * 100 for h in hops]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(hops, tc_rec, marker="o", linewidth=2.5, color="#1f77b4", label="TopoCoder (Ours)", markersize=8)
    ax.plot(hops, grep_rec, marker="s", linewidth=2.0, linestyle="--", color="#7f7f7f", label="ReAct-Grep", markersize=7)
    ax.plot(hops, rag_rec, marker="^", linewidth=2.0, linestyle=":", color="#d62728", label="BM25-RAG", markersize=7)

    ax.set_xlabel("Relational Hop Distance to Defect", fontsize=11, fontweight="bold")
    ax.set_ylabel("Bug Localization Recall@3 (%)", fontsize=11, fontweight="bold")
    ax.set_title("Multi-Hop Degradation: Flat Search vs. Topological Walk", fontsize=12, fontweight="bold")
    ax.set_xticks(hops)
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    fig.savefig("paper/figures/hop_degradation.png")
    plt.close(fig)
    print("Generated publication figures in paper/figures/")


run_evaluation_suite = run_all_evaluations


if __name__ == "__main__":
    run_all_evaluations()
