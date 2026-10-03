"""
Master Replication & Verification Runner for TopoCoder Research Artifacts.
Executes the full experimental battery: unit tests, multi-hop defect benchmark,
repository scaling simulation, real-world library evaluation, post-training loss
convergence, and PDF paper compilation.
"""

import json
import os
import subprocess
import sys
import time


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)


def run_command(cmd: str, desc: str) -> bool:
    print(f"\n[RUNNING] {desc}...")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    dur = time.time() - t0
    if res.returncode == 0:
        print(f"[PASSED] {desc} completed in {dur:.2f}s")
        return True
    else:
        print(f"[FAILED] {desc} failed (exit code {res.returncode}):")
        print(res.stderr[:500])
        return False


def verify_word_count(path: str = "paper/kaggle_writeup.md", max_words: int = 3000) -> int:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    words = len(text.split())
    status = "VALID" if words <= max_words else "OVER LIMIT"
    print(f"\n[AUDIT] Kaggle Writeup Word Count: {words:,} / {max_words:,} words [{status}]")
    return words


def main():
    print_header("TopoCoder Full Master Reproduction Pipeline")
    start_time = time.time()
    results = {}

    # 1. Unit Tests
    results["unit_tests"] = run_command("PYTHONPATH=. pytest tests/ -v", "Unit Tests (10/10)")

    # 2. Multi-Hop Defect Localization Benchmark
    results["benchmark"] = run_command("PYTHONPATH=. python3 benchmarks/eval_runner.py", "Multi-Hop Defect Localization (TopoBench)")

    # 3. Scalability Simulation (22,000 LOC)
    results["scalability"] = run_command("PYTHONPATH=. python3 benchmarks/large_repo_simulation.py", "Large-Scale Repository Invariance Simulation")

    # 4. Real-World Production Codebases
    results["real_world"] = run_command("PYTHONPATH=. python3 benchmarks/real_world.py", "Real-World Standard Library Evaluation")

    # 5. Scaled Training & DPO Dataset Generation
    results["dataset_gen"] = run_command("PYTHONPATH=. python3 train/generate_training_data.py", "Synthetic SFT & DPO Dataset Generation")

    # 6. LoRA Post-Training Loss Convergence
    results["post_training"] = run_command("python3 train/train_lora.py --steps 60", "LoRA Post-Training Simulation & Loss Logging")

    # 7. LaTeX Compilation
    results["paper_pdf"] = run_command("cd paper && pdflatex -interaction=nonstopmode main.tex > /dev/null && pdflatex -interaction=nonstopmode main.tex > /dev/null", "LaTeX Publication Paper PDF Compilation")

    # 8. Word Count Audit
    word_count = verify_word_count()
    results["word_count_compliance"] = word_count <= 3000

    # Summary
    print_header("Reproduction Pipeline Scorecard")
    all_passed = True
    for item, passed in results.items():
        tag = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"  * {item.replace('_', ' ').title():<35}: [{tag}]")

    total_duration = time.time() - start_time
    print(f"\nTotal Pipeline Execution Time: {total_duration:.2f} seconds")
    print(f"Overall Status: {'ALL CHECKS PASSED - READY FOR SUBMISSION' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 80 + "\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
