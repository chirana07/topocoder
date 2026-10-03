"""
Synthetic Agentic Trajectory & DPO Dataset Generator for Gemma Post-Training.
Generates multi-hop topological reasoning trajectories and preference pairs for
aligning Gemma on offline graph-augmented bug fixing.
"""

import json
import os
import random
from typing import Dict, List, Tuple
from benchmarks.tasks import TASKS
from codegraph.condensation import TopologicalCondenser
from codegraph.graph import CodeKnowledgeGraph
from codegraph.pagerank import PersonalizedCodeRank
from codegraph.parser import RepositoryParser


DEFECT_TEMPLATES = [
    {
        "domain": "Authentication & Token Validation",
        "seed_func": "test_token_expiry_enforced",
        "caller_func": "authenticate_bearer_token",
        "target_func": "verify_jwt_signature",
        "issue": "Expired JWT tokens are incorrectly accepted when clock_skew_seconds > 0.",
        "fix_code": "def verify_jwt_signature(token: str, secret: str, clock_skew: int = 0) -> bool:\n    import time\n    payload = decode_unverified(token)\n    exp = payload.get('exp', 0)\n    if exp + clock_skew < time.time():\n        return False\n    return compute_hmac(payload, secret) == payload.get('sig')",
        "buggy_code": "def verify_jwt_signature(token: str, secret: str, clock_skew: int = 0) -> bool:\n    payload = decode_unverified(token)\n    return compute_hmac(payload, secret) == payload.get('sig')",
        "rejected_type": "hallucination",
        "rejected_code": "import jwt_cloud_verifier_v2\ndef verify_jwt_signature(token, secret):\n    return jwt_cloud_verifier_v2.online_verify(token)"
    },
    {
        "domain": "Distributed Sharded Cache",
        "seed_func": "test_consistent_hash_ring_rebalance",
        "caller_func": "route_cache_read",
        "target_func": "locate_hash_ring_node",
        "issue": "Key lookup fails to wrap around ring modulo when hash value exceeds highest node point.",
        "fix_code": "def locate_hash_ring_node(key_hash: int, ring_nodes: list) -> str:\n    for point, node_id in ring_nodes:\n        if key_hash <= point:\n            return node_id\n    return ring_nodes[0][1] if ring_nodes else None",
        "buggy_code": "def locate_hash_ring_node(key_hash: int, ring_nodes: list) -> str:\n    for point, node_id in ring_nodes:\n        if key_hash <= point:\n            return node_id\n    return None",
        "rejected_type": "global_rewrite",
        "rejected_code": "# Rewrote entire caching infrastructure\nCACHE = {}\ndef get(k): return CACHE.get(k)"
    },
    {
        "domain": "Asynchronous Worker Queue",
        "seed_func": "test_dead_letter_retry_exhaustion",
        "caller_func": "dispatch_task_envelope",
        "target_func": "calculate_exponential_backoff",
        "issue": "Exponential backoff causes integer overflow when retry_count exceeds 30.",
        "fix_code": "def calculate_exponential_backoff(retry_count: int, base: float = 1.5, max_delay: float = 300.0) -> float:\n    bounded_count = min(retry_count, 16)\n    delay = base ** bounded_count\n    return min(delay, max_delay)",
        "buggy_code": "def calculate_exponential_backoff(retry_count: int, base: float = 1.5, max_delay: float = 300.0) -> float:\n    delay = base ** retry_count\n    return min(delay, max_delay)",
        "rejected_type": "broken_syntax",
        "rejected_code": "    def calculate_exponential_backoff(retry_count):\n    delay = base ** retry_count\n      return delay"
    },
    {
        "domain": "Database Connection Pool",
        "seed_func": "test_idle_connection_pruning",
        "caller_func": "acquire_database_connection",
        "target_func": "is_connection_stale",
        "issue": "Stale connections are returned to consumers because idle_duration uses integer millisecond truncation.",
        "fix_code": "def is_connection_stale(conn_entry: dict, max_idle_sec: float) -> bool:\n    import time\n    elapsed = time.time() - conn_entry.get('last_active', 0.0)\n    return elapsed >= max_idle_sec",
        "buggy_code": "def is_connection_stale(conn_entry: dict, max_idle_sec: float) -> bool:\n    import time\n    elapsed = int(time.time() - conn_entry.get('last_active', 0.0)) // 1000\n    return elapsed >= max_idle_sec",
        "rejected_type": "test_mutation",
        "rejected_code": "def test_idle_connection_pruning():\n    # Modify test to always pass\n    assert True"
    },
    {
        "domain": "Rate Limiter Sliding Window",
        "seed_func": "test_sliding_window_burst_mitigation",
        "caller_func": "throttle_incoming_api_call",
        "target_func": "evict_expired_window_timestamps",
        "issue": "Sliding window rate limiter evicts active timestamps if timestamp array is unsorted.",
        "fix_code": "def evict_expired_window_timestamps(timestamps: list, window_size: float, now: float) -> list:\n    cutoff = now - window_size\n    return [ts for ts in sorted(timestamps) if ts > cutoff]",
        "buggy_code": "def evict_expired_window_timestamps(timestamps: list, window_size: float, now: float) -> list:\n    cutoff = now - window_size\n    return [ts for ts in timestamps if ts < cutoff]",
        "rejected_type": "hallucination",
        "rejected_code": "import cloud_throttler\ndef evict_expired_window_timestamps(ts, w, n):\n    return cloud_throttler.filter(ts)"
    },
    {
        "domain": "JSON Serialization Pipeline",
        "seed_func": "test_decimal_serialization_preserves_precision",
        "caller_func": "serialize_payload_for_wire",
        "target_func": "encode_numeric_types",
        "issue": "High-precision financial decimals are coerced to IEEE 754 float64, losing cents precision.",
        "fix_code": "def encode_numeric_types(val):\n    from decimal import Decimal\n    if isinstance(val, Decimal):\n        return str(val)\n    return val",
        "buggy_code": "def encode_numeric_types(val):\n    from decimal import Decimal\n    if isinstance(val, Decimal):\n        return float(val)\n    return val",
        "rejected_type": "broken_syntax",
        "rejected_code": "def encode_numeric_types(val)\n    return float(val)"
    }
]


def generate_sft_dataset(output_path: str = "train/gemma_training_data.jsonl", total_count: int = 60) -> List[Dict]:
    """Generates instruction tuning trajectories in Gemma Chat format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records = []

    # 1. Base Multi-Hop Tasks (5 Core Tasks)
    for task in TASKS:
        parser = RepositoryParser(root_path=".")
        graph = CodeKnowledgeGraph()
        for fpath, code in task.files.items():
            nodes, edges = parser.parse_code_string(fpath, code)
            for n in nodes.values():
                graph.add_node(n)
            for e in edges:
                graph.add_edge(e)

        test_nodes = graph.search_nodes(task.failing_test_name)
        seed_id = test_nodes[0][0].id if test_nodes else list(graph.nodes.keys())[0]

        condenser = TopologicalCondenser(graph, default_budget_tokens=2500)
        condensed = condenser.condense(seeds={seed_id: 1.0}, budget_tokens=2500)

        target_node = graph.get_node(task.ground_truth_target)
        target_name = target_node.name if target_node else "target_func"

        prompt = f"""<start_of_turn>user
You are TopoCoder, an autonomous software engineering agent running offline on consumer hardware.
Your task is to fix a bug in the repository based on the topological context provided.

### ISSUE DESCRIPTION:
{task.issue_description}
Failing Test: {task.failing_test_name}

{condensed["formatted_slice"]}

### INSTRUCTIONS:
1. Examine the causal execution dependency chain to understand how the test interacts with the target.
2. Provide the corrected implementation for `{task.ground_truth_target}`.
3. Output the replacement code inside a ```python block. Ensure exact syntactic validity.<end_of_turn>
<start_of_turn>model
"""
        thought = (
            f"### Analysis:\n"
            f"1. **Causal Trace**: The failing test `{task.failing_test_name}` triggers execution through intermediate layers.\n"
            f"2. **Root Cause**: The defect in `{target_name}` violates the expected contract indicated in the test trace.\n"
            f"3. **Resolution**: Correct the condition while preserving argument types and return contracts.\n\n"
            f"### Verified Patch:\n"
            f"```python\n{target_node.source_code if target_node else 'def patch(): pass'}\n```"
        )
        records.append({
            "task_id": task.task_id,
            "prompt": prompt,
            "completion": thought,
            "full_text": prompt + thought + "<end_of_turn>"
        })

    # 2. Scaled Synthetic Graph Trajectories (Templates x Variations)
    idx = 1
    while len(records) < total_count:
        tmpl = DEFECT_TEMPLATES[(idx - 1) % len(DEFECT_TEMPLATES)]
        variant_id = f"SYNTH-TASK-{idx:03d}"
        
        causal_chain = f"  {tmpl['seed_func']} ──[CALLS]──> {tmpl['caller_func']} ──[CALLS]──> {tmpl['target_func']} (TARGET)"
        condensed_slice = f"""=== TOPOLOGICAL CONTEXT (Budget: 2500 tokens) ===
[CAUSAL EXECUTION CHAIN]
{causal_chain}

[FOCAL TARGET IMPLEMENTATION]
# File: repo/{tmpl['domain'].lower().replace(' ', '_')}.py
{tmpl['buggy_code']}

[INTERFACE CONTRACT STUBS]
# File: repo/caller.py
def {tmpl['caller_func']}(*args, **kwargs) -> Any:
    # [Implementation omitted]
"""
        prompt = f"""<start_of_turn>user
You are TopoCoder, an autonomous software engineering agent running offline on consumer hardware.
Your task is to fix a bug in the repository based on the topological context provided.

### ISSUE DESCRIPTION:
[{tmpl['domain']}] {tmpl['issue']}
Failing Test: {tmpl['seed_func']}

{condensed_slice}

### INSTRUCTIONS:
1. Examine the causal execution dependency chain to understand how the test interacts with the target.
2. Provide the corrected implementation for `{tmpl['target_func']}`.
3. Output the replacement code inside a ```python block. Ensure exact syntactic validity.<end_of_turn>
<start_of_turn>model
"""
        thought = (
            f"### Analysis:\n"
            f"1. **Causal Trace**: `{tmpl['seed_func']}` calls `{tmpl['caller_func']}`, which invokes `{tmpl['target_func']}`.\n"
            f"2. **Root Cause**: In `{tmpl['target_func']}`, {tmpl['issue'].lower()}.\n"
            f"3. **Resolution**: Apply the verified bugfix preserving contract semantics.\n\n"
            f"### Verified Patch:\n"
            f"```python\n{tmpl['fix_code']}\n```"
        )
        records.append({
            "task_id": variant_id,
            "prompt": prompt,
            "completion": thought,
            "full_text": prompt + thought + "<end_of_turn>"
        })
        idx += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(records)} scaled SFT trajectories in {output_path}")
    return records


def generate_dpo_preference_dataset(output_path: str = "train/dpo_preference_pairs.jsonl", total_count: int = 60) -> List[Dict]:
    """Generates Direct Preference Optimization (DPO) chosen vs rejected pairs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pairs = []

    # 1. Base Multi-Hop Tasks (5 Core Tasks)
    for task in TASKS:
        parser = RepositoryParser(root_path=".")
        graph = CodeKnowledgeGraph()
        for fpath, code in task.files.items():
            nodes, edges = parser.parse_code_string(fpath, code)
            for n in nodes.values():
                graph.add_node(n)
            for e in edges:
                graph.add_edge(e)

        target_node = graph.get_node(task.ground_truth_target)

        chosen = (
            f"### Analysis:\nIdentified root cause in `{task.ground_truth_target}` via topological call path.\n"
            f"```python\n{target_node.source_code if target_node else ''}\n```"
        )
        rejected = (
            f"### Analysis:\nI will rewrite the entire module and add external dependencies.\n"
            f"```python\n# [Hallucinated Rewrite]\nimport cloud_sdk_non_existent\ndef rewrite_everything():\n    return 401\n```"
        )
        pairs.append({
            "task_id": task.task_id,
            "prompt": f"Fix issue: {task.issue_description}",
            "chosen": chosen,
            "rejected": rejected
        })

    # 2. Scaled Synthetic DPO Pairs
    idx = 1
    while len(pairs) < total_count:
        tmpl = DEFECT_TEMPLATES[(idx - 1) % len(DEFECT_TEMPLATES)]
        variant_id = f"DPO-PAIR-{idx:03d}"

        chosen = (
            f"### Analysis:\n"
            f"1. Root cause identified in `{tmpl['target_func']}` via causal execution chain.\n"
            f"2. Applying localized, verified patch maintaining interface contract.\n\n"
            f"```python\n{tmpl['fix_code']}\n```"
        )
        rejected = (
            f"### Analysis:\n"
            f"Unconstrained modification violating topological boundary constraints.\n\n"
            f"```python\n{tmpl['rejected_code']}\n```"
        )
        pairs.append({
            "task_id": variant_id,
            "prompt": f"[{tmpl['domain']}] {tmpl['issue']}",
            "chosen": chosen,
            "rejected": rejected,
            "failure_mode": tmpl["rejected_type"]
        })
        idx += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Generated {len(pairs)} scaled DPO preference pairs in {output_path}")
    return pairs


if __name__ == "__main__":
    generate_sft_dataset(total_count=60)
    generate_dpo_preference_dataset(total_count=60)
