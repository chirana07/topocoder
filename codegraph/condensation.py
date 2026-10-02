"""
Topological Context Condensation (TCC) engine.
Solves the bounded token knapsack problem by dynamically abstracting code entities into
full implementation, AST signature stubs, and relational dependency paths under a strict token budget.
"""

from typing import Dict, List, Optional, Set, Tuple
from codegraph.graph import CodeKnowledgeGraph
from codegraph.pagerank import PersonalizedCodeRank
from codegraph.schema import CodeNode, EdgeType, NodeType


class TopologicalCondenser:
    """Condenses multi-relational code graphs into token-bounded agent prompts."""

    def __init__(self, graph: CodeKnowledgeGraph, default_budget_tokens: int = 2500):
        self.graph = graph
        self.default_budget_tokens = default_budget_tokens
        self.pcr = PersonalizedCodeRank(graph)

    def condense(
        self,
        seeds: Dict[str, float],
        budget_tokens: Optional[int] = None,
        max_focal_nodes: int = 3,
        reserve_prompt_tokens: int = 400
    ) -> Dict[str, any]:
        """
        Executes bounded subgraph condensation.

        Args:
            seeds: Mapping from seed node IDs -> initial weights (e.g. failing test or issue match)
            budget_tokens: Total token ceiling for the context window
            max_focal_nodes: Max number of nodes allowed to have full implementation
            reserve_prompt_tokens: Overhead buffer for instructions and scaffolding

        Returns:
            Dictionary with formatted prompt string, token metrics, and included node IDs.
        """
        effective_budget = (budget_tokens or self.default_budget_tokens) - reserve_prompt_tokens
        scores = self.pcr.compute(seeds)

        # Rank all nodes by Personalized CodeRank
        ranked_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        full_nodes: List[CodeNode] = []
        stub_nodes: List[CodeNode] = []
        consumed_tokens = 0

        # Step 1: Select Focal Nodes (highest PCR scores with non-trivial code)
        focal_candidates = [
            self.graph.nodes[nid] for nid, _ in ranked_nodes
            if nid in self.graph.nodes and self.graph.nodes[nid].node_type in (NodeType.FUNCTION, NodeType.METHOD, NodeType.TEST)
        ]

        focal_ids: Set[str] = set()
        for node in focal_candidates[:max_focal_nodes]:
            cost = node.token_count + 30  # code tokens + markdown header overhead
            if consumed_tokens + cost <= effective_budget * 0.7:  # Allocate up to 70% budget to full bodies
                full_nodes.append(node)
                focal_ids.add(node.id)
                consumed_tokens += cost

        # Step 2: Extract 1-hop Relational Context Neighbors for focal nodes
        context_ids: Set[str] = set()
        for fn in full_nodes:
            # Immediate callers and callees
            for caller in self.graph.get_callers(fn.id):
                if caller.id not in focal_ids:
                    context_ids.add(caller.id)
            for callee in self.graph.get_callees(fn.id):
                if callee.id not in focal_ids:
                    context_ids.add(callee.id)
            # Parent class if applicable
            parent = self.graph.get_parent(fn.id)
            if parent and parent.id not in focal_ids:
                context_ids.add(parent.id)

        # Add remaining high-ranking nodes to context
        for nid, _ in ranked_nodes:
            if nid not in focal_ids and nid in self.graph.nodes:
                context_ids.add(nid)
                if len(context_ids) > 15:
                    break

        # Step 3: Pack Structural AST Stubs for context nodes
        context_nodes_sorted = sorted(
            [self.graph.nodes[cid] for cid in context_ids],
            key=lambda n: scores.get(n.id, 0.0),
            reverse=True
        )

        for cnode in context_nodes_sorted:
            stub_cost = cnode.stub_token_count + 15
            if consumed_tokens + stub_cost <= effective_budget:
                stub_nodes.append(cnode)
                consumed_tokens += stub_cost
            else:
                break

        # Step 4: Extract Relational Path Chains (Causal flow between seeds and focal nodes)
        relational_paths = self._extract_relational_chains(seeds, focal_ids)

        # Step 5: Format Markdown Output
        formatted_slice = self._format_condensed_slice(full_nodes, stub_nodes, relational_paths)

        return {
            "formatted_slice": formatted_slice,
            "focal_node_ids": [n.id for n in full_nodes],
            "stub_node_ids": [n.id for n in stub_nodes],
            "total_tokens": consumed_tokens,
            "budget_tokens": effective_budget,
            "full_node_count": len(full_nodes),
            "stub_node_count": len(stub_nodes),
            "relational_paths": relational_paths
        }

    def _extract_relational_chains(self, seeds: Dict[str, float], focal_ids: Set[str]) -> List[str]:
        """Extracts readable execution dependency paths connecting seeds to focal nodes."""
        chains = []
        for s_id in seeds:
            if s_id in self.graph.nodes:
                callees = self.graph.get_callees(s_id)
                for callee in callees:
                    if callee.id in focal_ids:
                        chains.append(f"`{s_id}` ──[CALLS]──> `{callee.id}` (Focal Root-Cause Candidate)")
                    else:
                        # Check 2-hop
                        sub_callees = self.graph.get_callees(callee.id)
                        for sc in sub_callees:
                            if sc.id in focal_ids:
                                chains.append(f"`{s_id}` ──[CALLS]──> `{callee.id}` ──[CALLS]──> `{sc.id}` (Focal Candidate)")
        return list(set(chains))[:5]

    def _format_condensed_slice(
        self,
        full_nodes: List[CodeNode],
        stub_nodes: List[CodeNode],
        relational_paths: List[str]
    ) -> str:
        """Formats the compressed representation into a structured LLM prompt context."""
        parts = ["### [TOPOLOGICAL CODEBASE CONTEXT]"]

        # 1. Relational Graph Flow
        if relational_paths:
            parts.append("#### Causal Execution Dependencies:")
            for p in relational_paths:
                parts.append(f"- {p}")

        # 2. Focal Nodes (Full Code)
        if full_nodes:
            parts.append("\n#### Focal Implementation Targets (High Topological Relevance):")
            for node in full_nodes:
                parts.append(
                    f"```python\n# File: {node.file_path} (Lines {node.start_line}-{node.end_line}) | Node: {node.id}\n{node.source_code}\n```"
                )

        # 3. Context Nodes (AST Signature Stubs)
        if stub_nodes:
            parts.append("\n#### Boundary Context & Call Interface Skeletons:")
            for node in stub_nodes:
                parts.append(
                    f"```python\n# File: {node.file_path} | {node.id}\n{node.stub}\n```"
                )

        return "\n".join(parts)
