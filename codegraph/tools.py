"""
Standardized tool suite exposing H-CKG operations to the Gemma Developer Agent.
Provides graph navigation, symbol queries, topological slicing, and inspection.
"""

from typing import Any, Dict, List, Optional
from codegraph.condensation import TopologicalCondenser
from codegraph.graph import CodeKnowledgeGraph
from codegraph.pagerank import PersonalizedCodeRank
from codegraph.schema import NodeType


class CodeGraphToolSuite:
    """Tool execution interface for agentic reasoning over the codebase graph."""

    def __init__(self, graph: CodeKnowledgeGraph):
        self.graph = graph
        self.pcr = PersonalizedCodeRank(graph)
        self.condenser = TopologicalCondenser(graph)

    def get_repo_summary(self) -> Dict[str, Any]:
        """Returns high-level structural statistics and key modules."""
        stats = self.graph.stats()
        modules = [
            n.file_path for n in self.graph.nodes.values()
            if n.node_type == NodeType.MODULE
        ]
        return {
            "stats": stats,
            "modules": sorted(list(set(modules)))[:25]
        }

    def search_symbols(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Searches symbols (functions, classes, tests) matching the query string."""
        results = self.graph.search_nodes(query, limit=limit)
        return [
            {
                "id": node.id,
                "name": node.name,
                "type": node.node_type.value,
                "file_path": node.file_path,
                "signature": node.signature,
                "line_range": f"{node.start_line}-{node.end_line}",
                "relevance_score": round(score, 2)
            }
            for node, score in results
        ]

    def get_callers(self, node_id: str) -> List[Dict[str, Any]]:
        """Returns functions/methods that invoke the given node."""
        callers = self.graph.get_callers(node_id)
        return [
            {
                "id": c.id,
                "name": c.name,
                "type": c.node_type.value,
                "file_path": c.file_path,
                "signature": c.signature,
                "line_range": f"{c.start_line}-{c.end_line}"
            }
            for c in callers
        ]

    def get_callees(self, node_id: str) -> List[Dict[str, Any]]:
        """Returns functions/methods invoked by the given node."""
        callees = self.graph.get_callees(node_id)
        return [
            {
                "id": c.id,
                "name": c.name,
                "type": c.node_type.value,
                "file_path": c.file_path,
                "signature": c.signature,
                "line_range": f"{c.start_line}-{c.end_line}"
            }
            for c in callees
        ]

    def inspect_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves full source code and metadata for a specific node."""
        node = self.graph.get_node(node_id)
        if not node:
            return None
        return {
            "id": node.id,
            "type": node.node_type.value,
            "name": node.name,
            "file_path": node.file_path,
            "signature": node.signature,
            "line_range": f"{node.start_line}-{node.end_line}",
            "docstring": node.docstring,
            "source_code": node.source_code,
            "token_count": node.token_count
        }

    def get_condensed_context(
        self,
        seed_names_or_ids: List[str],
        budget_tokens: int = 2500
    ) -> Dict[str, Any]:
        """
        Computes Personalized CodeRank from seed symbols and returns a token-bounded context slice.
        """
        seeds: Dict[str, float] = {}
        for s in seed_names_or_ids:
            if s in self.graph.nodes:
                seeds[s] = 1.0
            else:
                matches = self.graph.search_nodes(s, limit=2)
                for node, score in matches:
                    seeds[node.id] = max(seeds.get(node.id, 0.0), score)

        if not seeds:
            # Fallback: all tests or top modules
            for nid, n in self.graph.nodes.items():
                if n.node_type == NodeType.TEST:
                    seeds[nid] = 1.0

        return self.condenser.condense(seeds, budget_tokens=budget_tokens)
