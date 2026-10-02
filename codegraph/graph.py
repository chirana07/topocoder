"""
Multi-Relational Code Knowledge Graph (H-CKG) representation and query engine.
Supports structural navigation, topological neighborhood retrieval, and graph analytics.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

from codegraph.schema import CodeEdge, CodeNode, EdgeType, NodeType


class CodeKnowledgeGraph:
    """In-memory multi-relational knowledge graph representing codebase structure and behavior."""

    def __init__(self):
        self.nodes: Dict[str, CodeNode] = {}
        self.edges: List[CodeEdge] = []
        # Adjacency indexes: source_id -> edge_type -> list of edges
        self.out_edges: Dict[str, Dict[EdgeType, List[CodeEdge]]] = defaultdict(lambda: defaultdict(list))
        # Reverse adjacency: target_id -> edge_type -> list of edges
        self.in_edges: Dict[str, Dict[EdgeType, List[CodeEdge]]] = defaultdict(lambda: defaultdict(list))
        # Secondary indexes
        self.file_index: Dict[str, List[str]] = defaultdict(list)
        self.name_index: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, node: CodeNode) -> None:
        self.nodes[node.id] = node
        self.file_index[node.file_path].append(node.id)
        self.name_index[node.name.lower()].append(node.id)

    def add_edge(self, edge: CodeEdge) -> None:
        if edge.source_id in self.nodes and edge.target_id in self.nodes:
            self.edges.append(edge)
            self.out_edges[edge.source_id][edge.edge_type].append(edge)
            self.in_edges[edge.target_id][edge.edge_type].append(edge)

    def get_node(self, node_id: str) -> Optional[CodeNode]:
        return self.nodes.get(node_id)

    def get_callers(self, node_id: str) -> List[CodeNode]:
        """Finds all functions/methods that call this node."""
        incoming = self.in_edges[node_id].get(EdgeType.CALLS, [])
        incoming += self.in_edges[node_id].get(EdgeType.TESTS, [])
        callers = []
        for e in incoming:
            node = self.nodes.get(e.source_id)
            if node:
                callers.append(node)
        return callers

    def get_callees(self, node_id: str) -> List[CodeNode]:
        """Finds all functions/methods that this node calls."""
        outgoing = self.out_edges[node_id].get(EdgeType.CALLS, [])
        outgoing += self.out_edges[node_id].get(EdgeType.TESTS, [])
        callees = []
        for e in outgoing:
            node = self.nodes.get(e.target_id)
            if node:
                callees.append(node)
        return callees

    def get_children(self, node_id: str) -> List[CodeNode]:
        """Finds entities contained within this node (e.g. methods in a class)."""
        outgoing = self.out_edges[node_id].get(EdgeType.CONTAINS, [])
        return [self.nodes[e.target_id] for e in outgoing if e.target_id in self.nodes]

    def get_parent(self, node_id: str) -> Optional[CodeNode]:
        """Finds the parent entity containing this node."""
        incoming = self.in_edges[node_id].get(EdgeType.CONTAINS, [])
        if incoming:
            return self.nodes.get(incoming[0].source_id)
        return None

    def get_k_hop_neighborhood(self, seed_ids: List[str], k: int = 1, allowed_relations: Optional[Set[EdgeType]] = None) -> Set[str]:
        """Extracts the set of node IDs within k relational hops from seeds."""
        visited: Set[str] = set()
        frontier: Set[str] = set(s for s in seed_ids if s in self.nodes)

        for _ in range(k + 1):
            next_frontier: Set[str] = set()
            for nid in frontier:
                if nid not in visited:
                    visited.add(nid)
                    # Outgoing neighbors
                    for rel, edges in self.out_edges[nid].items():
                        if allowed_relations is None or rel in allowed_relations:
                            for e in edges:
                                if e.target_id not in visited:
                                    next_frontier.add(e.target_id)
                    # Incoming neighbors
                    for rel, edges in self.in_edges[nid].items():
                        if allowed_relations is None or rel in allowed_relations:
                            for e in edges:
                                if e.source_id not in visited:
                                    next_frontier.add(e.source_id)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    def search_nodes(self, query: str, limit: int = 10) -> List[Tuple[CodeNode, float]]:
        """Performs lexical and structural symbol lookup."""
        q_lower = query.lower().strip()
        matches: List[Tuple[CodeNode, float]] = []

        for nid, node in self.nodes.items():
            score = 0.0
            name_lower = node.name.lower()
            if name_lower == q_lower:
                score += 10.0
            elif q_lower in name_lower:
                score += 5.0
            elif q_lower in node.id.lower():
                score += 3.0
            elif q_lower in node.docstring.lower():
                score += 1.0

            if score > 0:
                matches.append((node, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]

    def to_networkx(self) -> nx.MultiDiGraph:
        """Converts internal graph to NetworkX MultiDiGraph for graph algorithms."""
        G = nx.MultiDiGraph()
        for nid, node in self.nodes.items():
            G.add_node(
                nid,
                node_type=node.node_type.value,
                name=node.name,
                file_path=node.file_path,
                start_line=node.start_line,
                end_line=node.end_line,
                token_count=node.token_count
            )
        for edge in self.edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.edge_type.value,
                weight=edge.weight
            )
        return G

    def stats(self) -> Dict[str, Any]:
        """Returns structural statistics of the repository knowledge graph."""
        type_counts = defaultdict(int)
        for n in self.nodes.values():
            type_counts[n.node_type.value] += 1
        edge_counts = defaultdict(int)
        for e in self.edges:
            edge_counts[e.edge_type.value] += 1

        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "node_types": dict(type_counts),
            "edge_types": dict(edge_counts),
            "num_files": len(self.file_index)
        }
