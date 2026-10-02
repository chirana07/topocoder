"""
Personalized CodeRank (PCR) algorithm for bug-localization and relevance propagation on H-CKG.
Implements relational random walks with restart to compute exact structural relevance in O(|E|) time.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from codegraph.graph import CodeKnowledgeGraph
from codegraph.schema import EdgeType


# Default relation propagation weights
DEFAULT_RELATION_WEIGHTS: Dict[EdgeType, float] = {
    EdgeType.CALLS: 1.0,
    EdgeType.TESTS: 1.4,
    EdgeType.INHERITS: 0.8,
    EdgeType.CONTAINS: 0.4,
    EdgeType.IMPORTS: 0.3,
    EdgeType.REFERENCES: 0.5,
}


class PersonalizedCodeRank:
    """Computes Personalized PageRank over the multi-relational code property graph."""

    def __init__(
        self,
        graph: CodeKnowledgeGraph,
        alpha: float = 0.85,
        relation_weights: Optional[Dict[EdgeType, float]] = None,
        reverse_flow_weight: float = 0.4
    ):
        self.graph = graph
        self.alpha = alpha
        self.weights = relation_weights or DEFAULT_RELATION_WEIGHTS
        self.reverse_flow_weight = reverse_flow_weight
        self._build_transition_matrix()

    def _build_transition_matrix(self):
        """Constructs an adjacency transition map with relational and bidirectional weights."""
        self.node_list: List[str] = list(self.graph.nodes.keys())
        self.node_to_idx: Dict[str, int] = {nid: i for i, nid in enumerate(self.node_list)}
        self.n_nodes = len(self.node_list)

        # Adjacency transitions: u -> {v: weight}
        self.transitions: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.out_weight_sums: np.ndarray = np.zeros(self.n_nodes, dtype=np.float64)

        if self.n_nodes == 0:
            return

        for edge in self.graph.edges:
            u_idx = self.node_to_idx.get(edge.source_id)
            v_idx = self.node_to_idx.get(edge.target_id)
            if u_idx is None or v_idx is None:
                continue

            base_w = self.weights.get(edge.edge_type, 1.0) * edge.weight

            # Forward edge (source -> target)
            self.transitions[u_idx][v_idx] += base_w

            # Backward edge (target -> source, e.g. caller seeking callee root cause)
            self.transitions[v_idx][u_idx] += base_w * self.reverse_flow_weight

        # Precompute row sums for normalization
        for u, targets in self.transitions.items():
            self.out_weight_sums[u] = sum(targets.values())

    def compute(
        self,
        seeds: Dict[str, float],
        max_iter: int = 40,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        Runs Power Iteration with restart on the seed distribution.

        Args:
            seeds: Mapping from node_id -> initial seed weight (e.g. failing test = 1.0)
            max_iter: Maximum number of iterations
            tol: L1 convergence threshold

        Returns:
            Mapping from node_id -> stationary Personalized CodeRank score
        """
        if self.n_nodes == 0:
            return {}

        # 1. Initialize seed distribution p0
        p0 = np.zeros(self.n_nodes, dtype=np.float64)
        for nid, w in seeds.items():
            idx = self.node_to_idx.get(nid)
            if idx is not None:
                p0[idx] = w

        sum_p0 = p0.sum()
        if sum_p0 <= 0:
            # Uniform fallback if no seeds matched
            p0 = np.ones(self.n_nodes, dtype=np.float64) / self.n_nodes
        else:
            p0 = p0 / sum_p0

        # Initial distribution
        pi = p0.copy()

        # 2. Power Iteration: pi^(t+1) = (1 - alpha) * p0 + alpha * (pi^(t) * P)
        for _ in range(max_iter):
            next_pi = (1.0 - self.alpha) * p0
            dangling_mass = 0.0

            for u in range(self.n_nodes):
                u_mass = pi[u]
                if u_mass == 0.0:
                    continue
                deg = self.out_weight_sums[u]
                if deg > 0:
                    step = (self.alpha * u_mass) / deg
                    for v, w in self.transitions[u].items():
                        next_pi[v] += step * w
                else:
                    dangling_mass += self.alpha * u_mass

            if dangling_mass > 0:
                next_pi += dangling_mass * p0

            # Convergence check
            err = np.sum(np.abs(next_pi - pi))
            pi = next_pi
            if err < tol:
                break

        return {self.node_list[i]: float(pi[i]) for i in range(self.n_nodes)}

    def rank_nodes(
        self,
        seeds: Dict[str, float],
        top_k: int = 20,
        exclude_seeds: bool = False
    ) -> List[Tuple[str, float]]:
        """Computes PCR and returns top-k ranked nodes."""
        scores = self.compute(seeds)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if exclude_seeds:
            seed_set = set(seeds.keys())
            ranked = [item for item in ranked if item[0] not in seed_set]
        return ranked[:top_k]
