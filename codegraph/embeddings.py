"""
Structural and Hybrid Node Embeddings for Code Knowledge Graphs.
Implements Random Walk Representation Learning (inspired by DeepWalk / Node2Vec)
and hybrid semantic-topological retrieval for repository navigation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from codegraph.graph import CodeKnowledgeGraph
from codegraph.schema import EdgeType, NodeType


class StructuralGraphEmbedder:
    """
    Learns low-dimensional structural representations of code entities
    via truncated relational random walks and spectral factorization.
    """

    def __init__(self, graph: CodeKnowledgeGraph, embedding_dim: int = 64, walk_length: int = 10, walks_per_node: int = 15):
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.node_list: List[str] = list(graph.nodes.keys())
        self.node_to_idx: Dict[str, int] = {nid: i for i, nid in enumerate(self.node_list)}
        self.embeddings: Optional[np.ndarray] = None

    def fit(self, random_seed: int = 42) -> np.ndarray:
        """
        Executes truncated relational random walks and computes structural embeddings.
        Uses randomized SVD over co-occurrence transitions.
        """
        rng = np.random.RandomState(random_seed)
        n = len(self.node_list)
        if n == 0:
            return np.zeros((0, self.embedding_dim))

        # Build co-occurrence matrix via simulated random walks
        co_occurrence = np.zeros((n, n), dtype=np.float32)

        for u_idx, u_id in enumerate(self.node_list):
            for _ in range(self.walks_per_node):
                curr = u_id
                for _ in range(self.walk_length):
                    callees = self.graph.get_callees(curr)
                    callers = self.graph.get_callers(curr)
                    neighbors = callees + callers
                    if not neighbors:
                        break
                    # Random step
                    nxt = rng.choice(neighbors)
                    v_idx = self.node_to_idx.get(nxt.id)
                    if v_idx is not None:
                        co_occurrence[u_idx, v_idx] += 1.0
                        co_occurrence[v_idx, u_idx] += 0.5  # Asymmetric reverse mass
                    curr = nxt.id

        # Normalize rows to form transition probability
        row_sums = co_occurrence.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        normalized_trans = co_occurrence / row_sums

        # Truncated SVD for compact embedding factorization
        u, s, vt = np.linalg.svd(normalized_trans, full_matrices=False)
        dim = min(self.embedding_dim, n)
        self.embeddings = u[:, :dim] * np.sqrt(s[:dim])
        
        # If dimension is smaller than requested, pad with zeros
        if self.embeddings.shape[1] < self.embedding_dim:
            pad_width = self.embedding_dim - self.embeddings.shape[1]
            self.embeddings = np.pad(self.embeddings, ((0, 0), (0, pad_width)))

        # Normalize to unit sphere for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

        return self.embeddings

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        idx = self.node_to_idx.get(node_id)
        if idx is not None and self.embeddings is not None:
            return self.embeddings[idx]
        return None

    def find_structurally_similar(self, node_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Finds entities that play structurally similar roles in the codebase."""
        target_emb = self.get_embedding(node_id)
        if target_emb is None:
            return []

        sims = np.dot(self.embeddings, target_emb)
        ranked_indices = np.argsort(-sims)

        results = []
        for idx in ranked_indices:
            nid = self.node_list[idx]
            if nid != node_id:
                results.append((nid, float(sims[idx])))
            if len(results) >= top_k:
                break
        return results


class HybridRetriever:
    """
    Combines dense lexical token similarity with structural graph embeddings
    to enable unified semantic-topological symbol retrieval.
    """

    def __init__(self, graph: CodeKnowledgeGraph, embedder: StructuralGraphEmbedder):
        self.graph = graph
        self.embedder = embedder
        if embedder.embeddings is None:
            embedder.fit()

    def query(
        self,
        query_text: str,
        seed_node_ids: Optional[List[str]] = None,
        alpha_semantic: float = 0.5,
        top_k: int = 8
    ) -> List[Tuple[str, float]]:
        """
        Computes composite similarity:
        Score(v) = alpha * LexicalSemantic(v, query) + (1 - alpha) * StructuralProximity(v, seeds)
        """
        q_tokens = set(query_text.lower().split())
        scored: List[Tuple[str, float]] = []

        # Average seed structural embedding
        seed_emb = None
        if seed_node_ids:
            valid_embs = [self.embedder.get_embedding(s) for s in seed_node_ids if self.embedder.get_embedding(s) is not None]
            if valid_embs:
                seed_emb = np.mean(valid_embs, axis=0)
                seed_emb = seed_emb / (np.linalg.norm(seed_emb) + 1e-8)

        for nid, node in self.graph.nodes.items():
            # 1. Lexical / semantic score
            node_text = f"{node.name} {node.docstring} {node.file_path}".lower()
            overlap = sum(1 for t in q_tokens if t in node_text)
            sem_score = overlap / max(1, len(q_tokens))

            # 2. Structural score
            struct_score = 0.0
            if seed_emb is not None:
                n_emb = self.embedder.get_embedding(nid)
                if n_emb is not None:
                    struct_score = float(np.dot(n_emb, seed_emb))

            composite = alpha_semantic * sem_score + (1.0 - alpha_semantic) * struct_score
            scored.append((nid, composite))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
