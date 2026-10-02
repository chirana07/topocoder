"""
Unit and integration tests for the TopoCoder codegraph engine.
Tests parsing, graph indexing, Personalized CodeRank, and Context Condensation.
"""

import os
import pytest
from codegraph.schema import EdgeType, NodeType
from codegraph.parser import RepositoryParser
from codegraph.graph import CodeKnowledgeGraph
from codegraph.pagerank import PersonalizedCodeRank
from codegraph.condensation import TopologicalCondenser
from codegraph.tools import CodeGraphToolSuite


SAMPLE_CODE_A = '''"""Data processing module."""

class DataProcessor:
    """Handles parsing and transformation of input payloads."""
    
    def __init__(self, schema_version: int = 1):
        self.schema_version = schema_version
        
    def transform(self, payload: dict) -> dict:
        """Transforms payload data into normalized format."""
        cleaned = self._clean(payload)
        return self._normalize(cleaned)
        
    def _clean(self, raw: dict) -> dict:
        """Removes null bytes and extraneous whitespace."""
        return {k.strip(): v for k, v in raw.items() if v is not None}
        
    def _normalize(self, data: dict) -> dict:
        """Applies schema normalization."""
        # Intentionally bug-prone logic for benchmark
        data["version"] = self.schema_version
        return data
'''

SAMPLE_CODE_B = '''"""Pipeline orchestration and execution."""
from sample_a import DataProcessor

def run_pipeline(raw_input: dict) -> dict:
    """Executes the complete ingestion and normalization pipeline."""
    processor = DataProcessor(schema_version=2)
    return processor.transform(raw_input)

def test_pipeline_transformation():
    """Unit test for pipeline transformation."""
    raw = {" key ": "value", "empty": None}
    res = run_pipeline(raw)
    assert res.get("key") == "value"
    assert res.get("version") == 2
'''


@pytest.fixture
def sample_graph():
    """Builds a test CodeKnowledgeGraph from sample code."""
    parser = RepositoryParser(root_path=".")
    
    # Parse module A
    nodes_a, edges_a = parser.parse_code_string("pkg/sample_a.py", SAMPLE_CODE_A)
    # Parse module B
    nodes_b, edges_b = parser.parse_code_string("tests/test_pipeline.py", SAMPLE_CODE_B)
    
    graph = CodeKnowledgeGraph()
    for n in list(nodes_a.values()) + list(nodes_b.values()):
        graph.add_node(n)
    for e in edges_a + edges_b:
        graph.add_edge(e)
        
    return graph


def test_parser_and_graph_construction(sample_graph):
    """Verifies that nodes and edges are correctly extracted from AST."""
    stats = sample_graph.stats()
    assert stats["num_nodes"] > 5
    assert stats["num_edges"] > 3
    assert stats["node_types"].get(NodeType.MODULE.value, 0) == 2
    assert stats["node_types"].get(NodeType.CLASS.value, 0) == 1
    assert stats["node_types"].get(NodeType.METHOD.value, 0) >= 4


def test_call_graph_resolution(sample_graph):
    """Verifies caller and callee navigation."""
    # Find transform method
    transform_nodes = sample_graph.search_nodes("transform")
    assert len(transform_nodes) > 0
    transform_node = transform_nodes[0][0]
    
    # Check callees: transform calls _clean and _normalize
    callees = sample_graph.get_callees(transform_node.id)
    callee_names = [c.name for c in callees]
    assert "_clean" in callee_names
    assert "_normalize" in callee_names


def test_personalized_coderank_propagation(sample_graph):
    """Verifies that Personalized CodeRank propagates mass from test seed to target callee."""
    # Seed at the test function
    test_nodes = sample_graph.search_nodes("test_pipeline_transformation")
    assert len(test_nodes) > 0
    test_node = test_nodes[0][0]
    
    pcr = PersonalizedCodeRank(sample_graph, alpha=0.85)
    ranked = pcr.rank_nodes(seeds={test_node.id: 1.0}, top_k=5, exclude_seeds=True)
    
    assert len(ranked) > 0
    top_ids = [nid for nid, score in ranked]
    
    # The direct pipeline runner or DataProcessor methods must be in top ranks
    assert any("run_pipeline" in tid or "transform" in tid or "DataProcessor" in tid for tid in top_ids)
    # All scores must be non-negative and sum roughly bounded
    for _, score in ranked:
        assert score >= 0.0


def test_topological_condensation_budget(sample_graph):
    """Verifies that Topological Context Condensation stays within strict token budget."""
    test_nodes = sample_graph.search_nodes("test_pipeline_transformation")
    test_node = test_nodes[0][0]
    
    condenser = TopologicalCondenser(sample_graph, default_budget_tokens=800)
    result = condenser.condense(seeds={test_node.id: 1.0}, budget_tokens=500)
    
    assert result["total_tokens"] <= 500
    assert result["full_node_count"] >= 1
    assert "### [TOPOLOGICAL CODEBASE CONTEXT]" in result["formatted_slice"]
    # Check that stubs or focal nodes are cleanly formatted
    assert "class DataProcessor" in result["formatted_slice"] or "def run_pipeline" in result["formatted_slice"]


def test_tool_suite_integration(sample_graph):
    """Verifies the agent tool suite API."""
    tools = CodeGraphToolSuite(sample_graph)
    
    summary = tools.get_repo_summary()
    assert summary["stats"]["num_nodes"] > 0
    
    symbols = tools.search_symbols("clean")
    assert len(symbols) > 0
    assert symbols[0]["name"] == "_clean"
    
    node_detail = tools.inspect_node(symbols[0]["id"])
    assert node_detail is not None
    assert "def _clean" in node_detail["source_code"]


def test_structural_embeddings_and_retrieval(sample_graph):
    """Verifies that random walk structural embeddings capture neighborhood proximity."""
    from codegraph.embeddings import StructuralGraphEmbedder, HybridRetriever
    
    embedder = StructuralGraphEmbedder(sample_graph, embedding_dim=16, walk_length=6, walks_per_node=10)
    embs = embedder.fit(random_seed=42)
    assert embs.shape[0] == len(sample_graph.nodes)
    assert embs.shape[1] == 16
    
    # Test hybrid retrieval
    retriever = HybridRetriever(sample_graph, embedder)
    results = retriever.query("transform normalize payload", alpha_semantic=0.6, top_k=3)
    assert len(results) > 0
    top_ids = [nid for nid, _ in results]
    assert any("transform" in tid or "normalize" in tid for tid in top_ids)


def test_interactive_visualizer_generation(sample_graph, tmp_path):
    """Verifies that the interactive HTML graph is generated cleanly."""
    from codegraph.visualizer import generate_interactive_html
    
    out_file = str(tmp_path / "test_graph.html")
    res_path = generate_interactive_html(sample_graph, output_path=out_file)
    assert os.path.exists(res_path)
    with open(res_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "vis-network" in content
    assert "TopoCoder Graph Explorer" in content

