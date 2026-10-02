"""
Tests for PatchVerifier and DualProcessGemmaAgent.
"""

import pytest
from codegraph.parser import RepositoryParser
from codegraph.graph import CodeKnowledgeGraph
from agent.verifier import PatchVerifier
from agent.dual_process import DualProcessGemmaAgent


ORIG_FILE_CODE = '''"""Calculator module."""

def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a - b  # Buggy!

def multiply(a: int, b: int) -> int:
    return a * b
'''

REPLACEMENT_CODE = '''def add(a: int, b: int) -> int:
    """Adds two numbers correctly."""
    return a + b
'''


def test_syntax_verification():
    v = PatchVerifier()
    res_valid = v.verify_syntax("def foo():\n    return 42\n")
    assert res_valid.is_valid
    assert res_valid.error_message is None

    res_invalid = v.verify_syntax("def foo(:\n return 42")
    assert not res_valid.is_valid is False
    assert not res_invalid.is_valid
    assert "SyntaxError" in res_invalid.error_message


def test_function_replacement():
    v = PatchVerifier()
    success, new_source, err = v.apply_function_replacement(
        ORIG_FILE_CODE, "add", REPLACEMENT_CODE
    )
    assert success
    assert err is None
    assert "return a + b" in new_source
    assert "return a * b" in new_source  # Preserved other function


def test_dual_process_agent_solve():
    parser = RepositoryParser(root_path=".")
    nodes, edges = parser.parse_code_string("calc.py", ORIG_FILE_CODE)
    
    test_code = '''def test_add():
    assert add(2, 3) == 5
'''
    t_nodes, t_edges = parser.parse_code_string("test_calc.py", test_code)
    
    graph = CodeKnowledgeGraph()
    for n in list(nodes.values()) + list(t_nodes.values()):
        graph.add_node(n)
    for e in edges + t_edges:
        graph.add_edge(e)

    agent = DualProcessGemmaAgent(graph=graph, token_budget=2000)
    result = agent.solve_task(
        issue_description="test_add fails because addition returns difference instead of sum",
        failing_test_name="test_add"
    )

    assert result.success
    assert result.focal_node_id is not None
    assert "add" in result.focal_node_id
    assert result.total_tokens < 2000
    assert len(result.steps) >= 4
