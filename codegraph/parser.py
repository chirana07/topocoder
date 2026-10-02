"""
AST-based Repository Parser for constructing Hierarchical Code Knowledge Graphs.
Extracts modules, classes, functions, methods, call graphs, imports, and inheritance relations.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from codegraph.schema import CodeEdge, CodeNode, EdgeType, NodeType


class RepositoryParser:
    """Parses a Python repository into structured nodes and multi-relational edges."""

    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)
        self.nodes: Dict[str, CodeNode] = {}
        self.edges: List[CodeEdge] = []
        # Index for symbol resolution: mapping short name -> set of node_ids
        self.symbol_index: Dict[str, Set[str]] = {}
        # Import map: file_path -> {imported_alias: qualified_name}
        self.import_maps: Dict[str, Dict[str, str]] = {}
        # Pending call resolution: list of (func_id, node_type, func_ast)
        self.pending_calls: List[Tuple[str, NodeType, ast.AST]] = []

    def parse_repository(self, file_extensions: Tuple[str, ...] = (".py",)) -> Tuple[Dict[str, CodeNode], List[CodeEdge]]:
        """Scans the repository root and parses all matching code files."""
        for root, dirs, files in os.walk(self.root_path):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", "__pycache__", "build", "dist", "egg-info")]
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, self.root_path)
                    self.parse_file(rel_path, abs_path)

        # After all files and symbols are registered, resolve cross-node references and calls
        self._resolve_deferred_edges()
        return self.nodes, self.edges

    def parse_file(self, rel_path: str, abs_path: str) -> None:
        """Parses a single Python file into AST and extracts local definitions."""
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return

        try:
            tree = ast.parse(content, filename=rel_path)
        except SyntaxError:
            # If syntax error in broken file, record module node only
            return

        lines = content.splitlines()
        mod_id = self._make_module_id(rel_path)

        # 1. Register Module Node
        mod_doc = ast.get_docstring(tree) or ""
        mod_node = CodeNode(
            id=mod_id,
            node_type=NodeType.MODULE,
            name=os.path.basename(rel_path),
            file_path=rel_path,
            start_line=1,
            end_line=len(lines),
            signature=f"module {rel_path}",
            docstring=mod_doc,
            source_code=content,
            metadata={"num_lines": len(lines)}
        )
        self._register_node(mod_node)
        self.import_maps[rel_path] = {}

        # 2. Extract Imports & Definitions
        visitor = _ModuleASTVisitor(self, rel_path, lines, mod_id)
        visitor.visit(tree)

    def parse_code_string(self, rel_path: str, code: str) -> Tuple[Dict[str, CodeNode], List[CodeEdge]]:
        """Parses an in-memory code string (useful for unit tests and synthetic benchmarks)."""
        lines = code.splitlines()
        mod_id = self._make_module_id(rel_path)
        try:
            tree = ast.parse(code, filename=rel_path)
        except SyntaxError:
            return self.nodes, self.edges

        mod_doc = ast.get_docstring(tree) or ""
        mod_node = CodeNode(
            id=mod_id,
            node_type=NodeType.MODULE,
            name=os.path.basename(rel_path),
            file_path=rel_path,
            start_line=1,
            end_line=len(lines),
            signature=f"module {rel_path}",
            docstring=mod_doc,
            source_code=code,
            metadata={"num_lines": len(lines)}
        )
        self._register_node(mod_node)
        self.import_maps[rel_path] = {}

        visitor = _ModuleASTVisitor(self, rel_path, lines, mod_id)
        visitor.visit(tree)
        self._resolve_deferred_edges()
        return self.nodes, self.edges

    def _make_module_id(self, rel_path: str) -> str:
        clean = rel_path.replace("\\", "/").rstrip(".py")
        return clean.replace("/", ".")

    def _register_node(self, node: CodeNode):
        self.nodes[node.id] = node
        if node.name not in self.symbol_index:
            self.symbol_index[node.name] = set()
        self.symbol_index[node.name].add(node.id)

    def _resolve_call_target(self, call_node: ast.Call) -> Optional[str]:
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            return func.attr
        return None

    def _resolve_deferred_edges(self):
        """Resolves calls and references after all AST symbols are indexed."""
        for func_id, node_type, func_ast in self.pending_calls:
            for sub in ast.walk(func_ast):
                if isinstance(sub, ast.Call):
                    call_target_name = self._resolve_call_target(sub)
                    if call_target_name:
                        potential_targets = self.symbol_index.get(call_target_name, set())
                        for target_id in potential_targets:
                            if target_id != func_id:
                                edge_type = EdgeType.TESTS if node_type == NodeType.TEST else EdgeType.CALLS
                                self.edges.append(
                                    CodeEdge(func_id, target_id, edge_type, line_number=sub.lineno)
                                )
        self.pending_calls.clear()

        # Clean duplicate edges
        seen = set()
        unique_edges = []
        for e in self.edges:
            key = (e.source_id, e.target_id, e.edge_type)
            if key not in seen and e.source_id in self.nodes and e.target_id in self.nodes:
                seen.add(key)
                unique_edges.append(e)
        self.edges = unique_edges



class _ModuleASTVisitor(ast.NodeVisitor):
    def __init__(self, parser: RepositoryParser, rel_path: str, lines: List[str], mod_id: str):
        self.parser = parser
        self.rel_path = rel_path
        self.lines = lines
        self.mod_id = mod_id
        self.scope_stack: List[str] = [mod_id]
        self.class_stack: List[str] = []

    def _get_source_segment(self, start_line: int, end_line: int) -> str:
        return "\n".join(self.lines[start_line - 1 : end_line])

    def _extract_signature(self, node: ast.AST) -> str:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = []
            for a in node.args.args:
                arg_str = a.arg
                if a.annotation:
                    arg_str += f": {ast.unparse(a.annotation)}"
                args.append(arg_str)
            ret_ann = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
            return f"{prefix}{node.name}({', '.join(args)}){ret_ann}:"
        elif isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            base_str = f"({', '.join(bases)})" if bases else ""
            return f"class {node.name}{base_str}:"
        return ""

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            target_name = alias.name
            as_name = alias.asname or target_name
            self.parser.import_maps[self.rel_path][as_name] = target_name
            # If target exists in nodes, add edge
            if target_name in self.parser.nodes:
                self.parser.edges.append(
                    CodeEdge(self.mod_id, target_name, EdgeType.IMPORTS, line_number=node.lineno)
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            symbol = alias.name
            as_name = alias.asname or symbol
            full_qualified = f"{module}.{symbol}" if module else symbol
            self.parser.import_maps[self.rel_path][as_name] = full_qualified
            # Check if imported module or symbol matches known nodes
            for target_id in (full_qualified, module):
                if target_id in self.parser.nodes:
                    self.parser.edges.append(
                        CodeEdge(self.mod_id, target_id, EdgeType.IMPORTS, line_number=node.lineno)
                    )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        class_name = node.name
        class_id = f"{self.mod_id}:{class_name}"
        parent_scope = self.scope_stack[-1]

        start_line = node.lineno
        end_line = getattr(node, "end_lineno", start_line)
        source = self._get_source_segment(start_line, end_line)
        docstring = ast.get_docstring(node) or ""

        class_node = CodeNode(
            id=class_id,
            node_type=NodeType.CLASS,
            name=class_name,
            file_path=self.rel_path,
            start_line=start_line,
            end_line=end_line,
            signature=self._extract_signature(node),
            docstring=docstring,
            source_code=source,
            decorators=[ast.unparse(d) for d in node.decorator_list]
        )
        self.parser._register_node(class_node)

        # Parent contains class
        self.parser.edges.append(
            CodeEdge(parent_scope, class_id, EdgeType.CONTAINS, line_number=start_line)
        )

        # Class inheritance edges
        for base in node.bases:
            base_name = ast.unparse(base)
            # Find candidate base class
            candidates = self.parser.symbol_index.get(base_name, set())
            for c_id in candidates:
                self.parser.edges.append(
                    CodeEdge(class_id, c_id, EdgeType.INHERITS, line_number=start_line)
                )

        self.scope_stack.append(class_id)
        self.class_stack.append(class_name)
        self.generic_visit(node)
        self.class_stack.pop()
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_function(node, is_async=True)

    def _handle_function(self, node, is_async: bool):
        func_name = node.name
        is_in_class = len(self.class_stack) > 0
        parent_scope = self.scope_stack[-1]

        if is_in_class:
            node_type = NodeType.METHOD
            func_id = f"{parent_scope}.{func_name}"
        else:
            is_test = func_name.startswith("test_") or "test" in self.rel_path.lower()
            node_type = NodeType.TEST if is_test else NodeType.FUNCTION
            func_id = f"{self.mod_id}:{func_name}"

        start_line = node.lineno
        end_line = getattr(node, "end_lineno", start_line)
        source = self._get_source_segment(start_line, end_line)
        docstring = ast.get_docstring(node) or ""

        func_node = CodeNode(
            id=func_id,
            node_type=node_type,
            name=func_name,
            file_path=self.rel_path,
            start_line=start_line,
            end_line=end_line,
            signature=self._extract_signature(node),
            docstring=docstring,
            source_code=source,
            decorators=[ast.unparse(d) for d in node.decorator_list]
        )
        self.parser._register_node(func_node)

        # Containment edge
        self.parser.edges.append(
            CodeEdge(parent_scope, func_id, EdgeType.CONTAINS, line_number=start_line)
        )

        # Record pending call inspection for Pass 2 (after all symbols are registered)
        self.parser.pending_calls.append((func_id, node_type, node))

        self.scope_stack.append(func_id)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _resolve_call_target(self, call_node: ast.Call) -> Optional[str]:
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            return func.attr
        return None

