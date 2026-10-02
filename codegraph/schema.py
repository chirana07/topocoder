"""
Schema definitions for the Hierarchical Code Knowledge Graph (H-CKG).
Defines node types, edge types, code entities, and graph representations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any


class NodeType(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    TEST = "test"


class EdgeType(str, Enum):
    CONTAINS = "contains"      # Module contains Class, Class contains Method
    IMPORTS = "imports"        # Module imports Module or Symbol
    INHERITS = "inherits"      # Class inherits from Class
    CALLS = "calls"            # Function/Method calls Function/Method
    REFERENCES = "references"  # Node references Symbol or Attribute
    TESTS = "tests"            # Test function targets Function/Class


@dataclass
class CodeNode:
    """Represents a code entity in the repository knowledge graph."""
    id: str                       # Qualified unique ID (e.g. "repo.utils.math:Vector.norm")
    node_type: NodeType           # Entity category
    name: str                     # Short symbol name (e.g. "norm")
    file_path: str                # Relative path from repository root
    start_line: int               # 1-indexed start line
    end_line: int                 # 1-indexed end line
    signature: str = ""           # e.g. "def norm(self, p: int = 2) -> float:"
    docstring: str = ""           # Extracted docstring if present
    source_code: str = ""         # Full raw source code
    token_count: int = 0          # Approximate token count for budgeting
    decorators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.token_count and self.source_code:
            # Rule of thumb for code tokens: ~1 token per 3.5 characters or word-based
            self.token_count = max(1, len(self.source_code.split()))

    @property
    def stub(self) -> str:
        """Returns a compressed structural signature stub with docstring."""
        dec_str = "".join([f"@{d}\n" for d in self.decorators])
        doc_summary = ""
        if self.docstring:
            first_line = self.docstring.strip().split("\n")[0].strip()
            doc_summary = f'\n    """{first_line}"""'
        return f"{dec_str}{self.signature}{doc_summary}\n    # [Implementation omitted: lines {self.start_line}-{self.end_line}]"

    @property
    def stub_token_count(self) -> int:
        return max(1, len(self.stub.split()))


@dataclass
class CodeEdge:
    """Represents a directed multi-relational link between code entities."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    line_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
