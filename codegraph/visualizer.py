"""
Interactive Visualizer for Hierarchical Code Knowledge Graphs (H-CKG).
Generates a standalone, interactive HTML/JS dashboard using Vis.js to visualize
call graphs, Personalized CodeRank distributions, and topological context condensation.
"""

import json
import os
from typing import Dict, Optional
from codegraph.graph import CodeKnowledgeGraph
from codegraph.pagerank import PersonalizedCodeRank
from codegraph.schema import NodeType


def generate_interactive_html(
    graph: CodeKnowledgeGraph,
    pcr_scores: Optional[Dict[str, float]] = None,
    output_path: str = "paper/figures/interactive_graph.html"
) -> str:
    """
    Renders an interactive web-based graph exploration dashboard.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if pcr_scores is None:
        pcr = PersonalizedCodeRank(graph)
        first_node = list(graph.nodes.keys())[0] if graph.nodes else None
        pcr_scores = pcr.compute({first_node: 1.0}) if first_node else {}

    # Color map by node type
    colors = {
        NodeType.MODULE.value: "#2b5c8f",
        NodeType.CLASS.value: "#8e44ad",
        NodeType.FUNCTION.value: "#2980b9",
        NodeType.METHOD.value: "#27ae60",
        NodeType.TEST.value: "#e67e22"
    }

    nodes_data = []
    max_score = max(pcr_scores.values()) if pcr_scores else 1.0

    for nid, node in graph.nodes.items():
        score = pcr_scores.get(nid, 0.0)
        norm_size = 15 + (score / (max_score + 1e-8)) * 35

        nodes_data.append({
            "id": nid,
            "label": node.name,
            "title": f"<b>{node.id}</b><br>Type: {node.node_type.value}<br>File: {node.file_path}<br>PCR Score: {score:.4f}",
            "color": colors.get(node.node_type.value, "#95a5a6"),
            "size": norm_size,
            "font": {"size": 12, "color": "#ecf0f1"},
            "code": node.source_code.replace("<", "&lt;").replace(">", "&gt;"),
            "docstring": node.docstring,
            "file": node.file_path,
            "lines": f"{node.start_line}-{node.end_line}"
        })

    edges_data = []
    edge_colors = {
        "calls": "#3498db",
        "tests": "#e67e22",
        "contains": "#7f8c8d",
        "imports": "#9b59b6",
        "inherits": "#1abc9c"
    }

    for idx, e in enumerate(graph.edges):
        edges_data.append({
            "from": e.source_id,
            "to": e.target_id,
            "arrows": "to",
            "label": e.edge_type.value,
            "font": {"size": 9, "align": "middle", "color": "#bdc3c7"},
            "color": {"color": edge_colors.get(e.edge_type.value, "#7f8c8d"), "opacity": 0.75}
        })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>TopoCoder - Interactive Code Knowledge Graph</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 0;
      background: #11141a;
      color: #e1e4e8;
      display: flex;
      height: 100vh;
      overflow: hidden;
    }}
    #network-container {{
      flex: 1;
      height: 100%;
      position: relative;
    }}
    #sidebar {{
      width: 420px;
      background: #181d26;
      border-left: 1px solid #2d3748;
      display: flex;
      flex-direction: column;
      padding: 20px;
      box-sizing: border-box;
      overflow-y: auto;
    }}
    h1 {{
      font-size: 1.15rem;
      margin-top: 0;
      color: #58a6ff;
      border-bottom: 1px solid #30363d;
      padding-bottom: 8px;
    }}
    .stat-badge {{
      display: inline-block;
      padding: 4px 8px;
      background: #21262d;
      border-radius: 4px;
      font-size: 0.8rem;
      margin-right: 6px;
      margin-bottom: 8px;
    }}
    .legend {{
      margin: 10px 0;
      font-size: 0.85rem;
    }}
    .legend-item {{
      display: inline-block;
      margin-right: 12px;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      display: inline-block;
      border-radius: 50%;
      margin-right: 4px;
    }}
    pre {{
      background: #0d1117;
      padding: 12px;
      border-radius: 6px;
      border: 1px solid #30363d;
      font-size: 0.82rem;
      overflow-x: auto;
      color: #c9d1d9;
    }}
  </style>
</head>
<body>
  <div id="network-container"></div>
  <div id="sidebar">
    <h1>TopoCoder Graph Explorer</h1>
    <div class="legend">
      <div class="legend-item"><span class="dot" style="background:#2980b9;"></span>Function</div>
      <div class="legend-item"><span class="dot" style="background:#27ae60;"></span>Method</div>
      <div class="legend-item"><span class="dot" style="background:#8e44ad;"></span>Class</div>
      <div class="legend-item"><span class="dot" style="background:#e67e22;"></span>Test</div>
    </div>
    <div>
      <span class="stat-badge">Nodes: {len(nodes_data)}</span>
      <span class="stat-badge">Edges: {len(edges_data)}</span>
      <span class="stat-badge">PPR: Relational Random Walk</span>
    </div>
    <div id="node-details" style="margin-top: 15px;">
      <p style="color: #8b949e;">Click any node to inspect source code, topological properties, and call signatures.</p>
    </div>
  </div>

  <script type="text/javascript">
    const nodes = new vis.DataSet({json.dumps(nodes_data)});
    const edges = new vis.DataSet({json.dumps(edges_data)});
    const container = document.getElementById('network-container');
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      nodes: {{
        shape: 'dot',
        borderWidth: 2,
        shadow: true
      }},
      edges: {{
        width: 1.5,
        smooth: {{ type: 'continuous' }}
      }},
      physics: {{
        stabilization: true,
        barnesHut: {{
          gravitationalConstant: -3500,
          springLength: 95,
          springConstant: 0.04
        }}
      }}
    }};
    const network = new vis.Network(container, data, options);

    network.on("click", function (params) {{
      if (params.nodes.length > 0) {{
        const nodeId = params.nodes[0];
        const node = nodes.get(nodeId);
        document.getElementById('node-details').innerHTML = `
          <h2 style="font-size: 1rem; color: #58a6ff; word-break: break-all;">${{node.id}}</h2>
          <p><b>File:</b> ${{node.file}} (Lines ${{node.lines}})</p>
          <p><b>Docstring:</b> ${{node.docstring || 'None'}}</p>
          <h3>Source Implementation</h3>
          <pre><code>${{node.code}}</code></pre>
        `;
      }}
    }});
  </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path
