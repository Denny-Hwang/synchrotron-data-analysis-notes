"""Vis.js Network graph component for the Knowledge Graph page (Phase R11).

Replaces the previous Plotly + NetworkX graph (R2) with a true
draggable, force-directed network. The legacy ``eberlight-explorer/``
shipped a vis.js renderer that the user explicitly missed; this is a
modernised port that:

- Stays driven by ``lib.cross_refs::Graph`` (ADR-002 — the runtime
  graph builder hasn't changed; only the renderer).
- Offers three modes: 🌀 Force-directed (default, draggable),
  📊 Hierarchical (kind columns), ❄️ Freeze (lock positions for
  screenshotting / fine drag).
- Click-to-highlight: clicking a node fades unrelated nodes so the
  user can trace one entity's local neighbourhood.
- Click-to-open: a node carries its ``?note=…`` URL in ``data-href``
  so a double-click navigates the parent window to the underlying
  note (the iframe lives same-origin under Streamlit).

Usage:

    from components.visjs_graph import render_visjs_graph

    render_visjs_graph(nodes, edges, mode="physics", height=640)

``nodes`` is a list of dicts with keys ``id, label, group, size,
color, font_size?, shape?, level?, href?``; ``edges`` is a list of
``{from, to, label?, dashes?}``.
"""

from __future__ import annotations

import json

import streamlit.components.v1 as components

# vis-network from a public CDN, pinned. The file is ~250KB gzipped;
# loaded once per page render (Streamlit reruns reload the iframe
# but the browser caches the script).
_VISJS_CDN = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"


def _darken(hex_color: str, factor: float = 0.7) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"


def _lighten(hex_color: str, factor: float = 0.3) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (
        f"#{min(255, int(r + (255 - r) * factor)):02x}"
        f"{min(255, int(g + (255 - g) * factor)):02x}"
        f"{min(255, int(b + (255 - b) * factor)):02x}"
    )


def render_visjs_graph(
    nodes: list[dict],
    edges: list[dict],
    *,
    mode: str = "physics",
    height: int = 640,
    show_legend_caption: str = "",
) -> None:
    """Render an interactive vis.js network in a Streamlit iframe.

    Args:
        nodes: ``[{id, label, group, size, color, font_size?, shape?,
            level?, title?, href?}]``. ``title`` is the hover tooltip
            (HTML allowed). ``href`` (optional) makes a double-click
            navigate the parent window to that URL.
        edges: ``[{from, to, label?, dashes?}]``.
        mode: ``physics`` | ``hierarchy`` | ``freeze``. The user can
            switch mode at runtime via the toolbar; this is the
            initial mode.
        height: Pixel height of the graph canvas.
        show_legend_caption: Optional small caption rendered below
            the toolbar (e.g. node count summary).
    """
    if mode not in {"physics", "hierarchy", "freeze"}:
        mode = "physics"

    # Pre-compute hover-friendly node payload. Each node carries:
    # - color object with darker border + lighter highlight
    # - shadow + smooth font
    # - href passed through to data-attrs for navigation.
    vis_nodes: list[dict] = []
    for n in nodes:
        color = n.get("color") or "#888888"
        node = {
            "id": n["id"],
            "label": n.get("label", n["id"]),
            "group": n.get("group", "default"),
            "size": n.get("size", 18),
            "color": {
                "background": color,
                "border": _darken(color, 0.7),
                "highlight": {"background": _lighten(color, 0.3), "border": color},
                "hover": {"background": _lighten(color, 0.2), "border": color},
            },
            "font": {
                "size": n.get("font_size", 13),
                "color": "#1A1A1A",
                "face": "'Source Sans 3', system-ui, sans-serif",
            },
            "shape": n.get("shape", "dot"),
            "shadow": {
                "enabled": True,
                "color": "rgba(0,0,0,0.10)",
                "size": 6,
                "x": 2,
                "y": 2,
            },
            "borderWidth": 2,
        }
        if "level" in n:
            node["level"] = n["level"]
        if n.get("title"):
            node["title"] = n["title"]
        if n.get("href"):
            node["href"] = n["href"]  # consumed by JS click handler
        vis_nodes.append(node)

    vis_edges: list[dict] = []
    for e in edges:
        edge = {
            "from": e["from"],
            "to": e["to"],
            "color": {"color": "#C8D2E2", "highlight": "#0033A0", "opacity": 0.7},
            "width": 1.4,
            "smooth": {"type": "continuous", "roundness": 0.2},
            "arrows": {"to": {"enabled": False}},
        }
        if e.get("label"):
            edge["label"] = e["label"]
            edge["font"] = {"size": 10, "color": "#888", "strokeWidth": 0}
        if e.get("dashes"):
            edge["dashes"] = True
        vis_edges.append(edge)

    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)

    initial_mode = mode

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<script src="{_VISJS_CDN}"></script>
<style>
    body {{ margin: 0; padding: 0;
            font-family: 'Source Sans 3', system-ui, -apple-system, sans-serif; }}
    #toolbar {{ display: flex; gap: 8px; align-items: center;
                padding: 8px 12px; flex-wrap: wrap; }}
    .mode-btn {{ padding: 6px 14px; border: 1px solid #C8D2E2;
                 border-radius: 18px; background: #FFFFFF; cursor: pointer;
                 font-size: 13px; color: #1A1A1A; font-weight: 600;
                 transition: background 0.15s, border-color 0.15s, color 0.15s; }}
    .mode-btn:hover {{ background: #F0F4FB; border-color: #0033A0; color: #0033A0; }}
    .mode-btn.active {{ background: #0033A0; color: #FFFFFF; border-color: #0033A0; }}
    .toolbar-tip {{ margin-left: auto; font-size: 12px; color: #666; }}
    #graph {{ width: 100%; height: {height}px;
              border: 1px solid #E0E0E0; border-radius: 8px; background: #FAFBFC; }}
    #legend {{ font-size: 12px; color: #666; padding: 6px 12px; }}
</style>
</head><body>
<div id="toolbar" role="toolbar" aria-label="Graph layout">
    <button class="mode-btn" id="btn-physics" type="button">🌀 Force-directed</button>
    <button class="mode-btn" id="btn-hierarchy" type="button">📊 Hierarchical</button>
    <button class="mode-btn" id="btn-freeze" type="button">❄️ Freeze</button>
    <span class="toolbar-tip">
        Drag a node · Scroll to zoom · Click to highlight neighbours · Double-click to open
    </span>
</div>
<div id="graph" role="img" aria-label="Knowledge graph"></div>
<div id="legend">{show_legend_caption}</div>
<script>
(function() {{
    var nodeList = {nodes_json};
    var edgeList = {edges_json};
    var originalColors = {{}};

    // R14 — vis-network 9.x renders ``node.title`` strings via
    // ``document.createTextNode``, which escapes ``<b>…</b>`` /
    // ``<br>`` markup so users see the raw tags as text. To get
    // formatted tooltips we must hand vis.js an actual
    // ``HTMLElement``. The ``title`` strings come from our own
    // Python code (Knowledge Graph entity metadata), not from user
    // input, so ``innerHTML`` is safe here — the supply chain is
    // Python → JSON → this iframe.
    //
    // We keep the raw HTML on a separate ``titleHtml`` property so
    // ``JSON.parse(JSON.stringify(nodeList))`` (used by
    // ``createNetwork`` to deep-clone fresh DataSet rows) doesn't
    // drop the tooltip — ``JSON.stringify`` on an ``HTMLElement``
    // returns ``{{}}``. The actual ``title`` HTMLElement is built per
    // dataset row inside ``createNetwork`` so each rebuild gets a
    // fresh DOM node owned by that vis instance.
    function htmlToElement(html) {{
        var wrap = document.createElement('div');
        wrap.style.fontFamily = "'Source Sans 3', system-ui, sans-serif";
        wrap.style.fontSize = "12px";
        wrap.style.lineHeight = "1.45";
        wrap.style.maxWidth = "320px";
        wrap.innerHTML = html;
        return wrap;
    }}
    nodeList.forEach(function(n) {{
        originalColors[n.id] = JSON.parse(JSON.stringify(n.color));
        if (typeof n.title === 'string' && n.title.length) {{
            n.titleHtml = n.title;
        }}
    }});

    var container = document.getElementById('graph');
    var network = null;
    var currentMode = '{initial_mode}';

    function buildOptions(mode) {{
        var base = {{
            interaction: {{
                hover: true, tooltipDelay: 180,
                zoomView: true, dragNodes: true, dragView: true,
                multiselect: false, navigationButtons: false, keyboard: true,
            }},
            nodes: {{ scaling: {{ min: 12, max: 40 }} }},
            edges: {{ selectionWidth: 2, hoverWidth: 2 }},
        }};
        if (mode === 'hierarchy') {{
            base.layout = {{ hierarchical: {{
                enabled: true, direction: 'UD', sortMethod: 'directed',
                levelSeparation: 130, nodeSpacing: 160, treeSpacing: 200,
                blockShifting: true, edgeMinimization: true,
                parentCentralization: true,
            }} }};
            base.physics = {{ enabled: false }};
        }} else if (mode === 'physics') {{
            base.layout = {{ hierarchical: false, improvedLayout: false, randomSeed: 42 }};
            base.physics = {{
                enabled: true, solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -80, centralGravity: 0.012,
                    springLength: 180, springConstant: 0.04,
                    damping: 0.45, avoidOverlap: 0.7,
                }},
                stabilization: {{ iterations: 250, updateInterval: 25 }},
            }};
        }} else {{ /* freeze */
            base.layout = {{ hierarchical: false }};
            base.physics = {{ enabled: false }};
        }}
        return base;
    }}

    function setActive(mode) {{
        ['physics', 'hierarchy', 'freeze'].forEach(function(m) {{
            var b = document.getElementById('btn-' + m);
            if (b) b.classList.toggle('active', m === mode);
        }});
    }}

    function createNetwork(mode) {{
        if (network) {{ network.destroy(); network = null; }}
        var fresh = JSON.parse(JSON.stringify(nodeList));
        fresh.forEach(function(n) {{
            delete n.x;
            delete n.y;
            // Build a fresh HTMLElement tooltip per network rebuild;
            // a previous instance may have detached / consumed it.
            if (n.titleHtml) {{
                n.title = htmlToElement(n.titleHtml);
            }} else {{
                delete n.title;
            }}
        }});
        var nodesDS = new vis.DataSet(fresh);
        var edgesDS = new vis.DataSet(JSON.parse(JSON.stringify(edgeList)));
        network = new vis.Network(
            container, {{ nodes: nodesDS, edges: edgesDS }}, buildOptions(mode)
        );
        // Click → highlight selected node + neighbours.
        network.on("click", function(params) {{
            if (!params.nodes.length) {{
                nodesDS.get().forEach(function(n) {{
                    nodesDS.update({{ id: n.id, color: originalColors[n.id],
                                      font: {{ color: '#1A1A1A' }} }});
                }});
                return;
            }}
            var sel = params.nodes[0];
            var conn = network.getConnectedNodes(sel);
            nodesDS.get().forEach(function(n) {{
                if (n.id === sel || conn.indexOf(n.id) !== -1) {{
                    nodesDS.update({{ id: n.id, color: originalColors[n.id],
                                      font: {{ color: '#1A1A1A' }} }});
                }} else {{
                    nodesDS.update({{ id: n.id,
                                      color: {{ background: '#E5E5E5', border: '#CCC' }},
                                      font: {{ color: '#BBB' }} }});
                }}
            }});
        }});
        // Double-click → navigate parent to the node's href, if set.
        network.on("doubleClick", function(params) {{
            if (!params.nodes.length) return;
            var nid = params.nodes[0];
            var node = nodeList.filter(function(n) {{ return n.id === nid; }})[0];
            if (node && node.href) {{
                try {{ window.parent.location.href = node.href; }}
                catch (e) {{ window.location.href = node.href; }}
            }}
        }});
    }}

    document.getElementById('btn-physics').addEventListener('click', function() {{
        currentMode = 'physics'; setActive(currentMode); createNetwork(currentMode);
    }});
    document.getElementById('btn-hierarchy').addEventListener('click', function() {{
        currentMode = 'hierarchy'; setActive(currentMode); createNetwork(currentMode);
    }});
    document.getElementById('btn-freeze').addEventListener('click', function() {{
        currentMode = 'freeze'; setActive(currentMode);
        if (network) network.setOptions({{ physics: {{ enabled: false }} }});
    }});

    setActive(currentMode);
    createNetwork(currentMode);
}})();
</script>
</body></html>
"""
    components.html(html, height=height + 90, scrolling=False)
