"""Knowledge Graph — visual cross-reference of every entity in the repo.

Renders the typed graph built by :mod:`lib.cross_refs` as a real
**draggable** vis.js network plus three companion matrices. Every
node carries a deep-link to its underlying note (or to the
Experiment page for recipes), so the graph doubles as a navigation
surface — drag a node, click to highlight neighbours, double-click
to open.

R11 — replaced the Plotly + NetworkX layout (R2/R8) with vis.js so
users can actually rearrange nodes the way the legacy app allowed.

Features:

- **Force-directed network** with three layout modes (force-directed,
  hierarchical kind columns, freeze) — toolbar inside the iframe.
- **Kind filter**: hide / show each of the six entity kinds.
- **Cross-reference matrices**: modality × noise count, recipe →
  noise mapping with deep links, and tool ↔ paper mention table.
- **Entity navigator**: searchable selectbox that jumps to any
  entity's note via the same ``?note=…`` deep link the cluster
  pages use.

Ref: ADR-002 — Notes are the single source of truth (no YAML index).
Ref: ADR-008 — Section 10 Interactive Lab integrated as a graph layer.
Ref: FR-001, FR-007 — Cluster cards + cross-reference filtering.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from components.footer import render_footer
from components.header import render_header
from components.visjs_graph import render_visjs_graph
from lib.cross_refs import (
    Entity,
    Graph,
    build_graph,
    entity_url,
    iter_kinds,
    kind_color,
    kind_size,
)
from lib.detail_level import LEVEL_LABELS, LEVELS, normalise_level
from lib.routing import query_param

st.set_page_config(page_title="Knowledge Graph — eBERlight", page_icon="🧠", layout="wide")

_REPO_ROOT = _EXPLORER_DIR.parent
_CSS_PATH = _EXPLORER_DIR / "assets" / "styles.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource
def _cached_graph() -> Graph:
    return build_graph(_REPO_ROOT)


graph = _cached_graph()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

render_header(active_cluster=None)
st.markdown(
    '<h1 style="color:#0033A0;">🧠 Knowledge Graph</h1>'
    '<p style="color:#555;font-size:16px;margin-bottom:16px;">'
    "Cross-reference network of every modality, AI/ML method, paper, tool, "
    "Interactive-Lab recipe, and noise/artifact in this repository. "
    "Hover any node for details; use the kind filter to focus.</p>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Detail level (L0 stats · L1 stats+matrices · L2 full · L3 raw entities)
# ---------------------------------------------------------------------------

_KG_LEVEL = normalise_level(query_param("level"), default="L2")


def _render_kg_level_pills(current: str) -> None:
    pills = []
    for lvl in LEVELS:
        is_active = lvl == current
        bg = "#0033A0" if is_active else "#E8EEF6"
        fg = "#FFFFFF" if is_active else "#0033A0"
        href = f"?level={lvl}"
        pills.append(
            f'<a href="{href}" target="_self" '
            f'style="text-decoration:none;background:{bg};color:{fg};'
            f"padding:6px 14px;border-radius:14px;font-size:13px;"
            f"font-weight:600;margin-right:6px;display:inline-block;"
            f'margin-bottom:6px;">{LEVEL_LABELS[lvl]}</a>'
        )
    descriptions = {
        "L0": "Counts only — fastest scan of the graph's scale.",
        "L1": "Counts + cross-reference matrices, no spatial graph.",
        "L2": "Full graph + counts + matrices — the default working view.",
        "L3": "Raw entity / edge tables for export and audit.",
    }
    st.markdown(
        '<div style="margin:0 0 16px 0;">'
        '<div style="font-size:11px;color:#888;text-transform:uppercase;'
        'letter-spacing:0.5px;margin-bottom:6px;">Detail level</div>'
        + "".join(pills)
        + f'<div style="font-size:12px;color:#666;margin-top:6px;">'
        f"{descriptions[current]}</div></div>",
        unsafe_allow_html=True,
    )


_render_kg_level_pills(_KG_LEVEL)

# ---------------------------------------------------------------------------
# Top-row stats
# ---------------------------------------------------------------------------

stat_cols = st.columns(6)
for col, kind in zip(stat_cols, iter_kinds(), strict=True):
    items = graph.by_kind(kind)
    col.markdown(
        f'<div class="eberlight-card" style="text-align:center;border-top:3px solid '
        f'{kind_color(kind)};padding:12px 8px;">'
        f'<div style="font-size:11px;color:#888;text-transform:uppercase;'
        f'letter-spacing:0.5px;">{kind}</div>'
        f'<div style="font-size:28px;font-weight:600;color:{kind_color(kind)};">'
        f"{len(items)}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# L0 short-circuits after the stat row.
if _KG_LEVEL == "L0":
    st.caption(
        "L0 Overview — entity counts only. Switch to L1 for matrices, "
        "L2 for the spatial graph, or L3 for raw export."
    )
    render_footer()
    st.stop()

# ---------------------------------------------------------------------------
# Kind filter — multiselect
# ---------------------------------------------------------------------------

# R10 P1-6: first impression of the graph used to be ALL six layers on,
# which renders 100+ nodes packed tight — overwhelming for a first
# visitor. Now we ship with the 4-layer "core" view (modality + method +
# recipe + noise: the entities that drive the Interactive Lab + the
# scientific workflow story) on by default, and let the user opt-in to
# the citation-heavy ``paper`` and ``tool`` layers.
_DEFAULT_VISIBLE_KINDS = {"modality", "method", "recipe", "noise"}

with st.container():
    st.markdown("#### Layers visible")
    visible_kinds: set[str] = set()
    kind_cols = st.columns(len(list(iter_kinds())))
    for col, kind in zip(kind_cols, iter_kinds(), strict=True):
        default_on = kind in _DEFAULT_VISIBLE_KINDS
        if col.checkbox(
            f"{kind.title()} ({len(graph.by_kind(kind))})",
            value=default_on,
            key=f"kg_kind_{kind}",
        ):
            visible_kinds.add(kind)
    st.caption(
        "Tip: papers + tools are off by default to keep the first view "
        "uncluttered. Enable them above to widen the lens."
    )


# ---------------------------------------------------------------------------
# Build the vis.js graph payload (R11 I2 — replaces Plotly+NetworkX so
# users can actually drag nodes around).
# ---------------------------------------------------------------------------


def _kind_size_visjs(kind: str) -> int:
    # Slightly bigger than the Plotly equivalents — vis.js dot sizes
    # read smaller because the labels render outside the node.
    base = kind_size(kind)
    return max(14, int(base * 1.4))


def _build_visjs_payload(
    graph: Graph, visible: set[str]
) -> tuple[list[dict], list[dict], int, int]:
    visible_ids: set[str] = {e.id for e in graph.entities if e.kind in visible}
    nodes: list[dict] = []
    edges: list[dict] = []
    for e in graph.entities:
        if e.id not in visible_ids:
            continue
        tooltip_lines = [f"<b>{e.label}</b>", f"<i>{e.kind}</i>"]
        if e.category:
            tooltip_lines.append(f"category: {e.category}")
        if e.doc_path:
            tooltip_lines.append(f"📄 {e.doc_path}")
        nodes.append(
            {
                "id": e.id,
                "label": e.label,
                "group": e.kind,
                "color": kind_color(e.kind),
                "size": _kind_size_visjs(e.kind),
                "title": "<br>".join(tooltip_lines),
                "href": entity_url(e),
                "shape": "diamond" if e.kind == "modality" else "dot",
            }
        )
    for ed in graph.edges:
        if ed.source_id in visible_ids and ed.target_id in visible_ids:
            edges.append({"from": ed.source_id, "to": ed.target_id})
    return nodes, edges, len(visible_ids), len(edges)


if _KG_LEVEL == "L2":
    payload_nodes, payload_edges, n_visible, n_edges = _build_visjs_payload(graph, visible_kinds)
    if not payload_nodes:
        st.info("No entities visible — enable at least one layer above.")
    else:
        render_visjs_graph(
            payload_nodes,
            payload_edges,
            mode="physics",
            height=640,
            show_legend_caption=(
                f"{n_visible} of {len(graph.entities)} entities · "
                f"{n_edges} of {len(graph.edges)} edges visible · "
                "drag to rearrange · double-click to open the underlying note."
            ),
        )
else:
    st.caption(
        f"L1 — showing matrices for {len(graph.entities)} entities and "
        f"{len(graph.edges)} edges; switch to L2 for the spatial graph."
    )

if _KG_LEVEL == "L3":
    # L3 raw — entity + edge tables for export / audit, no graph at all.
    st.markdown("### L3 — raw entity / edge tables")
    ent_df = pd.DataFrame(
        [
            {
                "id": e.id,
                "kind": e.kind,
                "label": e.label,
                "category": e.category,
                "doc_path": e.doc_path or "",
            }
            for e in graph.entities
        ]
    )
    edge_df = pd.DataFrame(
        [{"source": ed.source_id, "target": ed.target_id, "kind": ed.kind} for ed in graph.edges]
    )
    with st.expander(f"Entities — {len(ent_df)} rows", expanded=True):
        st.dataframe(ent_df, width="stretch", hide_index=True, height=420)
    with st.expander(f"Edges — {len(edge_df)} rows", expanded=False):
        st.dataframe(edge_df, width="stretch", hide_index=True, height=420)
    render_footer()
    st.stop()

# ---------------------------------------------------------------------------
# Entity navigator (searchable jump-to) — L2 only.
# ---------------------------------------------------------------------------

if _KG_LEVEL == "L2":
    st.markdown("### Jump to an entity")
    nav_options: list[tuple[str, Entity]] = [
        (f"[{e.kind}] {e.label}", e)
        for e in sorted(graph.entities, key=lambda x: (x.kind, x.label))
    ]
    nav_idx = st.selectbox(
        "Select",
        options=list(range(len(nav_options))),
        format_func=lambda i: nav_options[i][0],
        label_visibility="collapsed",
        key="kg_nav",
    )
    selected_label, selected_entity = nav_options[nav_idx]
    target = entity_url(selected_entity)
    nbrs = graph.neighbours(selected_entity.id)
    st.markdown(
        f"**{selected_entity.label}** "
        f'<span style="background:{kind_color(selected_entity.kind)};color:white;'
        f'padding:2px 8px;border-radius:10px;font-size:11px;margin-left:6px;">'
        f"{selected_entity.kind}</span>"
        f' &nbsp;·&nbsp; <a href="{target}" target="_self">Open →</a>'
        f" &nbsp;·&nbsp; {len(nbrs)} connections",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Cross-reference matrices
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### Cross-reference matrices")


def _modality_noise_count_matrix(graph: Graph) -> pd.DataFrame:
    counts: dict[str, int] = defaultdict(int)
    for ed in graph.edges:
        if ed.source_id.startswith("modality:") and ed.target_id.startswith("noise:"):
            counts[ed.source_id] += 1
    rows = [
        {"Modality": m.label, "Noise types catalogued": counts.get(m.id, 0)}
        for m in graph.by_kind("modality")
    ]
    return pd.DataFrame(rows).sort_values("Noise types catalogued", ascending=False)


def _recipe_to_noise_table(graph: Graph) -> pd.DataFrame:
    rows = []
    by_id = {e.id: e for e in graph.entities}
    for ed in graph.edges:
        if ed.source_id.startswith("recipe:") and ed.target_id.startswith("noise:"):
            recipe = by_id[ed.source_id]
            noise = by_id[ed.target_id]
            rows.append(
                {
                    "Recipe": recipe.label,
                    "Modality": recipe.category or "—",
                    "Targets noise": noise.label,
                    "Noise file": noise.doc_path or "—",
                }
            )
    return pd.DataFrame(rows)


def _tool_paper_mention_matrix(graph: Graph) -> pd.DataFrame:
    by_id = {e.id: e for e in graph.entities}
    pairs = defaultdict(int)
    for ed in graph.edges:
        if ed.kind == "mentions":
            paper = by_id.get(ed.source_id)
            tool = by_id.get(ed.target_id)
            if paper and tool:
                pairs[(tool.label, paper.label)] += 1
    if not pairs:
        return pd.DataFrame()
    return pd.DataFrame(
        [{"Tool": t, "Paper": p, "Mentions": n} for (t, p), n in sorted(pairs.items())]
    )


with st.expander("Modality × noise-type count", expanded=True):
    st.dataframe(
        _modality_noise_count_matrix(graph),
        width="stretch",
        hide_index=True,
    )

with st.expander("Interactive-Lab recipes → noise types", expanded=True):
    rt = _recipe_to_noise_table(graph)
    if rt.empty:
        st.info("No recipe → noise edges found.")
    else:
        st.dataframe(rt, width="stretch", hide_index=True)
        st.caption(
            "Each row links a section-10 recipe (`experiments/`) to the "
            "noise-catalog entry it mitigates, via the `noise_catalog_ref` "
            "field of `recipe.yaml`."
        )

with st.expander("Tools mentioned in paper reviews (best-effort regex match)"):
    tp = _tool_paper_mention_matrix(graph)
    if tp.empty:
        st.info("No paper-mention edges found yet — content scanning matched no known tool names.")
    else:
        st.dataframe(tp, width="stretch", hide_index=True)
        st.caption(
            "Heuristic: paper-review markdown is scanned for canonical tool "
            "names (TomoPy, TomoGAN, PyXRF, etc.). Useful for navigation; "
            "not a citation database."
        )

render_footer()
