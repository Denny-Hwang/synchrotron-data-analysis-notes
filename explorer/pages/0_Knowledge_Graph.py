"""Knowledge Graph — visual cross-reference of every entity in the repo.

Renders the typed graph built by :mod:`lib.cross_refs` as an
interactive Plotly network plus three companion matrices. Every
node carries a deep-link to its underlying note (or to the
Experiment page for recipes), so the graph doubles as a navigation
surface.

Features (Phase R2 of the parity restoration plan):

- **Force-directed network** of 100+ entities (modalities ↔ methods
  ↔ papers ↔ tools ↔ recipes ↔ noise types) drawn with a NetworkX
  spring layout and rendered via Plotly.
- **Kind filter**: hide / show each of the six entity kinds. The
  Recipes (§10) layer is highlighted in red so the Interactive Lab
  is visible at a glance.
- **Cross-reference matrices**: modality × noise count, recipe →
  noise mapping with deep links, and tool ↔ paper mention table.
- **Entity navigator**: searchable selectbox that jumps to any
  entity's note via the same ``?note=…`` deep link the cluster
  pages use (introduced in PR #40).

Ref: ADR-002 — Notes are the single source of truth (no YAML index).
Ref: ADR-008 — Section 10 Interactive Lab integrated as a graph layer.
Ref: FR-001, FR-007 — Cluster cards + cross-reference filtering.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from urllib.parse import unquote

from components.footer import render_footer
from components.header import render_header
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


def _query_param(name: str) -> str | None:
    raw = st.query_params.get(name)
    if raw is None:
        return None
    if isinstance(raw, list):
        return unquote(raw[0]) if raw else None
    return unquote(str(raw))


# ---------------------------------------------------------------------------
# Detail level (L0 stats · L1 stats+matrices · L2 full · L3 raw entities)
# ---------------------------------------------------------------------------

_KG_LEVEL = normalise_level(_query_param("level"), default="L2")


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

with st.container():
    st.markdown("#### Layers visible")
    visible_kinds: set[str] = set()
    kind_cols = st.columns(len(list(iter_kinds())))
    for col, kind in zip(kind_cols, iter_kinds(), strict=True):
        # Recipes default-on AND noticeably highlighted — the user explicitly
        # asked for section 10 to be first-class in the graph.
        default_on = True
        if col.checkbox(
            f"{kind.title()} ({len(graph.by_kind(kind))})",
            value=default_on,
            key=f"kg_kind_{kind}",
        ):
            visible_kinds.add(kind)


# ---------------------------------------------------------------------------
# Build the Plotly figure
# ---------------------------------------------------------------------------


def _build_networkx(graph: Graph, visible_kinds: set[str]) -> nx.Graph:
    g = nx.Graph()
    visible_ids = {e.id for e in graph.entities if e.kind in visible_kinds}
    for e in graph.entities:
        if e.id in visible_ids:
            g.add_node(e.id, entity=e)
    for ed in graph.edges:
        if ed.source_id in visible_ids and ed.target_id in visible_ids:
            g.add_edge(ed.source_id, ed.target_id, kind=ed.kind)
    return g


# Layout-mode toggle — restores the legacy vis.js hierarchical option.
# "spring" packs related entities tightly (good for navigation),
# "hierarchical" lays kinds out in fixed columns left-to-right (good
# for understanding category structure).
_LAYOUT_MODES = ("spring", "hierarchical")
_layout_mode_label = {
    "spring": "🕸️ Force-directed (spring)",
    "hierarchical": "🪜 Hierarchical (kind columns)",
}
# kind → x-bucket order for hierarchical multipartite layout.
_KIND_LAYER_ORDER = {
    "modality": 0,
    "noise": 1,
    "method": 2,
    "recipe": 3,
    "paper": 4,
    "tool": 5,
}


@st.cache_data(show_spinner=False)
def _layout(
    visible_kinds_key: tuple[str, ...],
    mode: str = "spring",
) -> dict[str, tuple[float, float]]:
    """Cache layout computation per visible-kind selection + mode."""
    g = _build_networkx(graph, set(visible_kinds_key))
    if g.number_of_nodes() == 0:
        return {}
    if mode == "hierarchical":
        # Multipartite: each entity kind gets its own column; nodes
        # within a kind spread vertically by index. Reproduces the
        # legacy vis.js `hierarchical=True` layout intent.
        for nid in g.nodes():
            ent = g.nodes[nid]["entity"]
            g.nodes[nid]["layer"] = _KIND_LAYER_ORDER.get(ent.kind, 99)
        try:
            return nx.multipartite_layout(g, subset_key="layer", align="vertical")  # type: ignore[return-value]
        except Exception:
            # Fallback if multipartite raises (e.g. all-same-layer);
            # spring is always safe.
            pass
    # Spring layout with deterministic seed so the graph doesn't dance
    # on every rerender. ``k`` is tuned for 100+ nodes so isolated
    # nodes don't fly off the edge.
    return nx.spring_layout(g, k=0.55, iterations=80, seed=42)  # type: ignore[return-value]


layout_mode = _query_param("layout") or "spring"
if layout_mode not in _LAYOUT_MODES:
    layout_mode = "spring"

# Layout-mode pill row — sits next to the kind filter so users can
# flip between force-directed and hierarchical views without a reload
# fight. Only meaningful at L2 (graph) so we render it conditionally.
if _KG_LEVEL == "L2":
    pills = []
    for mode in _LAYOUT_MODES:
        is_active = mode == layout_mode
        bg = "#0033A0" if is_active else "#E8EEF6"
        fg = "#FFFFFF" if is_active else "#0033A0"
        # Preserve current ?level= so toggle stays in L2.
        href = f"?level={_KG_LEVEL}&layout={mode}"
        pills.append(
            f'<a href="{href}" target="_self" '
            f'style="text-decoration:none;background:{bg};color:{fg};'
            f"padding:6px 14px;border-radius:14px;font-size:13px;"
            f"font-weight:600;margin-right:6px;display:inline-block;"
            f'margin-bottom:6px;">{_layout_mode_label[mode]}</a>'
        )
    st.markdown(
        '<div style="margin:8px 0 8px 0;">'
        '<div style="font-size:11px;color:#888;text-transform:uppercase;'
        'letter-spacing:0.5px;margin-bottom:6px;">Layout</div>' + "".join(pills) + "</div>",
        unsafe_allow_html=True,
    )

layout_map = _layout(tuple(sorted(visible_kinds)), layout_mode)
nx_graph = _build_networkx(graph, visible_kinds)


def _build_plotly_figure(g: nx.Graph, pos: dict[str, tuple[float, float]]) -> go.Figure:
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.6, color="#BCC4CC"),
        hoverinfo="none",
        mode="lines",
    )

    # One trace per kind keeps the legend tidy and lets users hide a
    # whole layer via the legend (in addition to the checkbox row).
    traces: list[go.Scatter] = [edge_trace]
    by_kind: dict[str, list[str]] = defaultdict(list)
    for nid in g.nodes():
        ent: Entity = g.nodes[nid]["entity"]
        by_kind[ent.kind].append(nid)

    for kind, ids in by_kind.items():
        xs = [pos[nid][0] for nid in ids]
        ys = [pos[nid][1] for nid in ids]
        labels = [g.nodes[nid]["entity"].label for nid in ids]
        hover = []
        custom = []
        for nid in ids:
            ent = g.nodes[nid]["entity"]
            doc = f"<br>📄 {ent.doc_path}" if ent.doc_path else ""
            hover.append(
                f"<b>{ent.label}</b><br>"
                f'<span style="color:#888;">{ent.kind}</span>'
                f"{('<br>category: ' + ent.category) if ent.category else ''}"
                f"{doc}<extra></extra>"
            )
            custom.append(entity_url(ent))
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=labels if kind == "modality" else [""] * len(labels),
                textposition="bottom center",
                textfont=dict(size=11, color="#1A1A1A"),
                hovertemplate=hover,
                customdata=custom,
                name=kind,
                marker=dict(
                    size=kind_size(kind),
                    color=kind_color(kind),
                    line=dict(width=1.2, color="#FFFFFF"),
                    opacity=0.92 if kind != "recipe" else 1.0,
                ),
            )
        )

    return go.Figure(
        data=traces,
        layout=go.Layout(
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            paper_bgcolor="#FAFBFC",
            plot_bgcolor="#FAFBFC",
        ),
    )


if _KG_LEVEL == "L2":
    if nx_graph.number_of_nodes() == 0:
        st.info("No entities visible — enable at least one layer above.")
    else:
        fig = _build_plotly_figure(nx_graph, layout_map)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    st.caption(
        f"Showing {nx_graph.number_of_nodes()} of {len(graph.entities)} entities, "
        f"{nx_graph.number_of_edges()} of {len(graph.edges)} edges. Hover any node for details. "
        "Modality labels are pinned to the plot; other kinds reveal on hover to keep the graph readable."
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
