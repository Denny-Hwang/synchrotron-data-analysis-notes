"""Knowledge Graph & Cross-Reference Navigator (F8)."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.content_parser import load_yaml, read_local_file
from components.level_selector import render_level_selector
from components.common_styles import inject_styles

st.set_page_config(page_title="Knowledge Graph", page_icon="🧠", layout="wide")

inject_styles()

level = render_level_selector(key="xref_level")

modalities = load_yaml("modality_metadata.yaml")["modalities"]
categories = load_yaml("method_taxonomy.yaml")["categories"]
papers = load_yaml("publication_catalog.yaml")["publications"]
tools = load_yaml("tool_catalog.yaml")["tools"]
cross_refs = load_yaml("cross_references.yaml")

st.title("🧠 Knowledge Graph")
st.markdown(
    "Visual map of **connections** between modalities, AI/ML methods, tools, and publications. "
    "Explore how each piece of the synchrotron research ecosystem relates to the others."
)

# ── Knowledge Graph (always shown) ─────────────────────
st.markdown("---")
st.subheader("🔗 Interactive Research Network")

try:
    from streamlit_agraph import agraph, Node, Edge, Config
    from utils.graph_builder import build_knowledge_graph

    raw_nodes, raw_edges = build_knowledge_graph()

    nodes = [
        Node(
            id=n["id"],
            label=n["label"],
            size=n["size"],
            color=n["color"],
            font={"size": 14, "color": "#1A1A2E"},
        )
        for n in raw_nodes
    ]
    edges = [
        Edge(
            source=e["source"],
            target=e["target"],
            label=e.get("label", ""),
            color="#CCCCCC",
        )
        for e in raw_edges
    ]

    config = Config(
        width=1100,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#00D4AA",
    )
    agraph(nodes=nodes, edges=edges, config=config)

except ImportError:
    st.info(
        "💡 Install `streamlit-agraph` for the interactive graph. "
        "Showing text-based view instead."
    )

    # Text-based fallback with better formatting
    st.markdown("#### Modality → AI/ML Method Connections")
    mod_method = cross_refs.get("modality_method", {})
    for mod_id, method_ids in mod_method.items():
        mod_name = next((m["short_name"] for m in modalities if m["id"] == mod_id), mod_id)
        mod_icon = next((m["icon"] for m in modalities if m["id"] == mod_id), "")
        if method_ids:
            method_names = []
            for mid in method_ids:
                cat = next((c for c in categories if c["id"] == mid), None)
                method_names.append(f"{cat['icon']} {cat['name']}" if cat else mid)
            st.markdown(f"- {mod_icon} **{mod_name}** → {' · '.join(method_names)}")
        else:
            st.markdown(f"- {mod_icon} **{mod_name}** → _(no AI/ML methods yet)_")

    st.markdown("#### Tool → Pipeline Stage")
    display_tools = [t for t in tools if t["id"] != "aps_github_repos"]
    for t in display_tools:
        stage = (t.get("pipeline_stage") or "N/A").title()
        st.markdown(f"- {t['icon']} **{t['name']}** → {stage}")

# Legend
st.markdown("---")
legend_cols = st.columns(4)
with legend_cols[0]:
    st.markdown(
        "<span style='display:inline-block;width:14px;height:14px;"
        "background:#00D4AA;border-radius:50%;margin-right:6px;'></span>"
        " **Modality** (6)", unsafe_allow_html=True)
with legend_cols[1]:
    st.markdown(
        "<span style='display:inline-block;width:14px;height:14px;"
        "background:#FFB800;border-radius:50%;margin-right:6px;'></span>"
        " **AI/ML Method** (5)", unsafe_allow_html=True)
with legend_cols[2]:
    st.markdown(
        "<span style='display:inline-block;width:14px;height:14px;"
        "background:#1B3A5C;border-radius:50%;margin-right:6px;'></span>"
        " **Paper** (13)", unsafe_allow_html=True)
with legend_cols[3]:
    st.markdown(
        "<span style='display:inline-block;width:14px;height:14px;"
        "background:#E8515D;border-radius:50%;margin-right:6px;'></span>"
        " **Tool** (6)", unsafe_allow_html=True)

# ── Matrix Views ──────────────────────────────────────
if level in ("L0", "L1", "L2"):
    import pandas as pd

    st.markdown("---")
    st.subheader("📊 Cross-Reference Matrices")

    matrix_tabs = st.tabs([
        "Modality × Method",
        "Tool × Pipeline Stage",
        "Paper × Category",
    ])

    with matrix_tabs[0]:
        st.markdown("Which AI/ML method categories apply to which X-ray modalities?")
        mod_names = [f"{m['icon']} {m['short_name']}" for m in modalities]
        cat_names = [c["name"] for c in categories]
        matrix = []
        for m in modalities:
            row = {}
            for c in categories:
                row[c["name"]] = "✅" if m["id"] in c["modalities"] else "—"
            matrix.append(row)
        st.dataframe(pd.DataFrame(matrix, index=mod_names), use_container_width=True)

    with matrix_tabs[1]:
        st.markdown("Which tools operate at which pipeline stage?")
        display_tools = [t for t in tools if t["id"] != "aps_github_repos"]
        stages = ["acquisition", "processing", "analysis"]
        tool_matrix = []
        for t in display_tools:
            row = {s.title(): "✅" if t.get("pipeline_stage") == s else "—" for s in stages}
            tool_matrix.append(row)
        st.dataframe(
            pd.DataFrame(tool_matrix, index=[f"{t['icon']} {t['name']}" for t in display_tools]),
            use_container_width=True,
        )

    with matrix_tabs[2]:
        st.markdown("Which papers belong to which AI/ML category?")
        seen_files = set()
        unique_papers = []
        for p in papers:
            if p["file"] not in seen_files:
                seen_files.add(p["file"])
                unique_papers.append(p)
        paper_matrix = []
        for p in unique_papers:
            row = {c["name"]: "✅" if p["category"] == c["id"] else "—" for c in categories}
            paper_matrix.append(row)
        st.dataframe(
            pd.DataFrame(
                paper_matrix,
                index=[f"[{p['year']}] {p['title'][:40]}" for p in unique_papers],
            ),
            use_container_width=True,
        )

# ── References & Glossary ─────────────────────────────
if level in ("L2", "L3"):
    st.markdown("---")

    ref_tabs = st.tabs(["📖 Glossary", "📚 Bibliography", "🔗 Useful Links"])

    with ref_tabs[0]:
        glossary_content = read_local_file("08_references/glossary.md")
        if glossary_content:
            if level == "L3":
                st.code(glossary_content, language="markdown")
            else:
                st.markdown(glossary_content)

    with ref_tabs[1]:
        bib_content = read_local_file("08_references/bibliography.bib")
        if bib_content:
            if level == "L3":
                st.code(bib_content, language="bibtex")
            else:
                from utils.content_parser import parse_bibtex
                entries = parse_bibtex(bib_content)
                for entry in entries:
                    title = entry.get("title", "Untitled")
                    author = entry.get("author", "Unknown")
                    year = entry.get("year", "N/A")
                    journal = entry.get("journal", "")
                    doi = entry.get("doi", "")
                    journal_str = f" — _{journal}_" if journal else ""
                    doi_str = f" | [DOI](https://doi.org/{doi})" if doi else ""
                    st.markdown(
                        f"- **{title}**{journal_str}\n"
                        f"  - {author} ({year}){doi_str}"
                    )

    with ref_tabs[2]:
        links_content = read_local_file("08_references/useful_links.md")
        if links_content:
            if level == "L3":
                st.code(links_content, language="markdown")
            else:
                st.markdown(links_content)
