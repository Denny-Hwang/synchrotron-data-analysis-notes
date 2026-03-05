"""Cross-Reference Navigator (F8)."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.content_parser import load_yaml, read_local_file
from components.level_selector import render_level_selector

st.set_page_config(page_title="Cross-Reference", page_icon="🔗", layout="wide")

level = render_level_selector(key="xref_level")

modalities = load_yaml("modality_metadata.yaml")["modalities"]
categories = load_yaml("method_taxonomy.yaml")["categories"]
papers = load_yaml("publication_catalog.yaml")["publications"]
tools = load_yaml("tool_catalog.yaml")["tools"]
cross_refs = load_yaml("cross_references.yaml")

st.title("🔗 Cross-Reference Navigator")
st.markdown("Explore connections between modalities, methods, tools, and publications.")

if level in ("L0", "L1"):
    # Knowledge graph using streamlit-agraph
    st.subheader("Knowledge Graph")
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
            )
            for n in raw_nodes
        ]
        edges = [
            Edge(source=e["source"], target=e["target"], label=e.get("label", ""))
            for e in raw_edges
        ]

        config = Config(
            width=900,
            height=500,
            directed=True,
            physics=True,
            hierarchical=False,
        )
        agraph(nodes=nodes, edges=edges, config=config)

    except ImportError:
        st.warning(
            "Install `streamlit-agraph` for interactive graph visualization. "
            "Falling back to text view."
        )
        st.markdown("### Connections")
        mod_method = cross_refs.get("modality_method", {})
        for mod_id, method_ids in mod_method.items():
            mod_name = next((m["short_name"] for m in modalities if m["id"] == mod_id), mod_id)
            methods_str = ", ".join(method_ids)
            st.markdown(f"- **{mod_name}** → {methods_str}")

    # Legend
    st.markdown("---")
    legend_cols = st.columns(4)
    with legend_cols[0]:
        st.markdown("🟢 **Modality**")
    with legend_cols[1]:
        st.markdown("🟡 **Method**")
    with legend_cols[2]:
        st.markdown("🔵 **Paper**")
    with legend_cols[3]:
        st.markdown("🔴 **Tool**")

if level in ("L1", "L2"):
    st.markdown("---")

    # Matrix views
    import pandas as pd

    st.subheader("Modality × Method Matrix")
    mod_names = [m["short_name"] for m in modalities]
    cat_names = [c["name"] for c in categories]
    matrix = []
    for m in modalities:
        row = {}
        for c in categories:
            row[c["name"]] = "✅" if m["id"] in c["modalities"] else "—"
        matrix.append(row)
    st.dataframe(pd.DataFrame(matrix, index=mod_names), use_container_width=True)

    st.markdown("---")
    st.subheader("Tool × Pipeline Stage Matrix")
    display_tools = [t for t in tools if t["id"] != "aps_github_repos"]
    stages = ["acquisition", "processing", "analysis"]
    tool_matrix = []
    for t in display_tools:
        row = {s.title(): "✅" if t.get("pipeline_stage") == s else "—" for s in stages}
        tool_matrix.append(row)
    st.dataframe(
        pd.DataFrame(tool_matrix, index=[t["name"] for t in display_tools]),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Paper × Category Matrix")
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
        pd.DataFrame(paper_matrix, index=[f"[{p['year']}] {p['title'][:40]}" for p in unique_papers]),
        use_container_width=True,
    )

if level in ("L2", "L3"):
    st.markdown("---")

    # Glossary
    st.subheader("📖 Glossary")
    glossary_content = read_local_file("08_references/glossary.md")
    if glossary_content:
        if level == "L3":
            st.code(glossary_content, language="markdown")
        else:
            st.markdown(glossary_content)

    # Bibliography
    st.subheader("📚 Bibliography")
    bib_content = read_local_file("08_references/bibliography.bib")
    if bib_content:
        if level == "L3":
            st.code(bib_content, language="bibtex")
        else:
            # Show as formatted entries
            from utils.content_parser import parse_bibtex
            entries = parse_bibtex(bib_content)
            for entry in entries[:20]:
                title = entry.get("title", "Untitled")
                author = entry.get("author", "Unknown")
                year = entry.get("year", "N/A")
                st.markdown(f"- **{title}** — {author} ({year})")
            if len(entries) > 20:
                st.caption(f"... and {len(entries) - 20} more entries")

    # Useful links
    links_content = read_local_file("08_references/useful_links.md")
    if links_content:
        st.markdown("---")
        st.subheader("🔗 Useful Links")
        if level == "L3":
            st.code(links_content, language="markdown")
        else:
            st.markdown(links_content)
