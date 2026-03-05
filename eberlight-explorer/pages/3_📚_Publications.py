"""Publications Archive & Review Viewer (F4)."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.content_parser import load_yaml, read_local_file, extract_tldr, extract_section
from components.level_selector import render_level_selector
from components.paper_card import render_paper_card

st.set_page_config(page_title="Publications", page_icon="📚", layout="wide")

level = render_level_selector(key="pub_level")
papers = load_yaml("publication_catalog.yaml")["publications"]

st.title("📚 Publications Archive")
st.markdown("Paper reviews and key findings from synchrotron AI/ML research.")

# Deduplicate by unique file paths
seen_files = set()
unique_papers = []
for p in papers:
    if p["file"] not in seen_files:
        seen_files.add(p["file"])
        unique_papers.append(p)

if level == "L0":
    # Stats overview
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Reviews", len(unique_papers))
    with cols[1]:
        years = sorted(set(p["year"] for p in unique_papers))
        st.metric("Year Range", f"{years[0]}–{years[-1]}")
    with cols[2]:
        high_priority = sum(1 for p in unique_papers if p.get("priority") == "High")
        st.metric("High Priority", high_priority)
    with cols[3]:
        categories = set(p["category"] for p in unique_papers)
        st.metric("Categories", len(categories))

    # Timeline
    st.markdown("---")
    st.subheader("Publication Timeline")
    import plotly.express as px
    import pandas as pd

    year_counts = {}
    for p in unique_papers:
        year_counts[p["year"]] = year_counts.get(p["year"], 0) + 1
    df = pd.DataFrame(
        [{"Year": y, "Papers": c} for y, c in sorted(year_counts.items())]
    )
    fig = px.bar(df, x="Year", y="Papers", color_discrete_sequence=["#00D4AA"])
    fig.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

elif level == "L1":
    # Group by category
    st.subheader("Papers by Topic")
    from collections import defaultdict
    by_cat = defaultdict(list)
    for p in unique_papers:
        by_cat[p["category"]].append(p)

    cat_icons = {
        "denoising": "🧹",
        "reconstruction": "🏗️",
        "image_segmentation": "🎯",
        "autonomous_experiment": "🤖",
    }
    for cat, cat_papers in sorted(by_cat.items()):
        icon = cat_icons.get(cat, "📄")
        with st.expander(f"{icon} {cat.replace('_', ' ').title()} ({len(cat_papers)} papers)", expanded=True):
            for p in sorted(cat_papers, key=lambda x: x["year"], reverse=True):
                render_paper_card(p, show_detail=False)

elif level == "L2":
    # Individual paper detail
    paper_titles = [f"[{p['year']}] {p['title']}" for p in unique_papers]
    selected_idx = st.selectbox(
        "Select Paper",
        options=range(len(unique_papers)),
        format_func=lambda i: paper_titles[i],
    )
    paper = unique_papers[selected_idx]

    st.subheader(paper["title"])
    st.caption(f"{paper.get('authors', 'N/A')} | {paper['journal']} ({paper['year']})")

    content = read_local_file(paper["file"])
    if content:
        # Show structured sections
        tldr = extract_tldr(content)
        if tldr:
            st.info(f"**TL;DR:** {tldr}")

        tab_names = ["Method", "Key Results", "Strengths", "Limitations", "BER Relevance", "Full Review"]
        tabs = st.tabs(tab_names)

        section_map = {
            0: "Method",
            1: "Key Results",
            2: "Strengths",
            3: "Limitations",
            4: "Relevance to APS BER Program",
        }

        for i, tab in enumerate(tabs[:-1]):
            with tab:
                section_content = extract_section(content, section_map[i])
                if section_content:
                    st.markdown(section_content)
                else:
                    st.info("Section not found in this review.")

        with tabs[-1]:
            st.markdown(content, unsafe_allow_html=False)

elif level == "L3":
    # Source view
    paper_titles = [f"[{p['year']}] {p['title']}" for p in unique_papers]
    selected_idx = st.selectbox(
        "Select Paper",
        options=range(len(unique_papers)),
        format_func=lambda i: paper_titles[i],
    )
    paper = unique_papers[selected_idx]

    content = read_local_file(paper["file"])
    if content:
        st.code(content, language="markdown")
