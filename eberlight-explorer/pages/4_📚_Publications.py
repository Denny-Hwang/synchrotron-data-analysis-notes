"""Publications Archive & Review Viewer (F4)."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.content_parser import load_yaml, read_local_file, extract_tldr, extract_section
from components.level_selector import render_level_selector
from components.common_styles import inject_styles
from components.paper_card import render_paper_card
from components.mermaid_diagram import render_mermaid

st.set_page_config(page_title="Publications", page_icon="📚", layout="wide")

inject_styles()

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

# Method diagrams for known papers (raw Mermaid code, no fences)
PAPER_DIAGRAMS = {
    "review_tomogan_2020": {
        "code": """graph LR
    A["High-Dose Projections"] --> B["Dose Reduction"]
    B --> C["FBP Recon via TomoPy"]
    C --> D["Noisy Slices"]
    D --> E["TomoGAN Generator"]
    E --> F["Denoised Slices"]
    F --> G["Segmentation & Analysis"]
    H["PatchGAN Discriminator"] -.-> E
    I["VGG-16 Perceptual Loss"] -.-> E
    J["L1 Pixel Loss"] -.-> E""",
        "height": 300,
    },
    "review_roi_finder_2022": {
        "code": """graph LR
    A["Multi-element XRF Survey"] --> B["Spectral Fitting MAPS"]
    B --> C["Elemental Concentration Maps"]
    C --> D["PCA k=3-5"]
    D --> E["Fuzzy C-Means c=3-8"]
    E --> F["Membership Thresholding"]
    F --> G["ROI Scoring"]
    G --> H["Ranked ROI Boxes"]
    H --> I["Beamline Controller"]""",
        "height": 250,
    },
    "review_xrf_gmm_2013": {
        "code": """graph LR
    A["XRF Raster Scan"] --> B["Spectral Fitting"]
    B --> C["7-Channel Elemental Maps"]
    C --> D["Normalize"]
    D --> E["GMM via EM"]
    E --> F["BIC Sweep K=2-8"]
    F --> G["Posterior Probability Maps"]
    G --> H["Component Identification"]""",
        "height": 250,
    },
    "review_ptychonet_2019": {
        "code": """graph LR
    A["Diffraction Patterns"] --> B["Log-scale & Normalize"]
    B --> C["CNN Encoder-Decoder"]
    C --> D["Amplitude & Phase Patches"]
    D --> E["Overlap-weighted Stitching"]
    E --> F["Optional ePIE Refinement"]
    F --> G["Final Reconstruction"]""",
        "height": 250,
    },
    "review_fullstack_dl_tomo_2023": {
        "code": """graph TB
    A["Raw Projections"] --> B["Preprocessing"]
    B --> C["Reconstruction"]
    C --> D["Denoising"]
    D --> E["Segmentation"]
    E --> F["Quantification"]
    F --> G["Visualization"]""",
        "height": 350,
    },
    "review_ai_nerd_2024": {
        "code": """graph LR
    A["XPCS Measurement"] --> B["Speckle Pattern Analysis"]
    B --> C["AI-NERD Feature Extraction"]
    C --> D["Unsupervised Fingerprinting"]
    D --> E["Dynamics Classification"]
    E --> F["Autonomous Decision"]
    F -->|"next measurement"| A""",
        "height": 250,
    },
    "review_aidriven_xanes_2025": {
        "code": """graph LR
    A["Seed Points 5-10"] --> B["GP Surrogate Fit"]
    B --> C["Knowledge-Injected\\nAcquisition Function"]
    C --> D["Select Next Energy E*"]
    D --> E["Monochromator + Measure"]
    E --> F["Update GP"]
    F -->|"iterate"| B
    G["Edge Prior P_edge"] -.-> C
    H["Gradient |∂μ/∂E|"] -.-> C""",
        "height": 280,
    },
    "review_deep_residual_xrf_2023": {
        "code": """graph LR
    A["Low-res XRF Map"] --> B["Upscale Interpolation"]
    B --> C["Deep Residual Network"]
    C --> D["Residual Learning"]
    D --> E["Super-resolved XRF Map"]
    F["High-res Ground Truth"] -.-> C""",
        "height": 250,
    },
    "review_ai_edge_ptychography_2023": {
        "code": """graph LR
    A["Detector Stream"] --> B["Edge FPGA/GPU"]
    B --> C["Lightweight CNN"]
    C --> D["Real-time Phase"]
    D --> E["Feedback to Scan"]
    F["Full Recon on HPC"] -.-> D""",
        "height": 250,
    },
    "review_aiedge_ptycho_2023": {
        "code": """graph LR
    A["Detector Stream"] --> B["Edge FPGA/GPU"]
    B --> C["Lightweight CNN"]
    C --> D["Real-time Phase"]
    D --> E["Feedback to Scan"]
    F["Full Recon on HPC"] -.-> D""",
        "height": 250,
    },
    "review_realtime_uct_hpc_2020": {
        "code": """graph LR
    A["Detector @ 2-BM"] --> B["Streaming to HPC"]
    B --> C["TomoPy Recon"]
    C --> D["GPU Filtering"]
    D --> E["Real-time 3D Volume"]
    E --> F["Live Visualization"]""",
        "height": 250,
    },
    "review_ai_als_workshop_2024": {
        "code": """graph TB
    A["AI@ALS Workshop 2024"] --> B["Autonomous Experiments"]
    A --> C["Real-time Analysis"]
    A --> D["Data Management"]
    B --> E["Adaptive Scanning"]
    C --> F["Edge Computing"]
    D --> G["FAIR Data Practices"]""",
        "height": 300,
    },
    "review_alphafold_2021": {
        "code": """graph LR
    A["Amino Acid Sequence"] --> B["MSA + Templates"]
    B --> C["Evoformer"]
    C --> D["Structure Module"]
    D --> E["3D Coordinates"]
    E --> F["Confidence pLDDT"]""",
        "height": 250,
    },
    "review_fullstack_tomo_2023": {
        "code": """graph TB
    A["Raw Projections"] --> B["Preprocessing"]
    B --> C["Reconstruction"]
    C --> D["Denoising"]
    D --> E["Segmentation"]
    E --> F["Quantification"]
    F --> G["Visualization"]""",
        "height": 350,
    },
    "review_noise2void_2019": {
        "code": """graph LR
    A["Single Noisy Image"] --> B["Random Blind-Spot Masking"]
    B --> C["U-Net Prediction"]
    C --> D["Self-supervised Loss"]
    D --> E["Denoised Output"]
    F["No Clean Target Needed"] -.-> D""",
        "height": 250,
    },
    "review_diffusion_ct_2024": {
        "code": """graph LR
    A["Sparse-View Sinogram"] --> B["FBP Initial Recon"]
    B --> C["Forward Diffusion\\n(Add Noise)"]
    C --> D["Score Network\\nReverse Diffusion"]
    D --> E["Data Consistency\\nStep"]
    E --> F["Clean Reconstruction"]
    D -->|"iterate T steps"| D""",
        "height": 280,
    },
    "review_httomo_2024": {
        "code": """graph TB
    A["Raw Projections"] --> B["YAML Pipeline Config"]
    B --> C["GPU Preprocessing\\n(Ring, Stripe, Norm)"]
    C --> D["GPU Reconstruction\\n(FBP/Gridrec/CGLS)"]
    D --> E["Post-processing\\n(Segmentation)"]
    E --> F["HDF5/TIFF Output"]""",
        "height": 350,
    },
    "review_multimodal_synchrotron_data_2025": {
        "code": """graph TB
    A["Sample"] --> B["XRF Mapping"]
    A --> C["Ptychography"]
    A --> D["Micro-CT"]
    B --> E["Paired 3D Dataset"]
    C --> E
    D --> E
    E --> F["ML Benchmark\\nTraining"]""",
        "height": 350,
    },
    "review_hallucination_tomo_2021": {
        "code": """graph LR
    A["Sparse-View Data"] --> B["DL Reconstruction"]
    B --> C["Output Image"]
    C --> D["sFRC Analysis"]
    D --> E["Hallucination Map"]
    F["Reference FBP"] -.-> D""",
        "height": 250,
    },
}


def _paper_link(paper: dict) -> str:
    """Generate DOI or link string for a paper."""
    doi = paper.get("doi")
    url = paper.get("url")
    if doi:
        return f"[DOI: {doi}](https://doi.org/{doi})"
    elif url:
        return f"[Link]({url})"
    return ""


def _extract_review_section(content: str, name: str) -> str | None:
    """Extract section with common aliases."""
    alias_map = {
        "Relevance": ["Relevance to APS BER Program", "Relevance to eBERlight",
                       "Relevance to BER", "BER Relevance"],
        "Limitations": ["Limitations & Gaps", "Limitations"],
        "Background": ["Background & Motivation", "Background"],
        "Takeaways": ["Actionable Takeaways", "Takeaways"],
    }
    aliases = alias_map.get(name, [name])
    return extract_section(content, aliases[0], aliases[1:] if len(aliases) > 1 else None)


if level == "L0":
    # Stats overview
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Reviews", len(unique_papers))
    with cols[1]:
        years = sorted(set(p["year"] for p in unique_papers))
        st.metric("Year Range", f"{years[0]}--{years[-1]}")
    with cols[2]:
        high_priority = sum(1 for p in unique_papers if p.get("priority") == "High")
        st.metric("High Priority", high_priority)
    with cols[3]:
        cat_set = set(p["category"] for p in unique_papers)
        st.metric("Categories", len(cat_set))

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
    # Group by category with TL;DR and key results
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
                content = read_local_file(p["file"])
                if content:
                    tldr = extract_tldr(content)
                    if tldr:
                        st.markdown(f"> {tldr[:400]}{'...' if len(tldr) > 400 else ''}")

                    # Show key results summary table
                    key_results = extract_section(content, "Key Results")
                    if key_results:
                        lines = [l for l in key_results.split("\n") if l.strip()]
                        table_lines = [l for l in lines if l.strip().startswith("|")]
                        if table_lines:
                            st.markdown("\n".join(table_lines[:8]))
                st.markdown("---")

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

    # Author, journal, and link
    link_str = _paper_link(paper)
    meta_parts = [
        paper.get("authors", "N/A"),
        f'{paper["journal"]} ({paper["year"]})',
    ]
    if link_str:
        meta_parts.append(link_str)
    st.markdown(" | ".join(meta_parts))

    # Tags
    tag_str = " ".join(f"`{t}`" for t in paper.get("tags", []))
    if tag_str:
        st.markdown(tag_str)

    content = read_local_file(paper["file"])
    if content:
        # TL;DR
        tldr = extract_tldr(content)
        if tldr:
            st.info(f"**TL;DR:** {tldr}")

        # Method diagram if available
        paper_id = os.path.splitext(os.path.basename(paper["file"]))[0]
        diagram_info = PAPER_DIAGRAMS.get(paper_id)

        # Build section tabs
        section_defs = [
            ("Pipeline Diagram", None),
            ("Background", "Background"),
            ("Method", "Method"),
            ("Key Results", "Key Results"),
            ("Strengths", "Strengths"),
            ("Limitations", "Limitations"),
            ("BER Relevance", "Relevance"),
            ("Takeaways", "Takeaways"),
            ("Full Review", None),
        ]

        # Skip diagram tab if no diagram
        if not diagram_info:
            section_defs = section_defs[1:]

        tab_names = [s[0] for s in section_defs]
        tabs = st.tabs(tab_names)

        for idx, (tab_name, section_key) in enumerate(section_defs):
            with tabs[idx]:
                if tab_name == "Pipeline Diagram" and diagram_info:
                    st.markdown("#### Method Pipeline")
                    render_mermaid(diagram_info["code"], height=diagram_info["height"])
                elif tab_name == "Full Review":
                    st.markdown(content, unsafe_allow_html=False)
                elif section_key:
                    section_content = _extract_review_section(content, section_key)
                    if section_content:
                        st.markdown(section_content)
                    else:
                        st.info(f"Section '{tab_name}' not found in this review.")

elif level == "L3":
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
