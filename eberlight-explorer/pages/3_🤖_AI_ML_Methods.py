"""AI/ML Methods Taxonomy Map (F3)."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.content_parser import load_yaml, read_local_file
from components.level_selector import render_level_selector
from components.common_styles import inject_styles
from components.markdown_viewer import render_markdown
from components.mermaid_diagram import render_mermaid
from components.visjs_graph import render_visjs_graph

st.set_page_config(page_title="AI/ML Methods", page_icon="🤖", layout="wide")

inject_styles()

level = render_level_selector(key="aiml_level")
categories = load_yaml("method_taxonomy.yaml")["categories"]
modalities = load_yaml("modality_metadata.yaml")["modalities"]
papers = load_yaml("publication_catalog.yaml")["publications"]

st.title("🤖 AI/ML Methods")
st.markdown("Machine learning and AI methods applied to synchrotron data analysis.")

# ── Category-to-paper mapping ────────────────────────
_cat_paper_map = {}
for p in papers:
    _cat_paper_map.setdefault(p["category"], []).append(p)

# ── Representative diagrams per category ──────────────
CATEGORY_DIAGRAMS = {
    "image_segmentation": {
        "code": """graph LR
    A["Raw Image\\n2D/3D Volume"] --> B["Encoder\\nFeature Extraction"]
    B --> C["Bottleneck\\nCompressed Features"]
    C --> D["Decoder\\nUpsampling + Skip"]
    D --> E["Pixel-wise\\nClass Map"]
    F["Ground Truth\\nLabeled Mask"] -.->|"Dice + CE Loss"| D
    style E fill:#00D4AA,color:#fff""",
        "caption": "U-Net segmentation pipeline: encoder extracts multi-scale features, decoder with skip connections produces pixel-level classification.",
        "height": 280,
    },
    "denoising": {
        "code": """graph LR
    A["Low-dose\\nNoisy Input"] --> B["Generator\\nU-Net"]
    B --> C["Denoised\\nOutput"]
    D["Discriminator\\nPatchGAN"] -.->|"Adversarial"| B
    E["VGG-16"] -.->|"Perceptual"| B
    F["Clean\\nTarget"] -.->|"L1 Pixel"| B
    style C fill:#FFB800,color:#fff""",
        "caption": "GAN-based denoising: generator produces clean images while discriminator and perceptual losses preserve texture realism.",
        "height": 280,
    },
    "reconstruction": {
        "code": """graph LR
    A["Diffraction\\nPatterns"] --> B["CNN\\nEncoder-Decoder"]
    B --> C["Phase +\\nAmplitude"]
    C --> D["Overlap\\nStitching"]
    D --> E["Optional\\nIterative Refine"]
    E --> F["Reconstructed\\nObject"]
    style F fill:#1B3A5C,color:#fff""",
        "caption": "CNN replaces iterative phase retrieval: single forward pass produces initial reconstruction, optional refinement recovers fine details.",
        "height": 280,
    },
    "autonomous_experiment": {
        "code": """graph LR
    A["Measurement"] --> B["Feature\\nExtraction"]
    B --> C["AI Decision\\nEngine"]
    C --> D["Next Action:\\nScan / Move / Stop"]
    D -->|"feedback loop"| A
    E["Prior\\nKnowledge"] -.-> C
    style C fill:#E8515D,color:#fff""",
        "caption": "Autonomous experiment loop: AI analyzes live measurement data and decides next experimental action without human intervention.",
        "height": 280,
    },
    "multimodal_integration": {
        "code": """graph TB
    A["XRF Maps"] --> D["Joint\\nFeature Space"]
    B["Ptychography"] --> D
    C["Spectroscopy"] --> D
    D --> E["Correlation\\nAnalysis"]
    E --> F["Fused\\nInsight"]
    style D fill:#9B59B6,color:#fff""",
        "caption": "Multimodal integration fuses data from multiple X-ray techniques into a shared representation for richer scientific insight.",
        "height": 300,
    },
}

# ── Method-level diagrams ────────────────────────────
METHOD_DIAGRAMS = {
    "unet_variants": {
        "code": """graph TB
    A["Input Image"] --> B["Conv+BN+ReLU x2"]
    B --> C["MaxPool"]
    C --> D["Conv+BN+ReLU x2"]
    D --> E["MaxPool"]
    E --> F["Bottleneck"]
    F --> G["UpConv + Skip"]
    G --> H["Conv+BN+ReLU x2"]
    H --> I["UpConv + Skip"]
    I --> J["Conv 1x1"]
    J --> K["Segmentation Map"]
    B -.->|skip| I
    D -.->|skip| G""",
        "height": 400,
    },
    "tomogan": {
        "code": """graph LR
    A["Noisy Slice"] --> B["U-Net Generator"]
    B --> C["Denoised Slice"]
    D["PatchGAN\\nDiscriminator"] -.-> B
    E["VGG Perceptual"] -.-> B
    F["L1 Loss"] -.-> B""",
        "height": 250,
    },
    "deep_residual_xrf": {
        "code": """graph LR
    A["Low-res XRF"] --> B["Bicubic Upscale"]
    B --> C["Deep Residual\\nBlocks x16"]
    C --> D["Residual\\nLearning"]
    D --> E["Super-resolved\\nXRF Map"]""",
        "height": 250,
    },
    "ptychonet": {
        "code": """graph LR
    A["Diffraction Pattern"] --> B["CNN Encoder"]
    B --> C["Latent 4x4x512"]
    C --> D["CNN Decoder"]
    D --> E["Phase + Amplitude"]
    E --> F["ePIE Refine 5-20 iter"]""",
        "height": 250,
    },
    "ai_nerd": {
        "code": """graph LR
    A["XPCS Speckle"] --> B["Feature Extraction"]
    B --> C["Unsupervised\\nFingerprinting"]
    C --> D["Dynamics Map"]
    D --> E["Decision Engine"]
    E -->|"next"| A""",
        "height": 250,
    },
    "roi_finder": {
        "code": """graph LR
    A["XRF Survey"] --> B["PCA k=3-5"]
    B --> C["Fuzzy C-Means"]
    C --> D["Membership Maps"]
    D --> E["ROI Scoring"]
    E --> F["Bounding Boxes"]""",
        "height": 250,
    },
    "bayesian_opt": {
        "code": """graph LR
    A["Initial Samples"] --> B["Surrogate Model\\nGaussian Process"]
    B --> C["Acquisition\\nFunction"]
    C --> D["Next Sample\\nPoint"]
    D --> E["Experiment"]
    E -->|"update"| B""",
        "height": 250,
    },
    "ki_bo_xanes": {
        "code": """graph LR
    A["Seed Points"] --> B["GP Surrogate"]
    B --> C["Knowledge-Injected\\nAcquisition"]
    C --> D["Next Energy E*"]
    D --> E["Measure XANES"]
    E -->|"update GP"| B
    F["Edge Prior"] -.-> C
    G["Gradient ∂μ/∂E"] -.-> C""",
        "height": 280,
    },
}


def _build_method_graph():
    """Build vis.js graph nodes/edges for method taxonomy."""
    nodes = []
    edges = []
    mod_map = {m["id"]: m for m in modalities}

    # Central root node
    nodes.append({
        "id": "root", "label": "AI/ML for\nSynchrotron",
        "group": "root", "size": 35,
        "color": "#1A1A2E", "level": 0,
        "font_size": 14,
    })

    # Category nodes
    for cat in categories:
        nodes.append({
            "id": cat["id"], "label": cat["name"],
            "group": "category", "size": 28,
            "color": {"image_segmentation": "#00D4AA", "denoising": "#FFB800",
                      "reconstruction": "#1B3A5C", "autonomous_experiment": "#E8515D",
                      "multimodal_integration": "#9B59B6"}.get(cat["id"], "#888"),
            "level": 1, "font_size": 13,
        })
        edges.append({"from": "root", "to": cat["id"], "label": ""})

    # Method nodes
    for cat in categories:
        for m in cat["methods"]:
            nodes.append({
                "id": m["id"], "label": m["name"],
                "group": "method", "size": 18,
                "color": {"image_segmentation": "#66E8CC", "denoising": "#FFD566",
                          "reconstruction": "#4D7A9E", "autonomous_experiment": "#F08A90",
                          "multimodal_integration": "#C39BD3"}.get(cat["id"], "#AAA"),
                "level": 2, "font_size": 11,
            })
            edges.append({"from": cat["id"], "to": m["id"], "label": ""})

    # Modality nodes
    for cat in categories:
        for mod_id in cat.get("modalities", []):
            if mod_id in mod_map:
                mod = mod_map[mod_id]
                # Only add if not already added
                if not any(n["id"] == mod_id for n in nodes):
                    nodes.append({
                        "id": mod_id, "label": mod["short_name"],
                        "group": "modality", "size": 22,
                        "color": "#FF6B6B", "level": 3,
                        "shape": "diamond", "font_size": 11,
                    })
                edges.append({
                    "from": cat["id"], "to": mod_id,
                    "label": "applies to", "dashes": True,
                })

    return nodes, edges


# ── Interactive Method Taxonomy Graph ─────────────────
st.markdown("---")
st.subheader("🗺️ Method Taxonomy Map")
st.caption("Explore the landscape of AI/ML methods for synchrotron science. "
           "Click nodes to highlight connections. Toggle hierarchy for structured view.")

method_nodes, method_edges = _build_method_graph()
render_visjs_graph(method_nodes, method_edges, hierarchical=True, height=550)

# ── Level-specific content ────────────────────────────
if level == "L0":
    st.markdown("---")
    cols = st.columns(len(categories))
    for col, cat in zip(cols, categories):
        with col:
            with st.container(border=True):
                st.markdown(f"### {cat['icon']}")
                st.markdown(f"**{cat['name']}**")
                st.caption(cat["description"])
                st.metric("Methods", len(cat["methods"]))

elif level == "L1":
    # Category overview with representative diagrams
    st.markdown("---")
    st.subheader("Method Categories")
    for cat in categories:
        with st.expander(f"{cat['icon']} {cat['name']} ({len(cat['methods'])} methods)", expanded=True):
            desc_col, diag_col = st.columns([1, 1])
            with desc_col:
                st.markdown(cat["description"])
                st.markdown("**Methods:**")
                for m in cat["methods"]:
                    st.markdown(f"- **{m['name']}**")
                applicable = ", ".join(cat["modalities"])
                st.caption(f"Applicable modalities: {applicable}")

                # Show related papers
                cat_papers = _cat_paper_map.get(cat["id"], [])
                if cat_papers:
                    st.markdown("**Related Papers:**")
                    for p in cat_papers[:3]:
                        doi = p.get("doi", "")
                        link = f" — [DOI](https://doi.org/{doi})" if doi else ""
                        st.markdown(f"- [{p['year']}] {p['title']}{link}")

            with diag_col:
                diag = CATEGORY_DIAGRAMS.get(cat["id"])
                if diag:
                    render_mermaid(diag["code"], height=diag["height"])
                    st.caption(diag["caption"])

    # Heatmap
    st.markdown("---")
    st.subheader("Modality x Method Matrix")
    import pandas as pd

    mod_names = [f"{m['icon']} {m['short_name']}" for m in modalities]
    matrix = []
    for m in modalities:
        row = {}
        for c in categories:
            row[c["name"]] = "Y" if m["id"] in c["modalities"] else "-"
        matrix.append(row)
    df = pd.DataFrame(matrix, index=mod_names)
    st.dataframe(df, use_container_width=True)

elif level == "L2":
    # Method detail cards with diagrams
    st.markdown("---")
    cat_names = [f"{c['icon']} {c['name']}" for c in categories]
    selected_cat_name = st.selectbox("Select Category", options=cat_names)
    cat_idx = cat_names.index(selected_cat_name)
    cat = categories[cat_idx]

    st.subheader(f"{cat['icon']} {cat['name']}")
    st.markdown(cat["description"])

    # Show category diagram
    cat_diag = CATEGORY_DIAGRAMS.get(cat["id"])
    if cat_diag:
        with st.expander("📊 Category Overview Diagram", expanded=True):
            render_mermaid(cat_diag["code"], height=cat_diag["height"])
            st.caption(cat_diag["caption"])

    method_names = [m["name"] for m in cat["methods"]]
    selected_method = st.selectbox("Select Method", options=method_names)
    method = next(m for m in cat["methods"] if m["name"] == selected_method)

    st.markdown("---")

    # Show method-specific diagram if available
    m_diag = METHOD_DIAGRAMS.get(method["id"])
    if m_diag:
        diag_col, content_col = st.columns([1, 1])
        with diag_col:
            st.markdown(f"#### {method['name']} Architecture")
            render_mermaid(m_diag["code"], height=m_diag["height"])
        with content_col:
            render_markdown(method["file"], show_title=True)
    else:
        render_markdown(method["file"], show_title=True)

    # Show related papers with key results
    cat_papers = _cat_paper_map.get(cat["id"], [])
    if cat_papers:
        st.markdown("---")
        st.subheader("📚 Related Paper Results")
        for p in cat_papers:
            from utils.content_parser import read_local_file, extract_section
            content = read_local_file(p["file"])
            if not content:
                continue
            with st.expander(f"[{p['year']}] {p['title']}", expanded=False):
                key_results = extract_section(content, "Key Results")
                if key_results:
                    st.markdown(key_results)
                doi = p.get("doi", "")
                if doi:
                    st.markdown(f"[Full paper (DOI)](https://doi.org/{doi})")

elif level == "L3":
    # Source view
    all_files = []
    for cat in categories:
        for m in cat["methods"]:
            all_files.append((f"{cat['icon']} {cat['name']} / {m['name']}", m["file"]))

    selected = st.selectbox("Select file", options=all_files, format_func=lambda x: x[0])
    content = read_local_file(selected[1])
    if content:
        st.code(content, language="markdown")
