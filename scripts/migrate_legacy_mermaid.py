"""One-shot migration: lift every Mermaid diagram from the legacy
``eberlight-explorer/`` pages into the corresponding note markdown.

The legacy app shipped 35+ Mermaid diagrams in three page-side
``_DIAGRAMS = {...}`` dictionaries (per ADR-009 the legacy app is now
deprecated). Per ADR-002 the new explorer reads everything from the
notes themselves — so this script copies each diagram into the
matching note's markdown body as a fenced ``mermaid`` block.

Idempotency: a target note that *already* contains a ``mermaid``
fenced block is left untouched (this protects the demo diagrams R3
shipped, e.g. ``tomogan.md``).

Run:

    python scripts/migrate_legacy_mermaid.py

The script prints one line per note showing whether it was
``inserted`` or ``skipped`` and a final summary.

Ref: ADR-002 — Notes are the single source of truth.
Ref: ADR-009 — Deprecation of eberlight-explorer/.
Ref: R9 — final feature parity (Mermaid library restoration).
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Diagram payloads — copied verbatim from the legacy pages and converted
# from Python's escaped ``\n`` to actual newlines so the Mermaid source
# is human-readable when the user clicks 📄 Source (L3).
# ---------------------------------------------------------------------------


def _decode(code: str) -> str:
    """Convert legacy ``\\n`` escapes embedded in label strings to real ``\n``."""
    return code.replace("\\n", "\n")


CATEGORY_DIAGRAMS: dict[str, dict[str, str]] = {
    "image_segmentation": {
        "code": _decode(
            """graph LR
    A["Raw Image\\n2D/3D Volume"] --> B["Encoder\\nFeature Extraction"]
    B --> C["Bottleneck\\nCompressed Features"]
    C --> D["Decoder\\nUpsampling + Skip"]
    D --> E["Pixel-wise\\nClass Map"]
    F["Ground Truth\\nLabeled Mask"] -.->|"Dice + CE Loss"| D
    style E fill:#00D4AA,color:#fff"""
        ),
        "caption": (
            "U-Net segmentation pipeline: encoder extracts multi-scale features, "
            "decoder with skip connections produces pixel-level classification."
        ),
    },
    "denoising": {
        "code": _decode(
            """graph LR
    A["Low-dose\\nNoisy Input"] --> B["Generator\\nU-Net"]
    B --> C["Denoised\\nOutput"]
    D["Discriminator\\nPatchGAN"] -.->|"Adversarial"| B
    E["VGG-16"] -.->|"Perceptual"| B
    F["Clean\\nTarget"] -.->|"L1 Pixel"| B
    style C fill:#FFB800,color:#fff"""
        ),
        "caption": (
            "GAN-based denoising: generator produces clean images while "
            "discriminator and perceptual losses preserve texture realism."
        ),
    },
    "reconstruction": {
        "code": _decode(
            """graph LR
    A["Diffraction\\nPatterns"] --> B["CNN\\nEncoder-Decoder"]
    B --> C["Phase +\\nAmplitude"]
    C --> D["Overlap\\nStitching"]
    D --> E["Optional\\nIterative Refine"]
    E --> F["Reconstructed\\nObject"]
    style F fill:#1B3A5C,color:#fff"""
        ),
        "caption": (
            "CNN replaces iterative phase retrieval: single forward pass produces "
            "initial reconstruction, optional refinement recovers fine details."
        ),
    },
    "autonomous_experiment": {
        "code": _decode(
            """graph LR
    A["Measurement"] --> B["Feature\\nExtraction"]
    B --> C["AI Decision\\nEngine"]
    C --> D["Next Action:\\nScan / Move / Stop"]
    D -->|"feedback loop"| A
    E["Prior\\nKnowledge"] -.-> C
    style C fill:#E8515D,color:#fff"""
        ),
        "caption": (
            "Autonomous experiment loop: AI analyzes live measurement data and "
            "decides next experimental action without human intervention."
        ),
    },
    "multimodal_integration": {
        "code": _decode(
            """graph TB
    A["XRF Maps"] --> D["Joint\\nFeature Space"]
    B["Ptychography"] --> D
    C["Spectroscopy"] --> D
    D --> E["Correlation\\nAnalysis"]
    E --> F["Fused\\nInsight"]
    style D fill:#9B59B6,color:#fff"""
        ),
        "caption": (
            "Multimodal integration fuses data from multiple X-ray techniques "
            "into a shared representation for richer scientific insight."
        ),
    },
}

METHOD_DIAGRAMS: dict[str, str] = {
    "unet_variants": _decode(
        """graph TB
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
    D -.->|skip| G"""
    ),
    "deep_residual_xrf": _decode(
        """graph LR
    A["Low-res XRF"] --> B["Bicubic Upscale"]
    B --> C["Deep Residual\\nBlocks x16"]
    C --> D["Residual\\nLearning"]
    D --> E["Super-resolved\\nXRF Map"]"""
    ),
    "ptychonet": _decode(
        """graph LR
    A["Diffraction Pattern"] --> B["CNN Encoder"]
    B --> C["Latent 4x4x512"]
    C --> D["CNN Decoder"]
    D --> E["Phase + Amplitude"]
    E --> F["ePIE Refine 5-20 iter"]"""
    ),
    "ai_nerd": _decode(
        """graph LR
    A["XPCS Speckle"] --> B["Feature Extraction"]
    B --> C["Unsupervised\\nFingerprinting"]
    C --> D["Dynamics Map"]
    D --> E["Decision Engine"]
    E -->|"next"| A"""
    ),
    "roi_finder": _decode(
        """graph LR
    A["XRF Survey"] --> B["PCA k=3-5"]
    B --> C["Fuzzy C-Means"]
    C --> D["Membership Maps"]
    D --> E["ROI Scoring"]
    E --> F["Bounding Boxes"]"""
    ),
    "bayesian_optimization": _decode(
        """graph LR
    A["Initial Samples"] --> B["Surrogate Model\\nGaussian Process"]
    B --> C["Acquisition\\nFunction"]
    C --> D["Next Sample\\nPoint"]
    D --> E["Experiment"]
    E -->|"update"| B"""
    ),
    "knowledge_injected_bo": _decode(
        """graph LR
    A["Seed Points"] --> B["GP Surrogate"]
    B --> C["Knowledge-Injected\\nAcquisition"]
    C --> D["Next Energy E*"]
    D --> E["Measure XANES"]
    E -->|"update GP"| B
    F["Edge Prior"] -.-> C
    G["Gradient ∂μ/∂E"] -.-> C"""
    ),
    "noise2void": _decode(
        """graph LR
    A["Single Noisy\\nImage"] --> B["Blind-Spot\\nMasking"]
    B --> C["U-Net\\nEncoder-Decoder"]
    C --> D["Predict Masked\\nCenter Pixel"]
    D --> E["Self-supervised\\nLoss"]
    E -->|"iterate"| B
    style D fill:#00D4AA,color:#fff"""
    ),
    "diffusion_ct": _decode(
        """graph LR
    A["Sparse-View\\nSinogram"] --> B["Score Network\\nEstimate ∇log p"]
    B --> C["Reverse\\nDiffusion"]
    C --> D["Data Consistency\\nProjection"]
    D --> E["Reconstructed\\nVolume"]
    C -->|"denoise step"| C
    style E fill:#1B3A5C,color:#fff"""
    ),
    "pinns_xray": _decode(
        """graph LR
    A["Sparse\\nMeasurements"] --> B["Neural Network\\nf(x,y,z)"]
    B --> C["Physics Loss\\nForward Model"]
    C --> D["Data Loss\\nMeasurement Fit"]
    D --> E["Physically\\nConsistent Recon"]
    F["PDE\\nConstraints"] -.-> C
    style E fill:#9B59B6,color:#fff"""
    ),
    "foundation_models_beamline": _decode(
        """graph LR
    A["Natural Language\\nCommand"] --> B["LLM/VLM\\nFoundation Model"]
    B --> C["Experiment\\nPlan"]
    C --> D["Beamline\\nExecution"]
    D --> E["Live Data\\nFeedback"]
    E -->|"adapt"| B
    F["Pre-trained\\nVision Model"] -.-> B
    style B fill:#E8515D,color:#fff"""
    ),
}

PAPER_DIAGRAMS: dict[str, str] = {
    "review_tomogan_2020": _decode(
        """graph LR
    A["High-Dose Projections"] --> B["Dose Reduction"]
    B --> C["FBP Recon via TomoPy"]
    C --> D["Noisy Slices"]
    D --> E["TomoGAN Generator"]
    E --> F["Denoised Slices"]
    F --> G["Segmentation & Analysis"]
    H["PatchGAN Discriminator"] -.-> E
    I["VGG-16 Perceptual Loss"] -.-> E
    J["L1 Pixel Loss"] -.-> E"""
    ),
    "review_roi_finder_2022": _decode(
        """graph LR
    A["Multi-element XRF Survey"] --> B["Spectral Fitting MAPS"]
    B --> C["Elemental Concentration Maps"]
    C --> D["PCA k=3-5"]
    D --> E["Fuzzy C-Means c=3-8"]
    E --> F["Membership Thresholding"]
    F --> G["ROI Scoring"]
    G --> H["Ranked ROI Boxes"]
    H --> I["Beamline Controller"]"""
    ),
    "review_xrf_gmm_2013": _decode(
        """graph LR
    A["XRF Raster Scan"] --> B["Spectral Fitting"]
    B --> C["7-Channel Elemental Maps"]
    C --> D["Normalize"]
    D --> E["GMM via EM"]
    E --> F["BIC Sweep K=2-8"]
    F --> G["Posterior Probability Maps"]
    G --> H["Component Identification"]"""
    ),
    "review_ptychonet_2019": _decode(
        """graph LR
    A["Diffraction Patterns"] --> B["Log-scale & Normalize"]
    B --> C["CNN Encoder-Decoder"]
    C --> D["Amplitude & Phase Patches"]
    D --> E["Overlap-weighted Stitching"]
    E --> F["Optional ePIE Refinement"]
    F --> G["Final Reconstruction"]"""
    ),
    "review_fullstack_dl_tomo_2023": _decode(
        """graph TB
    A["Raw Projections"] --> B["Preprocessing"]
    B --> C["Reconstruction"]
    C --> D["Denoising"]
    D --> E["Segmentation"]
    E --> F["Quantification"]
    F --> G["Visualization"]"""
    ),
    "review_ai_nerd_2024": _decode(
        """graph LR
    A["XPCS Measurement"] --> B["Speckle Pattern Analysis"]
    B --> C["AI-NERD Feature Extraction"]
    C --> D["Unsupervised Fingerprinting"]
    D --> E["Dynamics Classification"]
    E --> F["Autonomous Decision"]
    F -->|"next measurement"| A"""
    ),
    "review_aidriven_xanes_2025": _decode(
        """graph LR
    A["Seed Points 5-10"] --> B["GP Surrogate Fit"]
    B --> C["Knowledge-Injected\\nAcquisition Function"]
    C --> D["Select Next Energy E*"]
    D --> E["Monochromator + Measure"]
    E --> F["Update GP"]
    F -->|"iterate"| B
    G["Edge Prior P_edge"] -.-> C
    H["Gradient |∂μ/∂E|"] -.-> C"""
    ),
    "review_deep_residual_xrf_2023": _decode(
        """graph LR
    A["Low-res XRF Map"] --> B["Upscale Interpolation"]
    B --> C["Deep Residual Network"]
    C --> D["Residual Learning"]
    D --> E["Super-resolved XRF Map"]
    F["High-res Ground Truth"] -.-> C"""
    ),
    "review_ai_edge_ptychography_2023": _decode(
        """graph LR
    A["Detector Stream"] --> B["Edge FPGA/GPU"]
    B --> C["Lightweight CNN"]
    C --> D["Real-time Phase"]
    D --> E["Feedback to Scan"]
    F["Full Recon on HPC"] -.-> D"""
    ),
    "review_aiedge_ptycho_2023": _decode(
        """graph LR
    A["Detector Stream"] --> B["Edge FPGA/GPU"]
    B --> C["Lightweight CNN"]
    C --> D["Real-time Phase"]
    D --> E["Feedback to Scan"]
    F["Full Recon on HPC"] -.-> D"""
    ),
    "review_realtime_uct_hpc_2020": _decode(
        """graph LR
    A["Detector @ 2-BM"] --> B["Streaming to HPC"]
    B --> C["TomoPy Recon"]
    C --> D["GPU Filtering"]
    D --> E["Real-time 3D Volume"]
    E --> F["Live Visualization"]"""
    ),
    "review_ai_als_workshop_2024": _decode(
        """graph TB
    A["AI@ALS Workshop 2024"] --> B["Autonomous Experiments"]
    A --> C["Real-time Analysis"]
    A --> D["Data Management"]
    B --> E["Adaptive Scanning"]
    C --> F["Edge Computing"]
    D --> G["FAIR Data Practices"]"""
    ),
    "review_alphafold_2021": _decode(
        """graph LR
    A["Amino Acid Sequence"] --> B["MSA + Templates"]
    B --> C["Evoformer"]
    C --> D["Structure Module"]
    D --> E["3D Coordinates"]
    E --> F["Confidence pLDDT"]"""
    ),
    "review_fullstack_tomo_2023": _decode(
        """graph TB
    A["Raw Projections"] --> B["Preprocessing"]
    B --> C["Reconstruction"]
    C --> D["Denoising"]
    D --> E["Segmentation"]
    E --> F["Quantification"]
    F --> G["Visualization"]"""
    ),
    "review_noise2void_2019": _decode(
        """graph LR
    A["Single Noisy Image"] --> B["Random Blind-Spot Masking"]
    B --> C["U-Net Prediction"]
    C --> D["Self-supervised Loss"]
    D --> E["Denoised Output"]
    F["No Clean Target Needed"] -.-> D"""
    ),
    "review_diffusion_ct_2024": _decode(
        """graph LR
    A["Sparse-View Sinogram"] --> B["FBP Initial Recon"]
    B --> C["Forward Diffusion\\n(Add Noise)"]
    C --> D["Score Network\\nReverse Diffusion"]
    D --> E["Data Consistency\\nStep"]
    E --> F["Clean Reconstruction"]
    D -->|"iterate T steps"| D"""
    ),
    "review_httomo_2024": _decode(
        """graph TB
    A["Raw Projections"] --> B["YAML Pipeline Config"]
    B --> C["GPU Preprocessing\\n(Ring, Stripe, Norm)"]
    C --> D["GPU Reconstruction\\n(FBP/Gridrec/CGLS)"]
    D --> E["Post-processing\\n(Segmentation)"]
    E --> F["HDF5/TIFF Output"]"""
    ),
    "review_multimodal_synchrotron_data_2025": _decode(
        """graph TB
    A["Sample"] --> B["XRF Mapping"]
    A --> C["Ptychography"]
    A --> D["Micro-CT"]
    B --> E["Paired 3D Dataset"]
    C --> E
    D --> E
    E --> F["ML Benchmark\\nTraining"]"""
    ),
    "review_hallucination_tomo_2021": _decode(
        """graph LR
    A["Sparse-View Data"] --> B["DL Reconstruction"]
    B --> C["Output Image"]
    C --> D["sFRC Analysis"]
    D --> E["Hallucination Map"]
    F["Reference FBP"] -.-> D"""
    ),
}

# ---------------------------------------------------------------------------
# Target paths
# ---------------------------------------------------------------------------

CATEGORY_TO_NOTE: dict[str, str] = {
    "image_segmentation": "03_ai_ml_methods/image_segmentation/README.md",
    "denoising": "03_ai_ml_methods/denoising/README.md",
    "reconstruction": "03_ai_ml_methods/reconstruction/README.md",
    "autonomous_experiment": "03_ai_ml_methods/autonomous_experiment/README.md",
    "multimodal_integration": "03_ai_ml_methods/multimodal_integration/README.md",
}

METHOD_TO_NOTE: dict[str, str] = {
    "unet_variants": "03_ai_ml_methods/image_segmentation/unet_variants.md",
    "deep_residual_xrf": "03_ai_ml_methods/denoising/deep_residual_xrf.md",
    "ptychonet": "03_ai_ml_methods/reconstruction/ptychonet.md",
    "ai_nerd": "03_ai_ml_methods/autonomous_experiment/ai_nerd.md",
    "roi_finder": "03_ai_ml_methods/autonomous_experiment/roi_finder.md",
    "bayesian_optimization": "03_ai_ml_methods/autonomous_experiment/bayesian_optimization.md",
    "knowledge_injected_bo": "03_ai_ml_methods/autonomous_experiment/knowledge_injected_bo.md",
    "noise2void": "03_ai_ml_methods/denoising/noise2void.md",
    "diffusion_ct": "03_ai_ml_methods/reconstruction/diffusion_ct.md",
    "pinns_xray": "03_ai_ml_methods/reconstruction/pinns_xray.md",
    "foundation_models_beamline": (
        "03_ai_ml_methods/autonomous_experiment/foundation_models_beamline.md"
    ),
}

# Paper diagrams all live under one folder.
PAPER_NOTE_BASE = "04_publications/ai_ml_synchrotron"

# ---------------------------------------------------------------------------
# Insertion logic
# ---------------------------------------------------------------------------


_HAS_MERMAID_RE = re.compile(r"^[ \t]*```mermaid\b", re.MULTILINE)


def _has_mermaid(text: str) -> bool:
    return bool(_HAS_MERMAID_RE.search(text))


def _insert_diagram(text: str, code: str, caption: str) -> str:
    """Append an ``## Architecture diagram`` section with the Mermaid block.

    We append rather than splice — the destination notes are diverse
    in structure and a top-anchor heuristic would mis-place the block
    in some. Authors can later move the section if desired.
    """
    if not text.endswith("\n"):
        text += "\n"
    block = (
        "\n## Architecture diagram\n\n"
        + (f"_{caption}_\n\n" if caption else "")
        + "```mermaid\n"
        + code.rstrip()
        + "\n```\n"
    )
    return text + block


def _migrate_one(target_path: Path, code: str, caption: str = "") -> str:
    """Return one of ``inserted``, ``skipped (existing diagram)``, ``skipped (missing)``."""
    if not target_path.exists():
        return "skipped (missing target)"
    text = target_path.read_text(encoding="utf-8")
    if _has_mermaid(text):
        return "skipped (existing diagram)"
    new_text = _insert_diagram(text, code, caption)
    target_path.write_text(new_text, encoding="utf-8")
    return "inserted"


def main() -> int:
    inserted = 0
    skipped = 0
    missing = 0

    print("--- Category diagrams ---")
    for cat_id, payload in CATEGORY_DIAGRAMS.items():
        rel = CATEGORY_TO_NOTE[cat_id]
        path = _REPO_ROOT / rel
        status = _migrate_one(path, payload["code"], payload["caption"])
        print(f"  {rel:60s} {status}")
        if status == "inserted":
            inserted += 1
        elif status.startswith("skipped (existing"):
            skipped += 1
        else:
            missing += 1

    print("--- Method diagrams ---")
    for method_id, code in METHOD_DIAGRAMS.items():
        rel = METHOD_TO_NOTE[method_id]
        path = _REPO_ROOT / rel
        status = _migrate_one(path, code)
        print(f"  {rel:60s} {status}")
        if status == "inserted":
            inserted += 1
        elif status.startswith("skipped (existing"):
            skipped += 1
        else:
            missing += 1

    print("--- Paper diagrams ---")
    for paper_id, code in PAPER_DIAGRAMS.items():
        rel = f"{PAPER_NOTE_BASE}/{paper_id}.md"
        path = _REPO_ROOT / rel
        status = _migrate_one(path, code)
        print(f"  {rel:60s} {status}")
        if status == "inserted":
            inserted += 1
        elif status.startswith("skipped (existing"):
            skipped += 1
        else:
            missing += 1

    total = len(CATEGORY_DIAGRAMS) + len(METHOD_DIAGRAMS) + len(PAPER_DIAGRAMS)
    print()
    print(
        f"Summary: {inserted} inserted, {skipped} skipped (already had Mermaid), "
        f"{missing} missing. Total {total}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
