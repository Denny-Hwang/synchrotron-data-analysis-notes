"""Experiment — Interactive Lab.

Loads recipes from ``experiments/`` and lets users replay noise mitigation
on real bundled data with parameter tuning. Side-by-side display, plus
PSNR/SSIM against a clean reference when available.

Ref: ADR-008 — Section 10 Interactive Lab.
Ref: FR-001 — landing page CTAs (Experiment is added in this revision).
Ref: 10_interactive_lab/README.md — bundled samples.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

# Make experiments/ importable (recipes resolve their function via dotted path).
_REPO_ROOT = _EXPLORER_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from components.breadcrumb import render_breadcrumb
from components.footer import render_footer
from components.header import render_header
from lib.experiments import (
    Recipe,
    compute_metrics,
    load_recipes,
    load_sample,
    run_pipeline,
)
from lib.ia import CLUSTER_META

st.set_page_config(page_title="Experiment — eBERlight", page_icon="🧪", layout="wide")

_CSS_PATH = _EXPLORER_DIR / "assets" / "styles.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

render_header()
render_breadcrumb([("Home", "/"), ("Interactive Lab", None)])

_BUILD_COLOR = CLUSTER_META["build"]["color"]

st.markdown(
    f'<h1 style="color:{_BUILD_COLOR};">Interactive Lab</h1>'
    '<p style="color:#555;font-size:16px;margin-bottom:8px;">'
    "Replay noise mitigation techniques from prior research on real bundled data. "
    "Tune parameters, compare before/after, and see PSNR/SSIM against a clean reference.</p>"
    '<p style="color:#888;font-size:13px;margin-bottom:24px;">'
    "See <code>10_interactive_lab/README.md</code> for the full dataset inventory "
    "and <code>experiments/README.md</code> for the recipe schema.</p>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def _cached_recipes() -> list[Recipe]:
    return load_recipes(_REPO_ROOT / "experiments")


@st.cache_data(show_spinner="Loading sample…")
def _cached_sample(manifest_path: str) -> np.ndarray:
    return load_sample(_REPO_ROOT, manifest_path)


@st.cache_data(show_spinner="Running pipeline…")
def _cached_run(recipe_id: str, manifest_path: str, params_items: tuple) -> np.ndarray:
    recipes = _cached_recipes()
    recipe = next(r for r in recipes if r.recipe_id == recipe_id)
    arr = load_sample(_REPO_ROOT, manifest_path)
    return run_pipeline(recipe, arr, dict(params_items))


def _to_display(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1] float32 for ``st.image``."""
    a = arr.astype(np.float32, copy=False)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Recipe selection
# ---------------------------------------------------------------------------


recipes = _cached_recipes()

if not recipes:
    st.warning(
        "No experiments found in `experiments/`. "
        "Add a `recipe.yaml` and a `pipeline.py` — see `experiments/README.md`."
    )
    render_footer()
    st.stop()

with st.sidebar:
    st.markdown("### Recipe")
    recipe_idx = st.selectbox(
        "Choose experiment",
        options=list(range(len(recipes))),
        format_func=lambda i: f"{recipes[i].title}  ({recipes[i].modality})",
        label_visibility="collapsed",
    )
recipe = recipes[recipe_idx]


# ---------------------------------------------------------------------------
# Recipe header
# ---------------------------------------------------------------------------


st.markdown(f"### {recipe.title}")
st.markdown(
    f'<p style="color:#666;font-size:13px;margin-top:-8px;">'
    f"Modality: <b>{recipe.modality}</b> &nbsp;·&nbsp; "
    f"Function: <code>{recipe.function}</code> &nbsp;·&nbsp; "
    f'Catalog: <a href="../{recipe.noise_catalog_ref}">{recipe.noise_catalog_ref}</a>'
    "</p>",
    unsafe_allow_html=True,
)
st.markdown(recipe.description)


# ---------------------------------------------------------------------------
# Sample picker
# ---------------------------------------------------------------------------


st.markdown("#### Sample")
sample_idx = st.selectbox(
    "Sample",
    options=list(range(len(recipe.samples))),
    format_func=lambda i: (
        f"{recipe.samples[i].label}  —  {recipe.samples[i].role}"
        + (f" — {recipe.samples[i].description}" if recipe.samples[i].description else "")
    ),
    label_visibility="collapsed",
)
sample = recipe.samples[sample_idx]


# ---------------------------------------------------------------------------
# Parameter widgets (auto-generated from recipe.yaml)
# ---------------------------------------------------------------------------


st.markdown("#### Parameters")
param_values: dict[str, Any] = {}
param_cols = st.columns(min(3, max(1, len(recipe.parameters))))
for i, param in enumerate(recipe.parameters):
    with param_cols[i % len(param_cols)]:
        if param.type == "int":
            val = st.slider(
                param.label,
                min_value=int(param.min),
                max_value=int(param.max),
                value=int(param.default),
                step=int(param.step or 1),
                help=param.help or None,
                key=f"{recipe.recipe_id}_{param.name}",
            )
            param_values[param.name] = int(val)
        elif param.type == "float":
            val = st.slider(
                param.label,
                min_value=float(param.min),
                max_value=float(param.max),
                value=float(param.default),
                step=float(param.step or 0.01),
                help=param.help or None,
                key=f"{recipe.recipe_id}_{param.name}",
            )
            param_values[param.name] = float(val)
        elif param.type == "select":
            options = list(param.options or [])
            default_idx = options.index(param.default) if param.default in options else 0
            val = st.selectbox(
                param.label,
                options=options,
                index=default_idx,
                help=param.help or None,
                key=f"{recipe.recipe_id}_{param.name}",
            )
            param_values[param.name] = val
        else:
            st.warning(f"Unknown parameter type '{param.type}' for '{param.name}'")


# ---------------------------------------------------------------------------
# Run + display
# ---------------------------------------------------------------------------


try:
    sino_input = _cached_sample(sample.manifest_path)
    sino_output = _cached_run(
        recipe.recipe_id,
        sample.manifest_path,
        tuple(sorted(param_values.items())),
    )
except Exception as exc:
    st.error(f"Pipeline failed: {exc}")
    st.exception(exc)
    render_footer()
    st.stop()


col_left, col_right = st.columns(2)
with col_left:
    st.markdown("**Original**")
    st.image(_to_display(sino_input), clamp=True, use_container_width=True)
    st.caption(f"shape={sino_input.shape}, dtype={sino_input.dtype}")
with col_right:
    st.markdown("**Processed**")
    st.image(_to_display(sino_output), clamp=True, use_container_width=True)
    st.caption(f"shape={sino_output.shape}, dtype={sino_output.dtype}")


# ---------------------------------------------------------------------------
# Download processed result (FR-XXX follow-up — see ADR-008 / P2-6)
# ---------------------------------------------------------------------------


def _serialise_npy(arr: np.ndarray) -> bytes:
    """Encode a numpy array as ``.npy`` bytes (no temp files)."""
    import io

    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _serialise_tiff(arr: np.ndarray) -> bytes:
    """Encode a 2-D array as a TIFF bytestring."""
    import io

    import tifffile

    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    return buf.getvalue()


_dl_col1, _dl_col2 = st.columns(2)
_safe_id = recipe.recipe_id.replace("/", "_")
_safe_sample = sample.manifest_path.split("/")[-1].rsplit(".", 1)[0]
_basename = f"{_safe_id}__{_safe_sample}__processed"

with _dl_col1:
    st.download_button(
        label="⬇ Download processed (.npy)",
        data=_serialise_npy(sino_output),
        file_name=f"{_basename}.npy",
        mime="application/octet-stream",
        help=(
            "Raw float32 array, loadable with numpy.load. Use this when "
            "you want to plug the result into your own analysis pipeline."
        ),
        key=f"{recipe.recipe_id}_dl_npy",
    )
with _dl_col2:
    if sino_output.ndim == 2:
        st.download_button(
            label="⬇ Download processed (.tiff)",
            data=_serialise_tiff(sino_output),
            file_name=f"{_basename}.tiff",
            mime="image/tiff",
            help="Lossless TIFF — for image viewers (ImageJ, Fiji, Tomviz).",
            key=f"{recipe.recipe_id}_dl_tiff",
        )


# ---------------------------------------------------------------------------
# Metrics vs clean reference
# ---------------------------------------------------------------------------


if recipe.clean_reference and recipe.metrics:
    if sample.role == "false_positive_trap":
        st.info(
            f"**False-positive trap sample.** `{sample.label}` is a different "
            f"scene from the clean reference (`{recipe.clean_reference.label}`) — "
            "what looks like stripes here is real sample structure. "
            "PSNR/SSIM against the reference are not meaningful, so they are "
            "skipped. Watch the visual output: a good algorithm should leave "
            "this image largely unchanged; a too-aggressive filter will smear "
            "the real features."
        )
    else:
        try:
            ref_arr = _cached_sample(recipe.clean_reference.manifest_path)
            try:
                in_metrics = compute_metrics(ref_arr, sino_input, list(recipe.metrics))
                out_metrics = compute_metrics(ref_arr, sino_output, list(recipe.metrics))
            except ValueError as exc:
                st.info(
                    f"Reference shape {ref_arr.shape} differs from sample "
                    f"shape {sino_input.shape} beyond the alignment tolerance; "
                    f"metrics skipped. Detail: {exc}"
                )
            else:
                st.markdown(f"#### Metrics vs clean reference (`{recipe.clean_reference.label}`)")
                if ref_arr.shape != sino_input.shape:
                    st.caption(
                        f"Reference {ref_arr.shape} centre-cropped to match "
                        f"sample {sino_input.shape} for metric computation."
                    )
                metric_cols = st.columns(len(recipe.metrics))
                for col, name in zip(metric_cols, recipe.metrics, strict=True):
                    in_v = in_metrics.get(name.lower())
                    out_v = out_metrics.get(name.lower())
                    if in_v is None or out_v is None:
                        continue
                    delta = out_v - in_v
                    col.metric(
                        label=name.upper(),
                        value=f"{out_v:.3f}",
                        delta=f"{delta:+.3f}  vs raw input ({in_v:.3f})",
                    )
        except FileNotFoundError as e:
            st.warning(f"Clean reference not found: {e}")


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


if recipe.references:
    with st.expander("References"):
        for ref in recipe.references:
            doi_link = f" [DOI](https://doi.org/{ref.doi})" if ref.doi else ""
            st.markdown(f"- **{ref.title}** — {ref.authors} ({ref.year}). _{ref.venue}_.{doi_link}")

with st.expander("Recipe metadata"):
    st.code(str(recipe.source_path.relative_to(_REPO_ROOT)), language="text")
    st.markdown(
        "All samples are bundled under `10_interactive_lab/datasets/`. "
        "See the `ATTRIBUTION.md` next to each sample for upstream source, "
        "license, and required citation."
    )

render_footer()
