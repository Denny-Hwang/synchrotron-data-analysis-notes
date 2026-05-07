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

# ---------------------------------------------------------------------------
# Step 1 — recipe picker (R10 P0-4: was in sidebar; moved to main column so
# mobile users can actually change it. R10 P1-7: numbered stepper headings.)
# ---------------------------------------------------------------------------


st.markdown("#### 1️⃣ Pick a recipe")
recipe_idx = st.selectbox(
    "Recipe",
    options=list(range(len(recipes))),
    format_func=lambda i: f"{recipes[i].title}  ({recipes[i].modality})",
    label_visibility="collapsed",
    key="exp_recipe_picker",
)
recipe = recipes[recipe_idx]

st.markdown(
    f'<p style="color:#666;font-size:13px;margin-top:-8px;">'
    f"Modality: <b>{recipe.modality}</b> &nbsp;·&nbsp; "
    f"Function: <code>{recipe.function}</code> &nbsp;·&nbsp; "
    f'Catalog: <a href="../{recipe.noise_catalog_ref}">{recipe.noise_catalog_ref}</a>'
    "</p>",
    unsafe_allow_html=True,
)

# R11 I5 — surface the recipe's "what was wrong / how it's fixed / what
# to look for" narrative as a 3-card row so users understand the impact
# story before they even press a slider. Falls back to the description
# block when the optional structured fields aren't filled in.
_narrative = recipe.problem or recipe.fix or recipe.observe
if _narrative:
    n_cols = st.columns(3)
    cards = [
        ("⚠️", "What was wrong", recipe.problem or "—", "#C8550E"),
        ("🛠️", "How we fix it", recipe.fix or "—", "#0033A0"),
        ("👀", "What you should observe", recipe.observe or "—", "#2E7D32"),
    ]
    for col, (icon, title, body, color) in zip(n_cols, cards, strict=True):
        col.markdown(
            f'<div class="eberlight-card" style="border-left:4px solid {color};'
            f'min-height:140px;">'
            f'<div style="font-size:11px;color:#888;text-transform:uppercase;'
            f'letter-spacing:0.5px;font-weight:700;margin-bottom:6px;">'
            f"{icon} {title}</div>"
            f'<div style="font-size:14px;color:#1A1A1A;line-height:1.45;">{body}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

with st.expander("Background & method detail", expanded=not _narrative):
    st.markdown(recipe.description)


# ---------------------------------------------------------------------------
# Step 2 — sample picker
# ---------------------------------------------------------------------------


st.markdown("#### 2️⃣ Pick a sample")
sample_idx = st.selectbox(
    "Sample",
    options=list(range(len(recipe.samples))),
    format_func=lambda i: (
        f"{recipe.samples[i].label}  —  {recipe.samples[i].role}"
        + (f" — {recipe.samples[i].description}" if recipe.samples[i].description else "")
    ),
    label_visibility="collapsed",
    key="exp_sample_picker",
)
sample = recipe.samples[sample_idx]


# ---------------------------------------------------------------------------
# Step 3 — parameter widgets (auto-generated from recipe.yaml)
# ---------------------------------------------------------------------------


st.markdown("#### 3️⃣ Tune parameters")
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


st.markdown("#### 4️⃣ Compare before / after")


def _difference_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return |a − b| min-max normalised to [0, 1] for ``st.image``.

    The output is what the recipe **changed** about the input —
    bright pixels are where the algorithm acted, dark pixels are
    where the input passed through unchanged. Centre-crops to the
    common minimum shape if the two arrays differ slightly.
    """
    a32 = a.astype(np.float32, copy=False)
    b32 = b.astype(np.float32, copy=False)
    if a32.shape != b32.shape:
        # Centre-crop to the minimum shape on each axis.
        slices = tuple(slice(0, min(a32.shape[i], b32.shape[i])) for i in range(a32.ndim))
        a32 = a32[slices]
        b32 = b32[slices]
    delta = np.abs(a32 - b32)
    lo, hi = float(delta.min()), float(delta.max())
    if hi - lo < 1e-12:
        return np.zeros_like(delta)
    return (delta - lo) / (hi - lo)


# 3-panel comparison: Original · Processed · |Δ| difference map. R11 I5 —
# the diff panel makes the algorithm's actual effect visible at a glance,
# instead of the user having to flick their eyes back and forth.
_compare_cols = st.columns(3)
with _compare_cols[0]:
    st.markdown("**Original**")
    st.image(_to_display(sino_input), clamp=True, width="stretch")
    st.caption(f"shape={sino_input.shape}, dtype={sino_input.dtype}")
with _compare_cols[1]:
    st.markdown("**Processed**")
    st.image(_to_display(sino_output), clamp=True, width="stretch")
    st.caption(f"shape={sino_output.shape}, dtype={sino_output.dtype}")
with _compare_cols[2]:
    st.markdown("**|Δ| — what changed**")
    if sino_input.shape == sino_output.shape:
        delta_img = _difference_map(sino_input, sino_output)
    else:
        delta_img = _difference_map(sino_input, sino_output)
    st.image(delta_img, clamp=True, width="stretch")
    nonzero = float((delta_img > 0.05).mean())
    st.caption(f"Bright pixels = where the algorithm acted (~{nonzero * 100:.1f}% of the area).")


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
                # R11 I5 — bigger, more dramatic impact card. The bare
                # st.metric row was too easy to miss; we now lead with
                # a "🎯 Impact" headline and a coloured wins/loses
                # banner so users see how much the algorithm helped at
                # a single glance.
                st.markdown(
                    f"#### 🎯 Impact &nbsp;<span style='color:#888;font-size:14px;'>"
                    f"vs clean reference (<code>{recipe.clean_reference.label}</code>)</span>",
                    unsafe_allow_html=True,
                )
                if ref_arr.shape != sino_input.shape:
                    st.caption(
                        f"Reference {ref_arr.shape} centre-cropped to match "
                        f"sample {sino_input.shape} for metric computation."
                    )
                wins = 0
                losses = 0
                metric_cols = st.columns(len(recipe.metrics))
                for col, name in zip(metric_cols, recipe.metrics, strict=True):
                    in_v = in_metrics.get(name.lower())
                    out_v = out_metrics.get(name.lower())
                    if in_v is None or out_v is None:
                        continue
                    delta = out_v - in_v
                    # PSNR + SSIM both improve when they go up; treat
                    # ``higher is better`` as the convention. Recipes
                    # that introduce a "lower is better" metric will
                    # need to surface that explicitly later.
                    if delta > 0:
                        wins += 1
                    elif delta < 0:
                        losses += 1
                    col.metric(
                        label=name.upper(),
                        value=f"{out_v:.3f}",
                        delta=f"{delta:+.3f}  vs raw input ({in_v:.3f})",
                    )
                if wins or losses:
                    if losses == 0:
                        banner_bg, banner_fg, banner_msg = (
                            "#E6F4EA",
                            "#1E6B33",
                            f"✅ All {wins} metric{'s' if wins != 1 else ''} improved.",
                        )
                    elif wins == 0:
                        banner_bg, banner_fg, banner_msg = (
                            "#FDECEA",
                            "#A82618",
                            (
                                f"⚠️ All {losses} metric{'s' if losses != 1 else ''} "
                                "regressed — try different parameters."
                            ),
                        )
                    else:
                        banner_bg, banner_fg, banner_msg = (
                            "#FFF7DB",
                            "#7A5A00",
                            f"➡️ {wins} improved, {losses} regressed — partial win.",
                        )
                    st.markdown(
                        f'<div style="background:{banner_bg};color:{banner_fg};'
                        f"padding:10px 14px;border-radius:8px;margin-top:8px;"
                        f'font-size:14px;font-weight:600;">{banner_msg}</div>',
                        unsafe_allow_html=True,
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
