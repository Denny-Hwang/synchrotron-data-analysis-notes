"""Experiment recipe loader and pipeline runner for the Interactive Lab.

Reads ``recipe.yaml`` files from ``experiments/`` and dispatches to the
declared pure-function pipelines on samples bundled in
``10_interactive_lab/datasets/``.

Designed to be imported by the Streamlit Lab page
(``explorer/pages/4_Experiment.py``) and by unit tests.

Ref: ADR-008 — Section 10 Interactive Lab.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses (recipe schema mirror)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One bundled sample referenced by a recipe."""

    manifest_path: str
    label: str
    role: str = "noisy_input"
    description: str = ""


@dataclass(frozen=True)
class Parameter:
    """One tunable parameter with widget metadata."""

    name: str
    type: str  # "int" | "float" | "select"
    label: str
    default: Any
    min: Any = None
    max: Any = None
    step: Any = None
    options: list[Any] | None = None
    help: str = ""


@dataclass(frozen=True)
class Reference:
    """Citation entry attached to a recipe."""

    title: str
    authors: str
    year: int
    venue: str = ""
    doi: str = ""


@dataclass(frozen=True)
class Recipe:
    """Parsed view of a single ``recipe.yaml``."""

    recipe_id: str
    title: str
    modality: str
    noise_catalog_ref: str
    description: str
    function: str
    samples: tuple[Sample, ...]
    clean_reference: Sample | None
    parameters: tuple[Parameter, ...]
    metrics: tuple[str, ...]
    references: tuple[Reference, ...]
    source_path: Path = field(default_factory=Path)


# ---------------------------------------------------------------------------
# Recipe parsing
# ---------------------------------------------------------------------------


def _parse_sample(d: dict) -> Sample:
    return Sample(
        manifest_path=str(d["manifest_path"]),
        label=str(d.get("label", d["manifest_path"])),
        role=str(d.get("role", "noisy_input")),
        description=str(d.get("description", "")),
    )


def _parse_parameter(d: dict) -> Parameter:
    return Parameter(
        name=str(d["name"]),
        type=str(d["type"]),
        label=str(d.get("label", d["name"])),
        default=d.get("default"),
        min=d.get("min"),
        max=d.get("max"),
        step=d.get("step"),
        options=list(d["options"]) if d.get("options") is not None else None,
        help=str(d.get("help", "")),
    )


def _parse_reference(d: dict) -> Reference:
    return Reference(
        title=str(d.get("title", "")),
        authors=str(d.get("authors", "")),
        year=int(d.get("year", 0)),
        venue=str(d.get("venue", "")),
        doi=str(d.get("doi", "")),
    )


def parse_recipe(recipe_path: Path) -> Recipe:
    """Parse a single ``recipe.yaml`` into a :class:`Recipe`.

    Raises:
        ValueError: If required fields are missing.
    """
    with recipe_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    required = {"recipe_id", "title", "modality", "function", "samples"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"recipe {recipe_path} missing required fields: {missing}")

    clean_ref_raw = data.get("clean_reference")
    clean_reference = _parse_sample(clean_ref_raw) if clean_ref_raw else None

    return Recipe(
        recipe_id=str(data["recipe_id"]),
        title=str(data["title"]),
        modality=str(data["modality"]),
        noise_catalog_ref=str(data.get("noise_catalog_ref", "")),
        description=str(data.get("description", "")),
        function=str(data["function"]),
        samples=tuple(_parse_sample(s) for s in data.get("samples", [])),
        clean_reference=clean_reference,
        parameters=tuple(_parse_parameter(p) for p in data.get("parameters", [])),
        metrics=tuple(str(m) for m in data.get("metrics", [])),
        references=tuple(_parse_reference(r) for r in data.get("references", [])),
        source_path=recipe_path,
    )


def load_recipes(experiments_root: Path) -> list[Recipe]:
    """Load every ``recipe.yaml`` under ``experiments_root``.

    Bad recipes are logged and skipped so a single broken file does not
    take down the whole page.
    """
    recipes: list[Recipe] = []
    for recipe_path in sorted(experiments_root.rglob("recipe.yaml")):
        try:
            recipes.append(parse_recipe(recipe_path))
        except (yaml.YAMLError, ValueError, KeyError) as e:
            logger.warning("Skipping invalid recipe %s: %s", recipe_path, e)
    logger.info("Loaded %d recipes", len(recipes))
    return recipes


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def load_sample(repo_root: Path, manifest_path: str) -> np.ndarray:
    """Load a sample file from ``10_interactive_lab/`` into a numpy array.

    Supports ``.tif/.tiff`` (via ``tifffile``), ``.npy``, ``.fits``
    (via ``astropy.io.fits``).

    Raises:
        ValueError: For unsupported file extensions.
        FileNotFoundError: If the file does not exist.
    """
    full_path = repo_root / "10_interactive_lab" / manifest_path
    if not full_path.exists():
        raise FileNotFoundError(f"Sample not found: {full_path}")

    suffix = full_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import tifffile

        return np.asarray(tifffile.imread(str(full_path)))
    if suffix == ".npy":
        return np.load(full_path)
    if suffix == ".fits":
        from astropy.io import fits

        with fits.open(full_path) as hdul:
            return np.asarray(hdul[0].data)
    raise ValueError(f"Unsupported sample format: {suffix}")


# ---------------------------------------------------------------------------
# Pipeline dispatch
# ---------------------------------------------------------------------------


def resolve_function(dotted_path: str):
    """Import and return the function declared by a recipe."""
    module_name, _, func_name = dotted_path.rpartition(".")
    if not module_name or not func_name:
        raise ValueError(f"Invalid function path: {dotted_path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def run_pipeline(recipe: Recipe, arr: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Dispatch the recipe's pipeline function on ``arr``.

    ``params`` are passed as keyword arguments after light coercion to
    the declared types.
    """
    func = resolve_function(recipe.function)
    coerced: dict[str, Any] = {}
    for p in recipe.parameters:
        if p.name not in params:
            coerced[p.name] = p.default
            continue
        v = params[p.name]
        if p.type == "int":
            coerced[p.name] = int(v)
        elif p.type == "float":
            coerced[p.name] = float(v)
        else:
            coerced[p.name] = v
    return func(arr, **coerced)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _normalize(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    lo = float(a.min())
    hi = float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def compute_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    metrics: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Compute the requested metrics against ``reference``.

    Both arrays are min-max normalised to ``[0, 1]`` before metric computation
    so that comparisons across raw / processed dtypes are meaningful.
    Unknown metric names are silently ignored.
    """
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape} vs candidate {candidate.shape}"
        )
    ref = _normalize(reference)
    cand = _normalize(candidate)

    out: dict[str, float] = {}
    for m in metrics:
        m_lower = m.lower()
        if m_lower == "psnr":
            from skimage.metrics import peak_signal_noise_ratio

            out[m_lower] = float(peak_signal_noise_ratio(ref, cand, data_range=1.0))
        elif m_lower == "ssim":
            from skimage.metrics import structural_similarity

            out[m_lower] = float(structural_similarity(ref, cand, data_range=1.0))
        else:
            logger.warning("Unknown metric '%s' — skipping", m)
    return out
