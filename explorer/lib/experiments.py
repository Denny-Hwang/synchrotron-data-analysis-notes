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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Public type alias: every pipeline function takes a 2-D array and kwargs
# and returns a 2-D array of the same shape.
PipelineFn = Callable[..., np.ndarray]


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
    # R11 I5 — narrative fields surfaced as a 3-card row in the Lab so
    # the user sees the impact story before tuning parameters. All
    # optional; ``description`` remains the canonical long-form text.
    problem: str = ""
    fix: str = ""
    observe: str = ""


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


_VALID_PARAM_TYPES = frozenset({"int", "float", "select"})


def _parse_parameter(d: dict) -> Parameter:
    """Parse one ``parameters[]`` entry, validating shape per type.

    Raises:
        ValueError: For unknown ``type``, missing ``default``, missing
            ``min``/``max`` on numeric, missing ``options`` on select,
            ``default`` outside ``[min, max]``, or ``default`` not in
            ``options``.
    """
    name = str(d["name"])
    p_type = str(d["type"])
    if p_type not in _VALID_PARAM_TYPES:
        raise ValueError(f"parameter '{name}': type '{p_type}' not in {sorted(_VALID_PARAM_TYPES)}")

    if "default" not in d:
        raise ValueError(f"parameter '{name}': 'default' is required")

    default = d["default"]
    p_min = d.get("min")
    p_max = d.get("max")
    options = list(d["options"]) if d.get("options") is not None else None

    if p_type in ("int", "float"):
        if p_min is None or p_max is None:
            raise ValueError(f"parameter '{name}': numeric type requires 'min' and 'max'")
        try:
            d_val = float(default)
            lo = float(p_min)
            hi = float(p_max)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"parameter '{name}': non-numeric bound or default ({exc})") from exc
        if lo > hi:
            raise ValueError(f"parameter '{name}': min ({lo}) > max ({hi})")
        if not (lo <= d_val <= hi):
            raise ValueError(f"parameter '{name}': default {d_val} outside [{lo}, {hi}]")
    elif p_type == "select":
        if not options:
            raise ValueError(f"parameter '{name}': 'select' requires 'options'")
        if default not in options:
            raise ValueError(f"parameter '{name}': default {default!r} not in options {options}")

    return Parameter(
        name=name,
        type=p_type,
        label=str(d.get("label", name)),
        default=default,
        min=p_min,
        max=p_max,
        step=d.get("step"),
        options=options,
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
        problem=str(data.get("problem", "")).strip(),
        fix=str(data.get("fix", "")).strip(),
        observe=str(data.get("observe", "")).strip(),
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

    For multi-extension FITS files, returns the first HDU whose ``data``
    is a 2-D image array.

    Raises:
        ValueError: For unsupported file extensions or FITS without image data.
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
            for hdu in hdul:
                if getattr(hdu, "data", None) is None:
                    continue
                arr = np.asarray(hdu.data)
                if arr.ndim >= 2:
                    return arr
        raise ValueError(f"No image HDU found in FITS file: {full_path}")
    raise ValueError(f"Unsupported sample format: {suffix}")


# ---------------------------------------------------------------------------
# Pipeline dispatch
# ---------------------------------------------------------------------------


_RECIPE_FUNCTION_PREFIX = "experiments."


def resolve_function(dotted_path: str) -> PipelineFn:
    """Import and return the function declared by a recipe.

    Args:
        dotted_path: ``module.submodule.function`` string. Must start
            with ``experiments.`` (R13 Rec #3 — security allow-list).
            Without this restriction a contributor could ship a
            ``recipe.yaml`` whose ``function: os.system`` would execute
            on every Lab run; PR review is the only safety net otherwise.

    Returns:
        The resolved callable. Caller is responsible for ensuring it
        matches the :data:`PipelineFn` shape (no introspection happens
        here — duck-typed at run time).

    Raises:
        ValueError: If ``dotted_path`` has no module component or does
            not target the ``experiments.`` namespace.
        ModuleNotFoundError: If the module cannot be imported.
        AttributeError: If the function does not exist in the module.
    """
    module_name, _, func_name = dotted_path.rpartition(".")
    if not module_name or not func_name:
        raise ValueError(f"Invalid function path: {dotted_path!r}")
    if not (
        module_name == _RECIPE_FUNCTION_PREFIX.rstrip(".")
        or module_name.startswith(_RECIPE_FUNCTION_PREFIX)
    ):
        raise ValueError(
            f"Recipe function {dotted_path!r} must live under the "
            f"{_RECIPE_FUNCTION_PREFIX!r} namespace; arbitrary modules "
            "are not allowed (R13 security allow-list)."
        )
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
    """Min-max normalise to [0, 1], silently zero-out NaN / inf inputs.

    Returns an all-zero array when the input has zero variance (or no
    finite values), so that downstream PSNR / SSIM see a well-defined
    operand instead of NaN. This is a defensive choice tuned for
    metric computation; it is **not** appropriate for general use as
    a normaliser.
    """
    a = arr.astype(np.float32, copy=True)
    if not np.all(np.isfinite(a)):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(a.min())
    hi = float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _align_shapes(
    a: np.ndarray, b: np.ndarray, tolerance: int = 2
) -> tuple[np.ndarray, np.ndarray] | None:
    """Centre-crop two 2-D arrays to their common minimum shape.

    Sarepy ships sinograms with off-by-one angular sampling (e.g. clean
    reference is ``(1801, 2560)`` while the noisy variants are
    ``(1800, 2560)``). The first 1800 angles are the same scene, so a
    centre crop of the larger array gives a meaningful PSNR / SSIM.

    Returns ``None`` if the shapes differ by more than ``tolerance`` in
    any dim — at that point the arrays are different scenes, not
    misaligned versions of the same one.
    """
    if a.shape == b.shape:
        return a, b
    if a.ndim != 2 or b.ndim != 2:
        return None
    if any(abs(sa - sb) > tolerance for sa, sb in zip(a.shape, b.shape, strict=True)):
        return None
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    def crop(x: np.ndarray) -> np.ndarray:
        oh = (x.shape[0] - h) // 2
        ow = (x.shape[1] - w) // 2
        return x[oh : oh + h, ow : ow + w]

    return crop(a), crop(b)


def compute_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    metrics: list[str] | tuple[str, ...],
    *,
    align_tolerance: int = 2,
) -> dict[str, float]:
    """Compute the requested metrics against ``reference``.

    Both arrays are min-max normalised to ``[0, 1]`` before metric
    computation so that comparisons across raw / processed dtypes are
    meaningful.

    If ``reference.shape != candidate.shape`` but the difference is at
    most ``align_tolerance`` along each dim, both arrays are
    centre-cropped to the common minimum shape (handles Sarepy's
    off-by-one angular sampling).  Beyond that tolerance the function
    raises :class:`ValueError`.

    Unknown metric names are silently ignored (warning logged).
    """
    if reference.shape != candidate.shape:
        aligned = _align_shapes(reference, candidate, tolerance=align_tolerance)
        if aligned is None:
            raise ValueError(
                f"shape mismatch beyond tolerance ({align_tolerance}): "
                f"reference {reference.shape} vs candidate {candidate.shape}"
            )
        reference, candidate = aligned
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
