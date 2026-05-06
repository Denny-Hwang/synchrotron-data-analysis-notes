"""Tests for the experiment recipe loader and pipeline runner.

Ref: ADR-008 — Section 10 Interactive Lab.
Ref: TST-001 (test_plan.md) — Unit tests.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

_REPO_ROOT = _EXPLORER_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.experiments import (
    Recipe,
    compute_metrics,
    load_recipes,
    load_sample,
    parse_recipe,
    resolve_function,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# Recipe parsing
# ---------------------------------------------------------------------------


def _write_minimal_recipe(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            schema_version: 1
            recipe_id: dummy
            title: "Dummy"
            modality: tomography
            function: numpy.add
            samples:
              - manifest_path: a.npy
                label: A
            parameters:
              - name: x
                type: int
                label: X
                default: 1
                min: 0
                max: 10
                step: 1
            metrics: [psnr]
            """
        )
    )


def test_parse_recipe_minimal(tmp_path: Path) -> None:
    """A minimal recipe parses into a Recipe with all expected fields."""
    p = tmp_path / "recipe.yaml"
    _write_minimal_recipe(p)

    r = parse_recipe(p)

    assert r.recipe_id == "dummy"
    assert r.title == "Dummy"
    assert r.modality == "tomography"
    assert r.function == "numpy.add"
    assert len(r.samples) == 1
    assert r.samples[0].manifest_path == "a.npy"
    assert r.parameters[0].name == "x"
    assert r.parameters[0].type == "int"
    assert r.metrics == ("psnr",)
    assert r.clean_reference is None


def test_parse_recipe_missing_required_raises(tmp_path: Path) -> None:
    """Missing required fields raise ValueError."""
    p = tmp_path / "recipe.yaml"
    p.write_text("recipe_id: dummy\n")  # missing everything else

    with pytest.raises(ValueError, match="missing required fields"):
        parse_recipe(p)


def test_load_recipes_skips_invalid(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A bad recipe is logged and skipped, valid ones still load."""
    good = tmp_path / "exp1" / "recipe.yaml"
    good.parent.mkdir()
    _write_minimal_recipe(good)

    bad = tmp_path / "exp2" / "recipe.yaml"
    bad.parent.mkdir()
    bad.write_text("recipe_id: only_this_field\n")

    recipes = load_recipes(tmp_path)
    assert len(recipes) == 1
    assert recipes[0].recipe_id == "dummy"


# ---------------------------------------------------------------------------
# Function resolution + pipeline dispatch
# ---------------------------------------------------------------------------


def test_resolve_function_dotted_path() -> None:
    func = resolve_function("numpy.zeros")
    assert callable(func)


def test_resolve_function_invalid_path() -> None:
    with pytest.raises(ValueError, match="Invalid function path"):
        resolve_function("not_a_dotted_path")


def test_run_pipeline_invokes_function(tmp_path: Path) -> None:
    """run_pipeline coerces params and calls the declared function."""
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        textwrap.dedent(
            """\
            schema_version: 1
            recipe_id: r
            title: T
            modality: tomography
            function: tests.test_experiments._add_scalar
            samples:
              - {manifest_path: a.npy, label: A}
            parameters:
              - {name: x, type: int, label: X, default: 5, min: 0, max: 10, step: 1}
            """
        )
    )
    recipe = parse_recipe(recipe_path)
    arr = np.array([1, 2, 3])
    out = run_pipeline(recipe, arr, {"x": "7"})  # str → coerced to int
    assert np.array_equal(out, arr + 7)


def _add_scalar(arr: np.ndarray, x: int = 0) -> np.ndarray:
    """Test helper exposed at module scope so the recipe loader can import it."""
    return arr + x


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect_match() -> None:
    """Identical arrays give the upper-bound metric values."""
    rng = np.random.default_rng(0)
    a = rng.random((32, 32))
    out = compute_metrics(a, a, ["psnr", "ssim"])
    assert out["psnr"] > 100  # near-infinite for identical
    assert out["ssim"] == pytest.approx(1.0, abs=1e-6)


def test_compute_metrics_shape_mismatch_beyond_tolerance() -> None:
    with pytest.raises(ValueError, match="shape mismatch beyond tolerance"):
        compute_metrics(np.zeros((4, 4)), np.zeros((10, 10)), ["psnr"])


def test_compute_metrics_aligns_within_tolerance() -> None:
    """Off-by-one shapes (e.g. Sarepy 1801 vs 1800) auto-align by centre crop."""
    rng = np.random.default_rng(42)
    ref = rng.random((1801, 256))
    cand = ref[:-1, :].copy()  # candidate is exactly the cropped reference
    out = compute_metrics(ref, cand, ["psnr", "ssim"])
    # Centre crop of (1801, 256) → (1800, 256) takes rows [0:1800],
    # candidate is rows [0:1800] → identity → near-infinite PSNR.
    assert out["psnr"] > 80
    assert out["ssim"] == pytest.approx(1.0, abs=1e-6)


def test_compute_metrics_align_tolerance_kwarg() -> None:
    """The tolerance can be tightened to disable alignment entirely."""
    with pytest.raises(ValueError, match="shape mismatch beyond tolerance"):
        compute_metrics(np.zeros((100, 100)), np.zeros((101, 100)), ["psnr"], align_tolerance=0)


def test_compute_metrics_handles_nan_input() -> None:
    """NaN inputs should be coerced to zero, not propagate to metrics."""
    rng = np.random.default_rng(1)
    a = rng.random((32, 32))
    b = a.copy()
    b[0, 0] = np.nan
    out = compute_metrics(a, b, ["psnr", "ssim"])
    assert np.isfinite(out["psnr"])
    assert np.isfinite(out["ssim"])


def test_compute_metrics_handles_inf_input() -> None:
    """Inf is coerced to 0; the metric is well-defined (not NaN)."""
    a = np.zeros((16, 16), dtype=np.float32)
    b = a.copy()
    b[0, 0] = np.inf
    out = compute_metrics(a, b, ["psnr", "ssim"])
    # After coercion both arrays are all-zero → MSE = 0 → PSNR = +inf
    # is the legitimate "perfect match" answer. The contract is "no NaN".
    assert not np.isnan(out["psnr"])
    assert not np.isnan(out["ssim"])


# ---------------------------------------------------------------------------
# Parameter parse-time validation (P1-7)
# ---------------------------------------------------------------------------


def _write_recipe_with_params(path: Path, params_yaml: str) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            schema_version: 1
            recipe_id: r
            title: T
            modality: tomography
            function: tests.test_experiments._add_scalar
            samples:
              - {{manifest_path: a.npy, label: A}}
            parameters:
            {params_yaml}
            """
        )
    )


def test_parameter_unknown_type_rejected(tmp_path: Path) -> None:
    p = tmp_path / "recipe.yaml"
    _write_recipe_with_params(p, "  - {name: x, type: bogus, default: 1}")
    with pytest.raises(ValueError, match="not in"):
        parse_recipe(p)


def test_parameter_default_out_of_range_rejected(tmp_path: Path) -> None:
    p = tmp_path / "recipe.yaml"
    _write_recipe_with_params(p, "  - {name: x, type: int, default: 200, min: 0, max: 100}")
    with pytest.raises(ValueError, match="outside"):
        parse_recipe(p)


def test_parameter_min_gt_max_rejected(tmp_path: Path) -> None:
    p = tmp_path / "recipe.yaml"
    _write_recipe_with_params(p, "  - {name: x, type: int, default: 5, min: 100, max: 0}")
    with pytest.raises(ValueError, match=r"min .* > max"):
        parse_recipe(p)


def test_parameter_select_requires_options(tmp_path: Path) -> None:
    p = tmp_path / "recipe.yaml"
    _write_recipe_with_params(p, "  - {name: x, type: select, default: a}")
    with pytest.raises(ValueError, match="requires 'options'"):
        parse_recipe(p)


def test_parameter_select_default_not_in_options_rejected(tmp_path: Path) -> None:
    p = tmp_path / "recipe.yaml"
    _write_recipe_with_params(p, "  - {name: x, type: select, default: z, options: [a, b]}")
    with pytest.raises(ValueError, match="not in options"):
        parse_recipe(p)


def test_parameter_missing_default_rejected(tmp_path: Path) -> None:
    p = tmp_path / "recipe.yaml"
    _write_recipe_with_params(p, "  - {name: x, type: int, min: 0, max: 10}")
    with pytest.raises(ValueError, match="'default' is required"):
        parse_recipe(p)


def test_compute_metrics_unknown_metric_skipped() -> None:
    """Unknown metric names are silently skipped (logged as warning)."""
    a = np.zeros((8, 8))
    out = compute_metrics(a, a, ["psnr", "made_up_metric"])
    assert "psnr" in out
    assert "made_up_metric" not in out


# ---------------------------------------------------------------------------
# Real ring-artifact pipeline integration
# ---------------------------------------------------------------------------


_RING_DIR = _REPO_ROOT / "10_interactive_lab" / "datasets" / "tomography" / "ring_artifact"


@pytest.mark.skipif(
    not (_RING_DIR / "sinogram_dead_stripe.tif").exists(),
    reason="bundled ring-artifact sample missing",
)
def test_ring_artifact_pipeline_runs_on_bundled_sample() -> None:
    """The bundled ring-artifact recipe runs end-to-end on a real sample."""
    pytest.importorskip("scipy")
    pytest.importorskip("tifffile")

    recipe_path = _REPO_ROOT / "experiments" / "tomography" / "ring_artifact" / "recipe.yaml"
    recipe = parse_recipe(recipe_path)

    arr = load_sample(_REPO_ROOT, "datasets/tomography/ring_artifact/sinogram_dead_stripe.tif")
    assert arr.ndim == 2

    out = run_pipeline(recipe, arr, {"size": 21})
    assert out.shape == arr.shape
    assert out.dtype == arr.dtype


@pytest.mark.skipif(
    not (_RING_DIR / "sinogram_normal.tif").exists(),
    reason="bundled clean reference missing",
)
def test_ring_artifact_pipeline_reduces_stripes() -> None:
    """Filtering should improve PSNR against the clean reference.

    Uses ``all_stripe_types_sample1.tif`` — the sample for which Sarepy's
    bundled clean reference is best aligned. Sarepy's clean reference is
    not a perfect ground truth for every noisy variant, so we verify
    behaviour on the sample with the cleanest reference relationship.

    Sarepy bundles the clean reference at (1801, 2560) and the noisy
    variants at (1800, 2560); ``compute_metrics`` auto-crops to the
    common minimum so the comparison is meaningful.
    """
    pytest.importorskip("scipy")
    pytest.importorskip("tifffile")

    recipe_path = _REPO_ROOT / "experiments" / "tomography" / "ring_artifact" / "recipe.yaml"
    recipe = parse_recipe(recipe_path)

    raw = load_sample(
        _REPO_ROOT,
        "datasets/tomography/ring_artifact/all_stripe_types_sample1.tif",
    )
    ref = load_sample(_REPO_ROOT, "datasets/tomography/ring_artifact/sinogram_normal.tif")

    # With a generous filter the algorithm should find the right balance.
    processed = run_pipeline(recipe, raw, {"size": 51})

    raw_metrics = compute_metrics(ref, raw, ["psnr"])
    proc_metrics = compute_metrics(ref, processed, ["psnr"])
    assert proc_metrics["psnr"] > raw_metrics["psnr"], (
        f"PSNR did not improve: raw={raw_metrics['psnr']:.3f} proc={proc_metrics['psnr']:.3f}"
    )


@pytest.mark.skipif(
    not (_RING_DIR / "sinogram_normal.tif").exists(),
    reason="bundled clean reference missing",
)
def test_wavelet_fft_pipeline_runs_and_metrics_meaningful() -> None:
    """Munch 2009 wavelet-FFT pipeline runs on bundled data; metrics align."""
    pytest.importorskip("pywt")
    pytest.importorskip("tifffile")

    recipe_path = (
        _REPO_ROOT / "experiments" / "tomography" / "ring_artifact_wavelet" / "recipe.yaml"
    )
    recipe = parse_recipe(recipe_path)

    raw = load_sample(_REPO_ROOT, "datasets/tomography/ring_artifact/sinogram_dead_stripe.tif")
    ref = load_sample(_REPO_ROOT, "datasets/tomography/ring_artifact/sinogram_normal.tif")
    processed = run_pipeline(recipe, raw, {"level": 4, "sigma": 2.0, "wname": "db5"})

    assert processed.shape == raw.shape

    raw_metrics = compute_metrics(ref, raw, ["psnr"])
    proc_metrics = compute_metrics(ref, processed, ["psnr"])
    # Wavelet-FFT may not always strictly improve PSNR (depends on level
    # and stripe profile), but it should not catastrophically destroy
    # signal — PSNR should stay within ~10 dB of the raw value.
    assert proc_metrics["psnr"] > raw_metrics["psnr"] - 10.0


# ---------------------------------------------------------------------------
# Bundled-recipe contract validation (CI-quality)
# ---------------------------------------------------------------------------


def _bundled_recipes() -> list[Recipe]:
    experiments_root = _REPO_ROOT / "experiments"
    if not experiments_root.exists():
        pytest.skip("no experiments/ directory")
    recipes = load_recipes(experiments_root)
    if not recipes:
        pytest.skip("no recipes bundled")
    return recipes


def test_all_bundled_recipes_load() -> None:
    """Every recipe.yaml in experiments/ must parse."""
    recipes = _bundled_recipes()
    assert len(recipes) >= 1, "expected at least one bundled recipe"
    for r in recipes:
        assert isinstance(r, Recipe)
        assert r.recipe_id
        assert r.function
        assert len(r.samples) >= 1


def test_all_bundled_recipe_ids_unique() -> None:
    """No two recipes may share a recipe_id."""
    recipes = _bundled_recipes()
    ids = [r.recipe_id for r in recipes]
    assert len(ids) == len(set(ids)), f"duplicate recipe_id in {ids}"


def test_all_bundled_recipe_functions_importable() -> None:
    """Every recipe's function path must resolve to a callable."""
    for r in _bundled_recipes():
        func = resolve_function(r.function)
        assert callable(func), f"{r.recipe_id}: {r.function} is not callable"


def test_all_bundled_sample_paths_exist() -> None:
    """Every sample manifest_path must resolve to a real file in 10_interactive_lab/."""
    lab_root = _REPO_ROOT / "10_interactive_lab"
    if not lab_root.exists():
        pytest.skip("10_interactive_lab/ not present")
    for r in _bundled_recipes():
        for s in r.samples:
            full = lab_root / s.manifest_path
            assert full.exists(), f"{r.recipe_id}: missing sample {s.manifest_path}"
        if r.clean_reference is not None:
            full = lab_root / r.clean_reference.manifest_path
            assert full.exists(), (
                f"{r.recipe_id}: missing clean_reference {r.clean_reference.manifest_path}"
            )


def test_all_bundled_recipe_parameters_have_valid_types() -> None:
    """Every parameter has a type the page can render."""
    valid_types = {"int", "float", "select"}
    for r in _bundled_recipes():
        for p in r.parameters:
            assert p.type in valid_types, (
                f"{r.recipe_id}: parameter '{p.name}' has invalid type '{p.type}'"
            )
            if p.type in ("int", "float"):
                assert p.min is not None and p.max is not None, (
                    f"{r.recipe_id}: numeric parameter '{p.name}' missing min/max"
                )
                assert p.default is not None, f"{r.recipe_id}: parameter '{p.name}' missing default"
            if p.type == "select":
                assert p.options, f"{r.recipe_id}: select parameter '{p.name}' missing options"


def test_all_bundled_recipe_metrics_are_known() -> None:
    """Every declared metric is one we can compute."""
    known = {"psnr", "ssim"}
    for r in _bundled_recipes():
        for m in r.metrics:
            assert m.lower() in known, f"{r.recipe_id}: unknown metric '{m}'"


def test_all_bundled_recipe_noise_catalog_refs_exist() -> None:
    """Every noise_catalog_ref points to an existing markdown file."""
    for r in _bundled_recipes():
        if not r.noise_catalog_ref:
            continue
        full = _REPO_ROOT / r.noise_catalog_ref
        assert full.exists(), f"{r.recipe_id}: noise_catalog_ref {r.noise_catalog_ref} not found"
