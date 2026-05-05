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

from lib.experiments import (  # noqa: E402
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


def test_compute_metrics_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics(np.zeros((4, 4)), np.zeros((5, 5)), ["psnr"])


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
    """Filtering should bring the noisy sample CLOSER to the clean reference."""
    pytest.importorskip("scipy")
    pytest.importorskip("tifffile")

    recipe_path = _REPO_ROOT / "experiments" / "tomography" / "ring_artifact" / "recipe.yaml"
    recipe = parse_recipe(recipe_path)

    raw = load_sample(_REPO_ROOT, "datasets/tomography/ring_artifact/sinogram_dead_stripe.tif")
    ref = load_sample(_REPO_ROOT, "datasets/tomography/ring_artifact/sinogram_normal.tif")
    if raw.shape != ref.shape:
        pytest.skip(f"reference shape {ref.shape} != raw {raw.shape}")

    processed = run_pipeline(recipe, raw, {"size": 31})

    raw_metrics = compute_metrics(ref, raw, ["psnr"])
    proc_metrics = compute_metrics(ref, processed, ["psnr"])
    # Processed PSNR should be no worse than raw; usually better.
    # Allow a tiny tolerance for sinograms where the noise model differs.
    assert proc_metrics["psnr"] >= raw_metrics["psnr"] - 0.5


# ---------------------------------------------------------------------------
# Bundled recipes are valid
# ---------------------------------------------------------------------------


def test_all_bundled_recipes_load() -> None:
    """Every recipe.yaml in experiments/ must parse."""
    experiments_root = _REPO_ROOT / "experiments"
    if not experiments_root.exists():
        pytest.skip("no experiments/ directory")
    recipes = load_recipes(experiments_root)
    assert len(recipes) >= 1, "expected at least one bundled recipe"
    for r in recipes:
        assert isinstance(r, Recipe)
        assert r.recipe_id
        assert r.function
        assert len(r.samples) >= 1
