"""Tests for the noise-catalog troubleshooter (Phase R4).

The data layer parses ``09_noise_catalog/troubleshooter.yaml`` —
machine-readable companion to the prose ``troubleshooter.md`` —
into typed ``Symptom``/``Case``/``Diagnosis`` objects. These tests
guarantee the schema invariants and that the YAML actually links
to real noise-catalog files / bundled before-after images / lab
recipes.

Ref: ADR-002, ADR-008.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPLORER_DIR.parent

if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from lib.troubleshooter import (
    VALID_SEVERITIES,
    Troubleshooter,
    list_before_after_images,
    load_troubleshooter,
    severity_color,
)


@pytest.fixture(scope="module")
def real() -> Troubleshooter:
    return load_troubleshooter(_REPO_ROOT)


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------


def test_eleven_symptoms_loaded(real: Troubleshooter) -> None:
    assert len(real.symptoms) == 11


def test_every_symptom_has_unique_id(real: Troubleshooter) -> None:
    ids = [s.id for s in real.symptoms]
    assert len(ids) == len(set(ids)), f"duplicate symptom ids: {ids}"


def test_every_symptom_has_at_least_one_case(real: Troubleshooter) -> None:
    for s in real.symptoms:
        assert s.cases, f"symptom {s.id!r} has no cases"


def test_every_diagnosis_severity_is_canonical(real: Troubleshooter) -> None:
    for diag in real.all_diagnoses():
        assert diag.severity in VALID_SEVERITIES, (
            f"diagnosis {diag.name!r} has bad severity {diag.severity!r}"
        )


def test_every_diagnosis_has_a_guide_path(real: Troubleshooter) -> None:
    for diag in real.all_diagnoses():
        assert diag.guide, f"diagnosis {diag.name!r} has empty guide path"


# ---------------------------------------------------------------------------
# Real-repo cross-references
# ---------------------------------------------------------------------------


def test_every_guide_path_resolves(real: Troubleshooter) -> None:
    """Each ``guide`` field must point to a real file under 09_noise_catalog/."""
    base = _REPO_ROOT / "09_noise_catalog"
    missing = []
    for diag in real.all_diagnoses():
        full = base / diag.guide
        if not full.exists():
            missing.append(f"{diag.name}: {diag.guide}")
    assert not missing, "missing noise-catalog guides:\n" + "\n".join(missing)


def test_every_image_filename_resolves(real: Troubleshooter) -> None:
    """When a diagnosis declares an image, the file must exist."""
    img_dir = _REPO_ROOT / "09_noise_catalog" / "images"
    missing = []
    for diag in real.all_diagnoses():
        if not diag.image:
            continue
        if not (img_dir / diag.image).exists():
            missing.append(f"{diag.name}: {diag.image}")
    assert not missing, "missing before/after images:\n" + "\n".join(missing)


def test_every_recipe_id_resolves_to_a_bundled_recipe(real: Troubleshooter) -> None:
    """When a diagnosis declares a recipe, that recipe must exist in experiments/."""
    from lib.experiments import load_recipes

    recipes = {r.recipe_id for r in load_recipes(_REPO_ROOT / "experiments")}
    if not recipes:
        pytest.skip("no recipes bundled")

    bad = []
    for diag in real.all_diagnoses():
        if diag.recipe and diag.recipe not in recipes:
            bad.append(f"{diag.name}: {diag.recipe}")
    assert not bad, "diagnosis recipe ids not in experiments/:\n" + "\n".join(bad)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------


def test_image_discovery_strips_suffix() -> None:
    images = list_before_after_images(_REPO_ROOT)
    if not images:
        pytest.skip("no images bundled")
    for stem in images:
        assert "_before_after" not in stem, (
            f"image key should be the noise stem, not the full filename: {stem}"
        )


def test_image_discovery_returns_paths_that_exist() -> None:
    for stem, path in list_before_after_images(_REPO_ROOT).items():
        assert path.is_file(), f"image discovery returned non-file: {stem} → {path}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_severity_color_is_hex_for_every_canonical_value() -> None:
    for s in VALID_SEVERITIES:
        assert severity_color(s).startswith("#")


def test_severity_color_falls_back_for_unknown() -> None:
    assert severity_color("nonsense").startswith("#")


# ---------------------------------------------------------------------------
# by_id lookup
# ---------------------------------------------------------------------------


def test_troubleshooter_by_id_returns_match(real: Troubleshooter) -> None:
    s = real.symptoms[0]
    assert real.by_id(s.id) is s


def test_troubleshooter_by_id_returns_none_for_missing(real: Troubleshooter) -> None:
    assert real.by_id("does-not-exist") is None
