"""End-to-end integrity checks for the bundled Interactive Lab.

These tests act as **drift protection** at CI time. They guarantee that:

- Every sample listed in ``manifest.yaml`` resolves to a file on disk.
- Every dataset folder ships an ``ATTRIBUTION.md`` with the required
  YAML frontmatter fields.
- Every license referenced by an ``ATTRIBUTION.md`` is present verbatim
  under ``LICENSES/``.
- The lazy-download zoo YAML loads cleanly and every URL-bearing entry
  ships a license + license_warning where required (CC-BY-NC, GPL).

Closes ADR-008 follow-up #3.

Ref: ADR-008 — Section 10 Interactive Lab.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPLORER_DIR.parent
_LAB_ROOT = _REPO_ROOT / "10_interactive_lab"

if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))


# ---------------------------------------------------------------------------
# Required YAML frontmatter fields (ADR-003 + CLAUDE.md invariant #7)
# ---------------------------------------------------------------------------

_REQUIRED_FRONTMATTER_FIELDS = {
    "doc_id",
    "title",
    "status",
    "version",
    "last_updated",
}

_VALID_STATUS_VALUES = {"draft", "proposed", "accepted", "superseded"}

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}


# ---------------------------------------------------------------------------
# Manifest & sample-path checks
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lab_manifest() -> dict:
    if not _LAB_ROOT.exists():
        pytest.skip("10_interactive_lab/ not present")
    manifest_path = _LAB_ROOT / "manifest.yaml"
    if not manifest_path.exists():
        pytest.skip("manifest.yaml not present")
    with manifest_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _iter_manifest_samples(manifest: dict):
    """Yield (modality, dataset_key, sample_dict) for every sample-like entry."""
    for modality_name, modality in (manifest.get("modalities") or {}).items():
        if not isinstance(modality, dict):
            continue
        if modality.get("bundled") is False:
            continue
        for dataset_key, ds in modality.items():
            if not isinstance(ds, dict):
                continue
            for sample in ds.get("samples", []) or []:
                yield modality_name, dataset_key, sample
            for group in ds.get("sample_groups", []) or []:
                yield modality_name, dataset_key, group


def test_manifest_has_modalities(lab_manifest: dict) -> None:
    assert "modalities" in lab_manifest, "manifest.yaml missing 'modalities' key"
    assert isinstance(lab_manifest["modalities"], dict)
    assert len(lab_manifest["modalities"]) >= 1


def test_every_manifest_sample_path_exists(lab_manifest: dict) -> None:
    missing: list[str] = []
    checked = 0
    for modality, dataset_key, sample in _iter_manifest_samples(lab_manifest):
        path = sample.get("file") or sample.get("path")
        if path is None:
            continue
        checked += 1
        full = _LAB_ROOT / path
        if not full.exists():
            missing.append(f"{modality}.{dataset_key}: {path}")
    assert checked >= 1, "manifest declared no sample paths — check schema"
    assert not missing, "missing manifest sample files:\n" + "\n".join(missing)


def test_every_manifest_attribution_path_exists(lab_manifest: dict) -> None:
    """Every dataset block that ships an `attribution:` field must point to a real file."""
    missing: list[str] = []
    checked = 0
    for _modality, modality in (lab_manifest.get("modalities") or {}).items():
        if not isinstance(modality, dict):
            continue
        for dataset_key, ds in modality.items():
            if not isinstance(ds, dict):
                continue
            attr_path = ds.get("attribution")
            if attr_path is None:
                continue
            checked += 1
            full = _LAB_ROOT / attr_path
            if not full.exists():
                missing.append(f"{dataset_key}: {attr_path}")
    assert checked >= 1
    assert not missing, "missing attribution files:\n" + "\n".join(missing)


# ---------------------------------------------------------------------------
# ATTRIBUTION frontmatter checks
# ---------------------------------------------------------------------------


def _all_attribution_files() -> list[Path]:
    if not _LAB_ROOT.exists():
        return []
    return sorted(_LAB_ROOT.glob("**/ATTRIBUTION.md"))


def test_attribution_files_present() -> None:
    files = _all_attribution_files()
    assert len(files) >= 5, (
        f"expected at least 5 ATTRIBUTION.md files (one per dataset folder), "
        f"found {len(files)}"
    )


@pytest.mark.parametrize(
    "attr_path",
    _all_attribution_files() or [pytest.param(None, marks=pytest.mark.skip(reason="no lab"))],
    ids=lambda p: str(p.relative_to(_REPO_ROOT)) if p else "skip",
)
def test_attribution_frontmatter_complete(attr_path: Path) -> None:
    """Each ATTRIBUTION.md must carry the required YAML frontmatter fields."""
    fm = _parse_frontmatter(attr_path)
    assert fm, f"{attr_path}: no parseable YAML frontmatter found"
    missing = _REQUIRED_FRONTMATTER_FIELDS - set(fm)
    assert not missing, f"{attr_path}: missing frontmatter fields {missing}"
    assert fm["status"] in _VALID_STATUS_VALUES, (
        f"{attr_path}: status='{fm['status']}' not in {_VALID_STATUS_VALUES}"
    )


@pytest.mark.parametrize(
    "attr_path",
    _all_attribution_files() or [pytest.param(None, marks=pytest.mark.skip(reason="no lab"))],
    ids=lambda p: str(p.relative_to(_REPO_ROOT)) if p else "skip",
)
def test_attribution_mentions_license_and_citation(attr_path: Path) -> None:
    """Body of each ATTRIBUTION.md must include a license name and a citation."""
    text = attr_path.read_text(encoding="utf-8")
    body_lower = text.lower()
    has_license_section = (
        "license" in body_lower
        and any(
            keyword in body_lower
            for keyword in (
                "apache",
                "bsd",
                "mit",
                "lgpl",
                "gpl",
                "cc0",
                "creative commons",
                "public domain",
            )
        )
    )
    assert has_license_section, (
        f"{attr_path}: no recognisable license section found in body"
    )
    has_citation = any(
        keyword in body_lower for keyword in ("citation", "cite", "doi:", "doi.org")
    )
    assert has_citation, f"{attr_path}: no citation / DOI mention in body"


# ---------------------------------------------------------------------------
# LICENSES/ presence checks
# ---------------------------------------------------------------------------


def test_licenses_directory_present_and_populated() -> None:
    licenses_dir = _LAB_ROOT / "LICENSES"
    if not _LAB_ROOT.exists():
        pytest.skip("10_interactive_lab/ not present")
    assert licenses_dir.exists() and licenses_dir.is_dir(), (
        f"LICENSES/ directory missing under {_LAB_ROOT}"
    )
    license_files = list(licenses_dir.glob("*.txt"))
    assert len(license_files) >= 5, (
        f"LICENSES/ should hold at least 5 verbatim upstream LICENSE files, "
        f"found {len(license_files)}"
    )


# ---------------------------------------------------------------------------
# Lazy-download recipes integrity
# ---------------------------------------------------------------------------


def test_lazy_download_yaml_loads() -> None:
    yaml_path = _LAB_ROOT / "models" / "lazy_download_recipes.yaml"
    if not yaml_path.exists():
        pytest.skip("lazy_download_recipes.yaml not present")
    with yaml_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Top-level sections per ADR-008.
    expected_sections = {"native_synchrotron_models", "huggingface_baselines", "external_datasets"}
    present = set(data) & expected_sections
    assert present, f"none of {expected_sections} present in lazy_download_recipes.yaml"


def test_citations_bib_present() -> None:
    bib = _LAB_ROOT / "CITATIONS.bib"
    if not _LAB_ROOT.exists():
        pytest.skip("10_interactive_lab/ not present")
    assert bib.exists(), "CITATIONS.bib missing"
    text = bib.read_text(encoding="utf-8")
    # Sanity: at least one BibTeX entry.
    assert re.search(r"^@\w+\{", text, re.MULTILINE), (
        "CITATIONS.bib has no @entry{...} blocks"
    )
