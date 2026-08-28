"""R16 comprehensive-review regression tests.

Covers the defect classes found in the 2026-08 service + experiments
review:

1. **Dead deep-links** — every URL-building helper must route ``?note=``
   through a cluster page that actually reads the param (Search results,
   Knowledge-Graph entities, and the Experiment page's catalog link all
   used to emit bare ``?note=…`` hrefs that resolved against pages with
   no handler).
2. **Copy-paste drift** — ``_CLUSTER_TAGLINE`` is duplicated between
   ``lib/cluster_page.py`` and ``scripts/build_static_site.py`` (the
   static script cannot import the Streamlit module); assert the two
   stay byte-identical.
3. **Requirements drift** — ``scripts/requirements.txt`` claims to be
   "kept in sync" with ``explorer/requirements.txt`` for shared pins;
   enforce it (the Pages build broke when ``numpy`` was missing).
4. **Recipe ↔ manifest drift** — every sample path referenced by a
   ``recipe.yaml`` (including per-sample ``clean_reference`` overrides)
   must be a real file and listed in ``10_interactive_lab/manifest.yaml``.
5. **Sample roles** — ``identity_check`` samples must literally be their
   own clean reference; per-sample ``clean_reference`` overrides must be
   shape-compatible scenes.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import yaml

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPLORER_DIR.parent

if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from lib.experiments import load_recipes
from lib.ia import FOLDER_TO_CLUSTER
from lib.routing import note_url

_CLUSTER_PAGE_FILES = {
    "discover": _EXPLORER_DIR / "pages" / "1_Discover.py",
    "explore": _EXPLORER_DIR / "pages" / "2_Explore.py",
    "build": _EXPLORER_DIR / "pages" / "3_Build.py",
}


# ---------------------------------------------------------------------------
# 1 — note_url routing
# ---------------------------------------------------------------------------


def test_note_url_routes_every_folder_to_its_cluster_page() -> None:
    for folder, cluster in FOLDER_TO_CLUSTER.items():
        url = note_url(f"{folder}/some_note.md")
        assert url.startswith(f"/{cluster.title()}?note="), (
            f"{folder}: expected /{cluster.title()}?note=…, got {url!r}"
        )


def test_note_url_unknown_folder_returns_hash() -> None:
    assert note_url("not_a_note_folder/x.md") == "#"
    assert note_url("") == "#"


def test_cluster_pages_render_via_cluster_page_module() -> None:
    """The pages note_url targets must delegate to render_cluster_page,
    which is where the ``?note=`` query-param handler lives."""
    for cluster, page in _CLUSTER_PAGE_FILES.items():
        src = page.read_text(encoding="utf-8")
        assert f'render_cluster_page("{cluster}")' in src, (
            f"{page.name} no longer calls render_cluster_page — note_url deep links would break."
        )


def test_no_bare_note_query_hrefs_in_pages() -> None:
    """No page may emit a relative ``href="?note=…"`` — those resolve
    against the emitting page, which has no ``note`` handler."""
    bad = re.compile(r"""href=[\"']\?note=""")
    for page in (_EXPLORER_DIR / "pages").glob("*.py"):
        src = page.read_text(encoding="utf-8")
        assert not bad.search(src), f"{page.name} emits a bare ?note= href"


# ---------------------------------------------------------------------------
# 2 — _CLUSTER_TAGLINE parity (invariant #9)
# ---------------------------------------------------------------------------


def _load_static_builder():
    spec = importlib.util.spec_from_file_location(
        "build_static_site_under_test",
        _REPO_ROOT / "scripts" / "build_static_site.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cluster_tagline_dicts_are_identical() -> None:
    from lib.cluster_page import _CLUSTER_TAGLINE as streamlit_taglines

    static = _load_static_builder()
    assert streamlit_taglines == static._CLUSTER_TAGLINE, (
        "_CLUSTER_TAGLINE drifted between lib/cluster_page.py and "
        "scripts/build_static_site.py — update both in the same change."
    )


# ---------------------------------------------------------------------------
# 3 — requirements sync
# ---------------------------------------------------------------------------


def _parse_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*(.*)$", line)
        if m:
            pins[m.group(1).lower()] = m.group(2).replace(" ", "")
    return pins


def test_requirements_shared_pins_in_sync() -> None:
    explorer_pins = _parse_pins(_EXPLORER_DIR / "requirements.txt")
    scripts_pins = _parse_pins(_REPO_ROOT / "scripts" / "requirements.txt")
    for name, spec in scripts_pins.items():
        assert name in explorer_pins, (
            f"scripts/requirements.txt pins {name!r} which explorer/requirements.txt lacks"
        )
        assert explorer_pins[name] == spec, (
            f"{name}: scripts pin {spec!r} != explorer pin {explorer_pins[name]!r} "
            "— the two files claim to be kept in sync."
        )


# ---------------------------------------------------------------------------
# 4 + 5 — recipe ↔ manifest drift and sample-role semantics
# ---------------------------------------------------------------------------


def _manifest_paths() -> set[str]:
    with (_REPO_ROOT / "10_interactive_lab" / "manifest.yaml").open(encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    paths: set[str] = set()
    for datasets in (manifest.get("modalities") or {}).values():
        for ds in (datasets or {}).values():
            if not isinstance(ds, dict):
                continue
            for sample in ds.get("samples") or []:
                if isinstance(sample, dict) and sample.get("file"):
                    paths.add(str(sample["file"]))
            for group in ds.get("sample_groups") or []:
                if isinstance(group, dict) and group.get("path"):
                    paths.add(str(group["path"]).rstrip("/") + "/")
    return paths


def _covered(path: str, manifest_paths: set[str]) -> bool:
    if path in manifest_paths:
        return True
    return any(path.startswith(prefix) for prefix in manifest_paths if prefix.endswith("/"))


def test_every_recipe_sample_is_in_manifest_and_on_disk() -> None:
    manifest_paths = _manifest_paths()
    recipes = load_recipes(_REPO_ROOT / "experiments")
    assert recipes, "no recipes loaded"
    for r in recipes:
        referenced = [s.manifest_path for s in r.samples]
        referenced += [s.clean_reference for s in r.samples if s.clean_reference]
        if r.clean_reference:
            referenced.append(r.clean_reference.manifest_path)
        for path in referenced:
            full = _REPO_ROOT / "10_interactive_lab" / path
            assert full.is_file(), f"{r.recipe_id}: sample file missing on disk: {path}"
            assert _covered(path, manifest_paths), (
                f"{r.recipe_id}: {path} not listed in 10_interactive_lab/manifest.yaml"
            )


def test_identity_check_samples_are_their_own_reference() -> None:
    recipes = load_recipes(_REPO_ROOT / "experiments")
    seen = 0
    for r in recipes:
        for s in r.samples:
            if s.role != "identity_check":
                continue
            seen += 1
            ref = s.clean_reference or (
                r.clean_reference.manifest_path if r.clean_reference else ""
            )
            assert ref == s.manifest_path, (
                f"{r.recipe_id}: identity_check sample {s.manifest_path} must "
                f"reference itself, got {ref!r}"
            )
    assert seen >= 3, "expected the three R16 identity_check samples to exist"


def test_per_sample_clean_reference_shapes_compatible() -> None:
    """Per-sample overrides exist to pair each scene with its own ground
    truth — a mismatched shape means the wrong file was wired up."""
    import numpy as np

    recipes = load_recipes(_REPO_ROOT / "experiments")
    checked = 0
    for r in recipes:
        for s in r.samples:
            if not s.clean_reference:
                continue
            sample_arr = np.load(_REPO_ROOT / "10_interactive_lab" / s.manifest_path)
            ref_arr = np.load(_REPO_ROOT / "10_interactive_lab" / s.clean_reference)
            assert sample_arr.shape == ref_arr.shape, (
                f"{r.recipe_id}: {s.manifest_path} {sample_arr.shape} vs "
                f"{s.clean_reference} {ref_arr.shape}"
            )
            checked += 1
    assert checked >= 3, "expected the phase-unwrap per-sample references to exist"
