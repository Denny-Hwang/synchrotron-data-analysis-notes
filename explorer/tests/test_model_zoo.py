"""Tests for the lazy-download zoo (offline-only).

We never hit the network from CI — the tests only verify parsing,
selection, and that ``DownloadError`` is raised under the right
conditions. End-to-end fetches are covered manually via the Streamlit
page.

Ref: ADR-008 — Section 10 Interactive Lab.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

_REPO_ROOT = _EXPLORER_DIR.parent

from lib.model_zoo import (  # noqa: E402
    DownloadError,
    ZooEntry,
    fetch,
    fetch_huggingface,
    find_entry,
    load_zoo,
)


def _write_zoo(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            schema_version: 1

            native_synchrotron_models:
              foo:
                purpose: test
                url: https://example.invalid/foo.h5
                size_bytes: 1024
                license: MIT
                framework: pytorch
              doc_only:
                purpose: documentation only — no url
                license: MIT

            huggingface_baselines:
              swin:
                hf_model_id: caidas/swin2SR-classical-sr-x2-64
                license: Apache-2.0
                framework: pytorch

            external_datasets:
              empiar_xx:
                purpose: cryo-EM benchmark
                url: ftp://ftp.example.invalid/empiar/data.tar
                size_bytes: 2_000_000_000
                license: CC0

            unknown_section:
              ignored:
                url: https://example.invalid/skip.bin
            """
        )
    )


def test_load_zoo_parses_three_sections(tmp_path: Path) -> None:
    p = tmp_path / "zoo.yaml"
    _write_zoo(p)
    entries = load_zoo(p)
    # foo, swin, empiar_xx — doc_only is skipped (no url, no hf_model_id);
    # unknown_section is ignored.
    names = [e.name for e in entries]
    assert names == ["foo", "swin", "empiar_xx"]


def test_zoo_entry_classification(tmp_path: Path) -> None:
    p = tmp_path / "zoo.yaml"
    _write_zoo(p)
    entries = load_zoo(p)

    foo = find_entry(entries, "foo")
    assert foo.is_url and not foo.is_huggingface
    assert foo.section == "native_synchrotron_models"

    swin = find_entry(entries, "swin")
    assert swin.is_huggingface and not swin.is_url

    empiar = find_entry(entries, "empiar_xx")
    assert empiar.is_url
    assert empiar.size_bytes == 2_000_000_000


def test_load_zoo_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_zoo(tmp_path / "nope.yaml")


def test_find_entry_missing_raises(tmp_path: Path) -> None:
    p = tmp_path / "zoo.yaml"
    _write_zoo(p)
    entries = load_zoo(p)
    with pytest.raises(KeyError):
        find_entry(entries, "does_not_exist")


def test_fetch_without_url_raises(tmp_path: Path) -> None:
    """Documentation-only entry should never attempt a download."""
    e = ZooEntry(name="doc", section="external_datasets")
    with pytest.raises(DownloadError, match="no direct URL"):
        fetch(e)


def test_fetch_huggingface_requires_hf_id() -> None:
    e = ZooEntry(name="x", section="native_synchrotron_models", url="https://example.invalid/x")
    with pytest.raises(DownloadError, match="not a Hugging Face entry"):
        fetch_huggingface(e)


def test_bundled_zoo_yaml_loads() -> None:
    """The shipped lazy_download_recipes.yaml must parse cleanly."""
    yaml_path = _REPO_ROOT / "10_interactive_lab" / "models" / "lazy_download_recipes.yaml"
    if not yaml_path.exists():
        pytest.skip("lazy_download_recipes.yaml not present")
    entries = load_zoo(yaml_path)
    assert len(entries) >= 1
    # Sanity: every entry advertises a license string.
    for e in entries:
        assert e.license, f"entry {e.name} has empty license"


def test_bundled_zoo_no_unbalanced_warnings() -> None:
    """CC-BY-NC and GPL entries must carry a license_warning."""
    yaml_path = _REPO_ROOT / "10_interactive_lab" / "models" / "lazy_download_recipes.yaml"
    if not yaml_path.exists():
        pytest.skip("lazy_download_recipes.yaml not present")
    for e in load_zoo(yaml_path):
        license_lower = e.license.lower()
        if "nc" in license_lower or "gpl" in license_lower:
            assert e.license_warning, (
                f"entry '{e.name}' has license '{e.license}' but no license_warning"
            )
