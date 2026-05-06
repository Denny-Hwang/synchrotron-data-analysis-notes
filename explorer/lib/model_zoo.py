"""Lazy-download utility for the Interactive Lab — model weights and external datasets.

Reads ``10_interactive_lab/models/lazy_download_recipes.yaml`` and exposes
helpers that fetch entries on demand, cache them under the OS cache dir,
and verify integrity via SHA-256.

Design contract (per ADR-008):

1. **No bundled weights.** The repository ships nothing that could
   conflict with a license (e.g., CC-BY-NC TomoGAN, GPL-3.0 Topaz).
2. **Show the license before downloading.** A ``ZooEntry.license_warning``
   string is surfaced to the caller (and rendered by the Streamlit page)
   so the user has a chance to opt out.
3. **Hash-pinned.** Once a file is vetted, its SHA-256 is recorded in the
   YAML; subsequent runs verify the cache and re-download on mismatch.
4. **Best-effort.** When the network is unavailable, the helpers raise a
   :class:`DownloadError` with a clear message; pages should fall back to
   an "open the external link" UI.

Pure-Python — no Streamlit imports, no global state. The Streamlit
page wraps the helpers with ``@st.cache_data`` and progress UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """Raised when a lazy download cannot be satisfied."""


@dataclass(frozen=True)
class ZooEntry:
    """One row in ``lazy_download_recipes.yaml``."""

    name: str
    section: str  # e.g. "native_synchrotron_models", "huggingface_baselines", "external_datasets"
    purpose: str = ""
    url: str | None = None
    hf_model_id: str | None = None
    known_hash: str | None = None  # "sha256:..." form
    size_bytes: int | None = None
    license: str = "unknown"
    license_warning: str = ""
    framework: str = ""
    paper_doi: str = ""
    repo: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_huggingface(self) -> bool:
        return self.hf_model_id is not None

    @property
    def is_url(self) -> bool:
        return self.url is not None and self.url.startswith(("http://", "https://", "ftp://"))


# ---------------------------------------------------------------------------
# Registry parsing
# ---------------------------------------------------------------------------


_TOP_LEVEL_SECTIONS = (
    "native_synchrotron_models",
    "huggingface_baselines",
    "external_datasets",
)


def load_zoo(yaml_path: Path) -> list[ZooEntry]:
    """Parse ``lazy_download_recipes.yaml`` into a flat list of :class:`ZooEntry`.

    Unknown top-level keys are ignored. Entries with no ``url`` and no
    ``hf_model_id`` are skipped (they are documentation-only).
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"zoo manifest not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    entries: list[ZooEntry] = []
    for section in _TOP_LEVEL_SECTIONS:
        bucket = data.get(section) or {}
        if not isinstance(bucket, dict):
            continue
        for name, fields in bucket.items():
            if not isinstance(fields, dict):
                continue
            url = fields.get("url")
            hf_id = fields.get("hf_model_id")
            if url is None and hf_id is None:
                logger.debug("Skipping %s (no url, no hf_model_id)", name)
                continue
            entries.append(
                ZooEntry(
                    name=str(name),
                    section=section,
                    purpose=str(fields.get("purpose", "")),
                    url=str(url) if url is not None else None,
                    hf_model_id=str(hf_id) if hf_id is not None else None,
                    known_hash=fields.get("known_hash") or None,
                    size_bytes=fields.get("size_bytes"),
                    license=str(fields.get("license", "unknown")),
                    license_warning=str(fields.get("license_warning", "")),
                    framework=str(fields.get("framework", "")),
                    paper_doi=str(fields.get("paper_doi", "")),
                    repo=str(fields.get("repo", "")),
                    extra={
                        k: v
                        for k, v in fields.items()
                        if k
                        not in {
                            "purpose",
                            "url",
                            "hf_model_id",
                            "known_hash",
                            "size_bytes",
                            "license",
                            "license_warning",
                            "framework",
                            "paper_doi",
                            "repo",
                        }
                    },
                )
            )
    logger.info("Loaded %d zoo entries from %s", len(entries), yaml_path.name)
    return entries


def find_entry(entries: list[ZooEntry], name: str) -> ZooEntry:
    """Return the entry with the given name, or raise :class:`KeyError`."""
    for e in entries:
        if e.name == name:
            return e
    raise KeyError(f"zoo entry '{name}' not found")


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


def cache_dir() -> Path:
    """Return the OS-appropriate cache directory used by ``pooch``."""
    import pooch

    return Path(pooch.os_cache("eberlight_lab"))


def is_cached(entry: ZooEntry) -> bool:
    """Return ``True`` if the entry already lives in the cache (best-effort)."""
    if not entry.is_url:
        return False
    candidate = cache_dir() / Path(entry.url).name  # type: ignore[arg-type]
    return candidate.exists()


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch(entry: ZooEntry, progressbar: bool = False) -> Path:
    """Fetch ``entry`` from its declared URL into the local cache.

    Args:
        entry: A :class:`ZooEntry` with ``url`` populated.
        progressbar: If ``True`` and ``pooch`` is available, a textual
            progress bar is shown.

    Returns:
        Path to the cached file.

    Raises:
        DownloadError: If the entry has no URL, the network fails, or the
            hash mismatches.
    """
    if not entry.is_url:
        raise DownloadError(
            f"Entry '{entry.name}' has no direct URL. "
            f"Section '{entry.section}'. See repo {entry.repo or '(no repo)'}"
        )

    try:
        import pooch
    except ImportError as e:  # pragma: no cover
        raise DownloadError("pooch is not installed; cannot lazy-download") from e

    try:
        path_str = pooch.retrieve(
            url=entry.url,  # type: ignore[arg-type]
            known_hash=entry.known_hash,  # may be None on first vet
            path=cache_dir(),
            progressbar=progressbar,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        # OSError covers network / disk / SSL / file I/O failures;
        # ValueError covers pooch hash-mismatch; RuntimeError covers
        # pooch's wrapper exceptions for missing optional deps.
        # KeyboardInterrupt / SystemExit / MemoryError propagate.
        raise DownloadError(
            f"Failed to fetch '{entry.name}' from {entry.url}: {exc}"
        ) from exc

    return Path(path_str)


def fetch_huggingface(entry: ZooEntry) -> Path:
    """Fetch a Hugging Face model snapshot.

    Uses ``huggingface_hub`` when present. Returns the local snapshot dir.

    Raises:
        DownloadError: If the entry is not an HF entry or hub is unavailable.
    """
    if not entry.is_huggingface:
        raise DownloadError(
            f"Entry '{entry.name}' is not a Hugging Face entry "
            f"(no hf_model_id)"
        )
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as e:
        raise DownloadError(
            "huggingface_hub not installed; pip install huggingface-hub"
        ) from e

    try:
        path_str = snapshot_download(
            repo_id=entry.hf_model_id,  # type: ignore[arg-type]
            cache_dir=cache_dir() / "hf",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        # As above — KeyboardInterrupt / SystemExit / MemoryError
        # propagate. huggingface_hub raises a `RepositoryNotFoundError`
        # which is a subclass of `OSError` (so this catches it cleanly).
        raise DownloadError(
            f"Failed to fetch HF model '{entry.hf_model_id}': {exc}"
        ) from exc
    return Path(path_str)
