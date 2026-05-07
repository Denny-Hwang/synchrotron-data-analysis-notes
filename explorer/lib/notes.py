"""Note loader for eBERlight Explorer.

Walks the note folders, parses optional YAML frontmatter, and returns
structured Note objects. Handles graceful degradation for notes without
frontmatter.

Ref: ADR-002 — Notes remain single source of truth.
Ref: ADR-003 — YAML frontmatter schema.
Ref: DC-001 (data_contracts.md) — Schema and controlled vocabularies.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .ia import FOLDER_TO_CLUSTER

logger = logging.getLogger(__name__)

# Controlled vocabularies from DC-001
VALID_CLUSTERS = {"discover", "explore", "build"}
VALID_MODALITIES = {
    "tomography",
    "xrf_microscopy",
    "ptychography",
    "spectroscopy",
    "crystallography",
    "scattering",
    "cross_cutting",
}
VALID_BEAMLINES = {
    "2-BM",
    "2-ID-D",
    "2-ID-E",
    "3-ID",
    "5-BM",
    "7-BM",
    "9-BM",
    "9-ID",
    "11-BM",
    "11-ID-B",
    "11-ID-C",
    "12-ID",
    "17-BM",
    "20-BM",
    "20-ID",
    "26-ID",
    "32-ID",
    "34-ID",
}

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Note:
    """Represents a single note file with parsed metadata."""

    path: Path
    folder: str
    title: str
    body: str
    cluster: str
    tags: list[str] = field(default_factory=list)
    modality: str | None = None
    beamline: list[str] = field(default_factory=list)
    related_publications: list[str] = field(default_factory=list)
    related_tools: list[str] = field(default_factory=list)
    description: str = ""
    category: str = ""
    has_frontmatter: bool = False
    # Optional rich metadata used by the note-detail metric panel.
    # All of these come from frontmatter — Notes that don't declare
    # them simply leave the metric panel hidden.
    resolution: str | None = None
    maturity: str | None = None
    language: str | None = None
    gpu: bool | None = None
    year: int | None = None
    journal: str | None = None
    authors: str | None = None
    doi: str | None = None
    priority: str | None = None
    pipeline_stage: str | None = None
    last_reviewed: str | None = None

    def url_id(self, repo_root: Path) -> str:
        """Stable URL-safe identifier for this note (path relative to repo root).

        Used as the ``?note=...`` query-parameter value so the Streamlit
        cluster pages can deep-link to a specific note. Always uses
        forward slashes to match the static-site mirror's URL scheme.
        """
        return self.path.relative_to(repo_root).as_posix()


def find_note_by_url_id(notes: list[Note], repo_root: Path, url_id: str) -> Note | None:
    """Reverse lookup of :meth:`Note.url_id`. Returns the matching note or None."""
    for n in notes:
        if n.url_id(repo_root) == url_id:
            return n
    return None


def find_note_by_basename(
    notes: list[Note], basename: str, *, folder_hint: str | None = None
) -> Note | None:
    """Resolve a bare filename (with or without ``.md``) to a unique note.

    Restores the legacy ``?doc=ring_artifact`` style deep links that
    ``8_📡_Noise_Catalog.py`` and ``7_📊_Data_Structures.py`` accepted.
    Where multiple notes share the same basename across folders, an
    optional ``folder_hint`` (e.g. ``09_noise_catalog``) disambiguates.

    Returns ``None`` when nothing matches or when the match is ambiguous
    and no folder hint disambiguates it — the caller should then fall
    through to the cluster overview with a warning.
    """
    target = basename.strip()
    if not target:
        return None
    if target.endswith(".md"):
        target = target[: -len(".md")]

    matches = [n for n in notes if n.path.stem == target]
    if folder_hint:
        scoped = [n for n in matches if n.folder == folder_hint]
        if scoped:
            matches = scoped
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_publication_ref(notes: list[Note], ref: str, repo_root: Path) -> Note | None:
    """Resolve a ``related_publications`` entry to the actual Note.

    The frontmatter encodes references as plain filenames (per DC-001),
    e.g. ``review_tomogan_2020.md``. We search for any note under
    ``04_publications/`` whose filename matches; the actual file may
    live one or two folders deep (``04_publications/ai_ml_synchrotron/
    review_tomogan_2020.md``).
    """
    target = ref.strip()
    if not target:
        return None
    if not target.endswith(".md"):
        target += ".md"
    for n in notes:
        if n.folder == "04_publications" and n.path.name == target:
            return n
    return None


def resolve_tool_ref(notes: list[Note], ref: str, repo_root: Path) -> Note | None:
    """Resolve a ``related_tools`` entry to the actual Note.

    Tools live under ``05_tools_and_code/<tool_slug>/`` with a
    ``README.md`` that's the canonical entry. The reference is the
    tool slug — e.g. ``tomocupy`` → ``05_tools_and_code/tomocupy/README.md``.
    """
    slug = ref.strip().lower()
    if not slug:
        return None
    candidate_dir = repo_root / "05_tools_and_code" / slug
    candidate = candidate_dir / "README.md"
    for n in notes:
        if n.path == candidate:
            return n
    # Fall back to any *.md inside the tool's folder if no README.
    for n in notes:
        if n.folder == "05_tools_and_code" and n.path.parent.name.lower() == slug:
            return n
    return None


def neighbor_notes(notes: list[Note], current: Note) -> tuple[Note | None, Note | None]:
    """Return ``(previous, next)`` notes within ``current``'s folder.

    Used by the note-detail view to render "← prev | next →" navigation
    inside a folder. Notes are ordered by their relative path so the
    sequence is stable across runs.
    """
    siblings = sorted(
        (n for n in notes if n.folder == current.folder),
        key=lambda n: n.path.as_posix(),
    )
    try:
        idx = siblings.index(current)
    except ValueError:
        return None, None
    prev_n = siblings[idx - 1] if idx > 0 else None
    next_n = siblings[idx + 1] if idx + 1 < len(siblings) else None
    return prev_n, next_n


def discover_sibling_notebooks(note: Note, repo_root: Path) -> list[Path]:
    """Return ``.ipynb`` files in or near the note's folder.

    Notes about tools / data structures often ship Jupyter notebooks
    in either the same folder as the markdown (
    ``05_tools_and_code/roi_finder/foo.ipynb``) or in a dedicated
    ``notebooks/`` subdirectory (the actual layout in this repo —
    ``05_tools_and_code/roi_finder/notebooks/01_data_loading.ipynb``).
    We surface both so the explorer can link out to nbviewer / GitHub.
    """
    folder = note.path.parent
    if not folder.exists():
        return []
    found = list(folder.glob("*.ipynb"))
    notebooks_subdir = folder / "notebooks"
    if notebooks_subdir.is_dir():
        found.extend(notebooks_subdir.glob("*.ipynb"))
    return sorted(found)


def _title_from_filename(filename: str) -> str:
    """Infer a human-readable title from a filename.

    Args:
        filename: The filename without extension (e.g., 'ai_ml_methods').

    Returns:
        Title-cased string with underscores replaced by spaces.
    """
    name = filename.replace("_", " ").replace("-", " ")
    return name.title()


def _validate_vocabulary(value: str, valid_set: set[str], field_name: str, path: Path) -> bool:
    """Check if a value is in the controlled vocabulary.

    Args:
        value: The value to validate.
        valid_set: Set of allowed values.
        field_name: Name of the field (for logging).
        path: Path to the note file (for logging).

    Returns:
        True if valid, False otherwise.
    """
    if value not in valid_set:
        logger.warning(
            "Invalid %s value '%s' in %s (allowed: %s)",
            field_name,
            value,
            path,
            ", ".join(sorted(valid_set)),
        )
        return False
    return True


def _parse_note(path: Path, folder: str) -> Note:
    """Parse a single note file, extracting frontmatter if present.

    Args:
        path: Path to the markdown file.
        folder: Name of the parent note folder.

    Returns:
        A Note object with parsed or inferred metadata.
    """
    content = path.read_text(encoding="utf-8")

    # Try to extract YAML frontmatter
    fm_match = _FRONTMATTER_RE.match(content)
    if fm_match:
        try:
            fm = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError:
            logger.warning("Invalid YAML frontmatter in %s", path)
            fm = {}
        body = content[fm_match.end() :]
        has_frontmatter = bool(fm)
    else:
        fm = {}
        body = content
        has_frontmatter = False

    # Infer cluster from folder if not in frontmatter
    cluster = fm.get("cluster", FOLDER_TO_CLUSTER.get(folder, "explore"))
    if isinstance(cluster, str):
        _validate_vocabulary(cluster, VALID_CLUSTERS, "cluster", path)

    # Validate modality
    modality = fm.get("modality")
    if modality and isinstance(modality, str):
        _validate_vocabulary(modality, VALID_MODALITIES, "modality", path)

    # Validate beamlines
    beamline_raw = fm.get("beamline", [])
    beamlines = beamline_raw if isinstance(beamline_raw, list) else [beamline_raw]
    for bl in beamlines:
        if isinstance(bl, str):
            _validate_vocabulary(bl, VALID_BEAMLINES, "beamline", path)

    def _opt_str(key: str) -> str | None:
        v = fm.get(key)
        return v.strip() if isinstance(v, str) and v.strip() else None

    def _opt_int(key: str) -> int | None:
        v = fm.get(key)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        return None

    def _opt_bool(key: str) -> bool | None:
        v = fm.get(key)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            lo = v.strip().lower()
            if lo in {"true", "yes", "y", "1"}:
                return True
            if lo in {"false", "no", "n", "0"}:
                return False
        return None

    return Note(
        path=path,
        folder=folder,
        title=fm.get("title", _title_from_filename(path.stem)),
        body=body,
        cluster=cluster if isinstance(cluster, str) else "explore",
        tags=fm.get("tags", []) if isinstance(fm.get("tags"), list) else [],
        modality=modality if isinstance(modality, str) else None,
        beamline=[b for b in beamlines if isinstance(b, str)],
        related_publications=fm.get("related_publications", []) or [],
        related_tools=fm.get("related_tools", []) or [],
        description=fm.get("description", ""),
        category=fm.get("category", ""),
        has_frontmatter=has_frontmatter,
        resolution=_opt_str("resolution"),
        maturity=_opt_str("maturity"),
        language=_opt_str("language"),
        gpu=_opt_bool("gpu"),
        year=_opt_int("year"),
        journal=_opt_str("journal"),
        authors=_opt_str("authors"),
        doi=_opt_str("doi"),
        priority=_opt_str("priority"),
        pipeline_stage=_opt_str("pipeline_stage"),
        last_reviewed=_opt_str("last_reviewed"),
    )


def load_notes(root: Path) -> list[Note]:
    """Load all notes from the note folders.

    Walks each folder in FOLDER_TO_CLUSTER, finds all .md files, and
    parses them into Note objects. Notes without YAML frontmatter load
    with inferred metadata (title from filename, cluster from folder).

    Args:
        root: Path to the repository root.

    Returns:
        List of Note objects sorted by folder then filename.
    """
    notes: list[Note] = []

    for folder in sorted(FOLDER_TO_CLUSTER.keys()):
        folder_path = root / folder
        if not folder_path.is_dir():
            logger.warning("Note folder not found: %s", folder_path)
            continue

        for md_path in sorted(folder_path.rglob("*.md")):
            note = _parse_note(md_path, folder)
            notes.append(note)

    logger.info("Loaded %d notes from %d folders", len(notes), len(FOLDER_TO_CLUSTER))
    return notes
