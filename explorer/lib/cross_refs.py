"""Cross-reference graph extraction for the Knowledge Graph page.

Builds a typed graph (modalities ↔ methods ↔ papers ↔ tools ↔ recipes
↔ noise types) from the repository's folder structure plus the
recipe YAML files. The legacy ``eberlight-explorer/`` shipped five
hand-curated YAML catalogs to power the equivalent view; ADR-002
chose to keep the notes as the single source of truth, so we extract
the relationships at runtime instead of duplicating them.

Sources of edges (in order of reliability):

1. **Folder structure** — always available:
   - ``02_xray_modalities/<modality>/`` → modality nodes
   - ``03_ai_ml_methods/<category>/<method>.md`` → method nodes
     (category derived from the parent folder)
   - ``04_publications/<paper>.md`` → paper nodes
   - ``05_tools_and_code/<tool>/`` → tool nodes
   - ``09_noise_catalog/<modality>/<noise>.md`` → noise nodes
     (modality derived from the parent folder; the cross-cutting,
     medical_imaging, electron_microscopy and scattering_diffraction
     subfolders are categorised as "cross-domain")
2. **Recipe YAML** — ``experiments/**/recipe.yaml`` provides
   ``modality`` and ``noise_catalog_ref`` fields, giving the most
   precise edges for section 10 / Interactive Lab content (ADR-008).
3. **Content scanning** — best-effort. We scan paper review markdown
   for tool names and method names; matches generate paper→tool and
   paper→method edges. Misses are accepted; this is a navigability
   aid, not a citation database.

Pure data layer — no Streamlit, no network. Designed to be imported
by the Knowledge Graph page and unit-tested in isolation.

Ref: ADR-002 — Notes are the single source of truth.
Ref: ADR-008 — Section 10 Interactive Lab.
Ref: FR-007 — Cross-reference matrices.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


_ENTITY_KINDS = ("modality", "method", "paper", "tool", "recipe", "noise")


@dataclass(frozen=True)
class Entity:
    """One node in the cross-reference graph."""

    id: str
    kind: str  # one of _ENTITY_KINDS
    label: str
    doc_path: str | None = None  # path relative to repo root
    category: str | None = None  # method category, noise modality, etc.


@dataclass(frozen=True)
class Edge:
    """One directed edge between two entities."""

    source_id: str
    target_id: str
    kind: str = "related"  # "in_modality" | "applies_to" | "mitigates" | …


@dataclass
class Graph:
    """Aggregated cross-reference graph."""

    entities: list[Entity] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[Entity]:
        return [e for e in self.entities if e.kind == kind]

    def neighbours(self, entity_id: str) -> list[str]:
        """Return ids of entities adjacent to ``entity_id`` (either direction)."""
        out: set[str] = set()
        for ed in self.edges:
            if ed.source_id == entity_id:
                out.add(ed.target_id)
            elif ed.target_id == entity_id:
                out.add(ed.source_id)
        return sorted(out)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


# Maps the noise-catalog "modality folder" to the canonical modality
# label used elsewhere in the graph. Cross-domain folders surface as
# their own pseudo-modality nodes so the graph stays connected.
_NOISE_FOLDER_TO_MODALITY = {
    "tomography": "tomography",
    "xrf_microscopy": "xrf_microscopy",
    "ptychography": "ptychography",
    "spectroscopy": "spectroscopy",
    "scattering_diffraction": "scattering",
    "medical_imaging": "medical_imaging",
    "electron_microscopy": "electron_microscopy",
    "cross_cutting": "cross_cutting",
}

_PRETTY = {
    "xrf_microscopy": "XRF",
    "tomography": "Tomography",
    "ptychography": "Ptychography",
    "spectroscopy": "Spectroscopy",
    "scattering": "Scattering",
    "scattering_diffraction": "Scattering / Diffraction",
    "crystallography": "Crystallography",
    "medical_imaging": "Medical Imaging",
    "electron_microscopy": "Electron Microscopy",
    "cross_cutting": "Cross-cutting",
}


def _humanise(slug: str) -> str:
    return _PRETTY.get(slug, slug.replace("_", " ").replace("-", " ").title())


def _modality_id(slug: str) -> str:
    return f"modality:{slug}"


def _method_id(category: str, name: str) -> str:
    return f"method:{category}/{name}"


def _paper_id(stem: str) -> str:
    return f"paper:{stem}"


def _tool_id(slug: str) -> str:
    return f"tool:{slug}"


def _recipe_id(rid: str) -> str:
    return f"recipe:{rid}"


def _noise_id(modality: str, name: str) -> str:
    return f"noise:{modality}/{name}"


# ---------------------------------------------------------------------------
# Entity collectors
# ---------------------------------------------------------------------------


def _collect_modalities(repo_root: Path) -> list[Entity]:
    """6 standard modalities + any cross-domain pseudo-modalities seen in 09_*."""
    base = repo_root / "02_xray_modalities"
    real: list[Entity] = []
    if base.is_dir():
        for child in sorted(p for p in base.iterdir() if p.is_dir()):
            real.append(
                Entity(
                    id=_modality_id(child.name),
                    kind="modality",
                    label=_humanise(child.name),
                    doc_path=f"02_xray_modalities/{child.name}/README.md",
                )
            )

    # Cross-domain pseudo-modalities only created if the noise catalog
    # actually has a folder for them — keeps the graph honest.
    seen = {e.id for e in real}
    noise_root = repo_root / "09_noise_catalog"
    if noise_root.is_dir():
        for child in sorted(p for p in noise_root.iterdir() if p.is_dir()):
            if child.name not in _NOISE_FOLDER_TO_MODALITY:
                continue
            modality_slug = _NOISE_FOLDER_TO_MODALITY[child.name]
            mid = _modality_id(modality_slug)
            if mid in seen:
                continue
            real.append(
                Entity(
                    id=mid,
                    kind="modality",
                    label=_humanise(modality_slug),
                    doc_path=None,
                )
            )
            seen.add(mid)
    return real


def _collect_methods(repo_root: Path) -> list[Entity]:
    base = repo_root / "03_ai_ml_methods"
    out: list[Entity] = []
    if not base.is_dir():
        return out
    for category_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for md_file in sorted(category_dir.glob("*.md")):
            if md_file.name.upper() == "README.MD":
                continue
            out.append(
                Entity(
                    id=_method_id(category_dir.name, md_file.stem),
                    kind="method",
                    label=_humanise(md_file.stem),
                    doc_path=f"03_ai_ml_methods/{category_dir.name}/{md_file.name}",
                    category=category_dir.name,
                )
            )
    return out


def _collect_papers(repo_root: Path) -> list[Entity]:
    base = repo_root / "04_publications"
    out: list[Entity] = []
    if not base.is_dir():
        return out
    for md_file in sorted(base.rglob("*.md")):
        stem = md_file.stem.lower()
        if stem in {"readme", "template_paper_review", "ber_program_publications"}:
            continue
        out.append(
            Entity(
                id=_paper_id(md_file.stem),
                kind="paper",
                label=_humanise(md_file.stem),
                doc_path=str(md_file.relative_to(repo_root).as_posix()),
            )
        )
    return out


def _collect_tools(repo_root: Path) -> list[Entity]:
    base = repo_root / "05_tools_and_code"
    out: list[Entity] = []
    if not base.is_dir():
        return out
    for child in sorted(p for p in base.iterdir() if p.is_dir()):
        if child.name == "aps_github_repos":
            # Meta-entry in the legacy catalog; not a real tool.
            continue
        readme = child / "README.md"
        out.append(
            Entity(
                id=_tool_id(child.name),
                kind="tool",
                label=_humanise(child.name),
                doc_path=f"05_tools_and_code/{child.name}/README.md" if readme.exists() else None,
            )
        )
    return out


def _collect_noise(repo_root: Path) -> list[Entity]:
    base = repo_root / "09_noise_catalog"
    out: list[Entity] = []
    if not base.is_dir():
        return out
    for modality_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if modality_dir.name not in _NOISE_FOLDER_TO_MODALITY:
            continue
        modality_slug = _NOISE_FOLDER_TO_MODALITY[modality_dir.name]
        for md_file in sorted(modality_dir.glob("*.md")):
            if md_file.name.upper() in {"README.MD", "INDEX.MD"}:
                continue
            out.append(
                Entity(
                    id=_noise_id(modality_slug, md_file.stem),
                    kind="noise",
                    label=_humanise(md_file.stem),
                    doc_path=f"09_noise_catalog/{modality_dir.name}/{md_file.name}",
                    category=modality_slug,
                )
            )
    return out


def _collect_recipes(repo_root: Path) -> tuple[list[Entity], list[dict]]:
    """Return (entities, raw recipes) so edge extraction can reuse the parsed yaml."""
    base = repo_root / "experiments"
    entities: list[Entity] = []
    raw: list[dict] = []
    if not base.is_dir():
        return entities, raw
    for recipe_path in sorted(base.rglob("recipe.yaml")):
        try:
            with recipe_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError) as exc:
            logger.warning("Skipping recipe %s: %s", recipe_path, exc)
            continue
        rid = str(data.get("recipe_id", recipe_path.parent.name))
        title = str(data.get("title", _humanise(rid)))
        entities.append(
            Entity(
                id=_recipe_id(rid),
                kind="recipe",
                label=title,
                doc_path=str(recipe_path.relative_to(repo_root).as_posix()),
                category=str(data.get("modality", "")) or None,
            )
        )
        raw.append({**data, "_path": recipe_path.relative_to(repo_root).as_posix()})
    return entities, raw


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------


_TOOL_NAME_PATTERNS: dict[str, re.Pattern[str]] = {
    "tomopy": re.compile(r"\btomopy\b", re.I),
    "tomocupy": re.compile(r"\btomocupy\b", re.I),
    "tike": re.compile(r"\btike\b", re.I),
    "httomo": re.compile(r"\bhttomo\b", re.I),
    "maps_software": re.compile(r"\bmaps\b(?!\w)", re.I),
    "mlexchange": re.compile(r"\bmlexchange\b", re.I),
    "pyxrf": re.compile(r"\bpyxrf\b", re.I),
    "bluesky_epics": re.compile(r"\bbluesky\b|\bepics\b", re.I),
    "roi_finder": re.compile(r"\broi[\s_-]?finder\b", re.I),
}


def _extract_paper_edges(papers: list[Entity], repo_root: Path) -> list[Edge]:
    """Best-effort tool/method mentions inside paper review markdown."""
    edges: list[Edge] = []
    for paper in papers:
        if paper.doc_path is None:
            continue
        try:
            text = (repo_root / paper.doc_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for tool_slug, pattern in _TOOL_NAME_PATTERNS.items():
            if pattern.search(text):
                edges.append(Edge(paper.id, _tool_id(tool_slug), kind="mentions"))
    return edges


def _extract_recipe_edges(recipes_raw: list[dict]) -> list[Edge]:
    """Edges encoded in each recipe.yaml: recipe→modality and recipe→noise."""
    edges: list[Edge] = []
    for r in recipes_raw:
        rid = _recipe_id(str(r.get("recipe_id", "")))
        modality = str(r.get("modality", "")).strip()
        if modality:
            edges.append(Edge(rid, _modality_id(modality), kind="targets"))
        noise_ref = str(r.get("noise_catalog_ref", "")).strip()
        # noise_catalog_ref looks like '09_noise_catalog/<modality>/<noise>.md'
        if noise_ref.startswith("09_noise_catalog/"):
            parts = noise_ref.split("/")
            if len(parts) >= 3:
                noise_modality_folder = parts[1]
                noise_stem = Path(parts[2]).stem
                if noise_modality_folder in _NOISE_FOLDER_TO_MODALITY:
                    edges.append(
                        Edge(
                            rid,
                            _noise_id(
                                _NOISE_FOLDER_TO_MODALITY[noise_modality_folder],
                                noise_stem,
                            ),
                            kind="mitigates",
                        )
                    )
    return edges


def _extract_modality_noise_edges(noises: list[Entity]) -> list[Edge]:
    out: list[Edge] = []
    for n in noises:
        if n.category:
            out.append(Edge(_modality_id(n.category), n.id, kind="suffers"))
    return out


def _extract_method_modality_edges(
    methods: list[Entity], modality_ids: set[str], repo_root: Path
) -> list[Edge]:
    """Best-effort: scan each method file body for the canonical modality
    slugs / labels. Misses default to "applies_to" with no edge.
    """
    out: list[Edge] = []
    for m in methods:
        if m.doc_path is None:
            continue
        try:
            text = (repo_root / m.doc_path).read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for mid in modality_ids:
            slug = mid.split(":", 1)[1]
            label = _humanise(slug).lower()
            if slug.replace("_", " ") in text or label in text:
                out.append(Edge(m.id, mid, kind="applies_to"))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_graph(repo_root: Path) -> Graph:
    """Build the full cross-reference graph from the repository state."""
    modalities = _collect_modalities(repo_root)
    methods = _collect_methods(repo_root)
    papers = _collect_papers(repo_root)
    tools = _collect_tools(repo_root)
    noises = _collect_noise(repo_root)
    recipes, recipes_raw = _collect_recipes(repo_root)

    entities: list[Entity] = [*modalities, *methods, *papers, *tools, *noises, *recipes]

    edges: list[Edge] = []
    edges.extend(_extract_modality_noise_edges(noises))
    edges.extend(_extract_recipe_edges(recipes_raw))
    edges.extend(_extract_paper_edges(papers, repo_root))
    edges.extend(_extract_method_modality_edges(methods, {m.id for m in modalities}, repo_root))

    # Dedupe edges (the various extractors can overlap slightly).
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Edge] = []
    for ed in edges:
        key = (ed.source_id, ed.target_id, ed.kind)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ed)

    return Graph(entities=entities, edges=deduped)


def entity_url(entity: Entity, repo_root: Path | None = None) -> str:
    """Return the Streamlit deep-link for an entity, or ``"#"`` if no doc.

    Modality / method / paper / tool / noise → ``?note=<doc_path>``.
    Recipe → ``/Experiment`` (the lab page; the recipe selector lives there).
    """
    if entity.kind == "recipe":
        return "/Experiment"
    if entity.doc_path:
        from urllib.parse import quote

        return f"?note={quote(entity.doc_path, safe='/')}"
    return "#"


def kind_color(kind: str) -> str:
    """Stable color per entity kind for the network plot."""
    return {
        "modality": "#0033A0",  # ANL blue
        "method": "#F47B20",  # build-orange
        "paper": "#9B59B6",  # purple
        "tool": "#27AE60",  # green
        "recipe": "#E8515D",  # red — section 10 highlight
        "noise": "#7F8C8D",  # gray
    }.get(kind, "#555555")


def kind_size(kind: str) -> int:
    """Stable node size per entity kind."""
    return {
        "modality": 32,
        "tool": 22,
        "method": 18,
        "recipe": 22,
        "paper": 14,
        "noise": 12,
    }.get(kind, 16)


def iter_kinds() -> Iterable[str]:
    """Public iterator over the canonical kind list."""
    return _ENTITY_KINDS
