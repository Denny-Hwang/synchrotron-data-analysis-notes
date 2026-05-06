"""Tests for the cross-reference graph builder.

Closes the audit gap that surfaced after the Phase R0 review: the
Knowledge Graph data layer must be reproducibly extractable from the
repository folder structure plus ``experiments/**/recipe.yaml``,
without depending on YAML catalogs that don't exist in the new
explorer (per ADR-002).

Ref: ADR-002 — Notes are the single source of truth.
Ref: ADR-008 — Section 10 Interactive Lab.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPLORER_DIR.parent

if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from lib.cross_refs import (
    Edge,
    Entity,
    Graph,
    build_graph,
    entity_url,
    iter_kinds,
    kind_color,
    kind_size,
)


@pytest.fixture(scope="module")
def real_graph() -> Graph:
    return build_graph(_REPO_ROOT)


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------


def test_iter_kinds_complete() -> None:
    kinds = list(iter_kinds())
    assert kinds == ["modality", "method", "paper", "tool", "recipe", "noise"]


def test_kind_color_and_size_for_every_kind() -> None:
    for kind in iter_kinds():
        assert kind_color(kind).startswith("#")
        assert kind_size(kind) > 0


# ---------------------------------------------------------------------------
# Real-repo extraction
# ---------------------------------------------------------------------------


def test_graph_has_all_six_kinds(real_graph: Graph) -> None:
    for kind in iter_kinds():
        assert real_graph.by_kind(kind), f"no {kind} entities extracted"


def test_modalities_include_canonical_six(real_graph: Graph) -> None:
    """The six APS X-ray modalities must be detected from `02_xray_modalities/`."""
    labels = {m.label.lower() for m in real_graph.by_kind("modality")}
    expected = {
        "tomography",
        "xrf",
        "ptychography",
        "spectroscopy",
        "scattering",
        "crystallography",
    }
    missing = expected - labels
    assert not missing, f"missing modalities: {missing} (got: {labels})"


def test_section_10_recipes_have_modality_and_noise_edges(real_graph: Graph) -> None:
    """Every bundled recipe must map to its modality AND its noise type."""
    recipe_ids = {e.id for e in real_graph.by_kind("recipe")}
    assert recipe_ids, "no recipes extracted"

    for rid in recipe_ids:
        nbrs = real_graph.neighbours(rid)
        # at least one modality + one noise neighbour
        has_modality = any(n.startswith("modality:") for n in nbrs)
        has_noise = any(n.startswith("noise:") for n in nbrs)
        assert has_modality and has_noise, f"recipe {rid} missing modality/noise edges (got {nbrs})"


def test_no_duplicate_edges(real_graph: Graph) -> None:
    seen = set()
    for ed in real_graph.edges:
        key = (ed.source_id, ed.target_id, ed.kind)
        assert key not in seen, f"duplicate edge: {key}"
        seen.add(key)


def test_every_entity_has_unique_id(real_graph: Graph) -> None:
    ids = [e.id for e in real_graph.entities]
    assert len(ids) == len(set(ids)), "entity ids must be unique"


def test_entity_id_namespacing(real_graph: Graph) -> None:
    """Each entity id is namespaced by its kind."""
    for e in real_graph.entities:
        assert e.id.startswith(f"{e.kind}:"), f"id {e.id!r} not namespaced for kind {e.kind!r}"


# ---------------------------------------------------------------------------
# Edge endpoints
# ---------------------------------------------------------------------------


def test_every_edge_endpoint_resolves(real_graph: Graph) -> None:
    ids = {e.id for e in real_graph.entities}
    for ed in real_graph.edges:
        assert ed.source_id in ids, f"dangling edge source: {ed}"
        assert ed.target_id in ids, f"dangling edge target: {ed}"


def test_modality_to_noise_edges_exist(real_graph: Graph) -> None:
    """09_noise_catalog/<modality>/* should produce modality→noise edges."""
    count = sum(
        1
        for ed in real_graph.edges
        if ed.source_id.startswith("modality:") and ed.target_id.startswith("noise:")
    )
    # At least one noise per known modality.
    assert count >= len(real_graph.by_kind("modality")) - 2, (
        f"expected modality→noise edges, found only {count}"
    )


def test_recipe_to_noise_edges_exist(real_graph: Graph) -> None:
    """The 3 bundled recipes should each link to at least one noise type."""
    recipe_count = len(real_graph.by_kind("recipe"))
    edge_count = sum(
        1
        for ed in real_graph.edges
        if ed.source_id.startswith("recipe:") and ed.target_id.startswith("noise:")
    )
    assert edge_count >= recipe_count, (
        f"expected ≥{recipe_count} recipe→noise edges, got {edge_count}"
    )


# ---------------------------------------------------------------------------
# Entity URL helpers
# ---------------------------------------------------------------------------


def test_entity_url_recipe_points_to_experiment_page() -> None:
    e = Entity(id="recipe:foo", kind="recipe", label="Foo")
    assert entity_url(e) == "/Experiment"


def test_entity_url_with_doc_path_uses_query_param() -> None:
    e = Entity(
        id="noise:tomography/ring_artifact",
        kind="noise",
        label="Ring Artifact",
        doc_path="09_noise_catalog/tomography/ring_artifact.md",
    )
    url = entity_url(e)
    assert url.startswith("?note=")
    assert "ring_artifact" in url


def test_entity_url_no_doc_path_returns_hash() -> None:
    e = Entity(id="modality:phantom", kind="modality", label="Phantom")
    assert entity_url(e) == "#"


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def test_graph_neighbours_symmetric() -> None:
    g = Graph(
        entities=[
            Entity(id="a", kind="modality", label="A"),
            Entity(id="b", kind="noise", label="B"),
        ],
        edges=[Edge(source_id="a", target_id="b", kind="suffers")],
    )
    assert g.neighbours("a") == ["b"]
    assert g.neighbours("b") == ["a"]


def test_graph_by_kind_filters_correctly() -> None:
    g = Graph(
        entities=[
            Entity(id="a", kind="modality", label="A"),
            Entity(id="b", kind="modality", label="B"),
            Entity(id="c", kind="recipe", label="C"),
        ]
    )
    assert {e.id for e in g.by_kind("modality")} == {"a", "b"}
    assert {e.id for e in g.by_kind("recipe")} == {"c"}
    assert g.by_kind("nonexistent") == []
