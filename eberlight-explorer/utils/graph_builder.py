"""Build cross-reference graph data for visualization."""

from utils.content_parser import load_yaml


def build_knowledge_graph() -> tuple[list[dict], list[dict]]:
    """Build nodes and edges for the knowledge graph.

    Returns (nodes, edges) where each node has {id, label, group, size}
    and each edge has {source, target, label}.
    """
    modalities = load_yaml("modality_metadata.yaml")["modalities"]
    methods = load_yaml("method_taxonomy.yaml")["categories"]
    papers = load_yaml("publication_catalog.yaml")["publications"]
    tools = load_yaml("tool_catalog.yaml")["tools"]
    cross_refs = load_yaml("cross_references.yaml")

    nodes = []
    edges = []

    # Modality nodes
    for m in modalities:
        nodes.append({
            "id": m["id"],
            "label": m["short_name"],
            "group": "modality",
            "size": 30,
            "color": "#00D4AA",
        })

    # Method category nodes
    for cat in methods:
        nodes.append({
            "id": cat["id"],
            "label": cat["name"],
            "group": "method",
            "size": 25,
            "color": "#FFB800",
        })

    # Paper nodes
    for p in papers:
        nodes.append({
            "id": p["id"],
            "label": f"{p['year']} {p['title'][:30]}...",
            "group": "paper",
            "size": 15,
            "color": "#1B3A5C",
        })

    # Tool nodes
    for t in tools:
        if t["id"] == "aps_github_repos":
            continue
        nodes.append({
            "id": t["id"],
            "label": t["name"],
            "group": "tool",
            "size": 20,
            "color": "#E8515D",
        })

    # Modality -> Method edges
    mod_method = cross_refs.get("modality_method", {})
    for mod_id, method_ids in mod_method.items():
        for method_id in method_ids:
            edges.append({
                "source": mod_id,
                "target": method_id,
                "label": "uses",
            })

    # Method -> Paper edges
    method_paper = cross_refs.get("method_paper", {})
    for method_id, paper_ids in method_paper.items():
        for cat in methods:
            for m in cat["methods"]:
                if m["id"] == method_id or method_id in m["id"]:
                    for pid in paper_ids:
                        edges.append({
                            "source": cat["id"],
                            "target": pid,
                            "label": "reviewed in",
                        })

    # Tool -> Pipeline stage edges
    for t in tools:
        if t.get("pipeline_stage"):
            for mod_id in t.get("modalities", []):
                edges.append({
                    "source": t["id"],
                    "target": mod_id,
                    "label": "supports",
                })

    return nodes, edges
