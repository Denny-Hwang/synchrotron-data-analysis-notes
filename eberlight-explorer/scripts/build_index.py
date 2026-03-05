#!/usr/bin/env python3
"""Scan the synchrotron-data-analysis-notes repo and validate/rebuild YAML indexes."""

import os
import sys
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def validate_file_references():
    """Check that all files referenced in YAML catalogs actually exist."""
    errors = []

    yaml_files = [
        "content_index.yaml",
        "modality_metadata.yaml",
        "method_taxonomy.yaml",
        "publication_catalog.yaml",
        "tool_catalog.yaml",
        "cross_references.yaml",
    ]

    for yf in yaml_files:
        filepath = os.path.join(DATA_DIR, yf)
        if not os.path.exists(filepath):
            errors.append(f"Missing YAML file: {yf}")
            continue

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        paths = _extract_paths(data)
        for p in paths:
            full_path = os.path.join(REPO_ROOT, p)
            if not os.path.exists(full_path):
                errors.append(f"[{yf}] Missing file: {p}")

    return errors


def _extract_paths(obj, paths=None):
    """Recursively extract file path strings from a YAML structure."""
    if paths is None:
        paths = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in ("path", "file", "readme", "data_format", "ai_ml",
                       "architecture", "pros_cons", "reproduction",
                       "reverse_eng", "workflow", "catalog"):
                if isinstance(val, str) and "/" in val:
                    paths.append(val)
            else:
                _extract_paths(val, paths)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str) and "/" in item and item.endswith((".md", ".ipynb", ".bib")):
                paths.append(item)
            else:
                _extract_paths(item, paths)
    return paths


def count_repo_files():
    """Count total markdown and notebook files in the repo."""
    counts = {"md": 0, "ipynb": 0, "bib": 0}
    for root, dirs, files in os.walk(REPO_ROOT):
        if ".git" in root or "eberlight-explorer" in root:
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lstrip(".")
            if ext in counts:
                counts[ext] += 1
    return counts


if __name__ == "__main__":
    print("=" * 60)
    print("eBERlight Explorer - Index Validator")
    print("=" * 60)

    counts = count_repo_files()
    print(f"\nRepo file counts: {counts}")

    errors = validate_file_references()
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n✅ All file references are valid!")
        sys.exit(0)
