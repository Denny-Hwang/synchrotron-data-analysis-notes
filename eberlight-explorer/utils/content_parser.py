"""Content parsing utilities for markdown, YAML, and BibTeX files."""

import os
import re
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_yaml(filename: str) -> dict:
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    filepath = os.path.join(data_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_local_file(relative_path: str) -> str | None:
    filepath = os.path.join(REPO_ROOT, relative_path)
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def extract_title(markdown_text: str) -> str:
    for line in markdown_text.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


def extract_metadata_table(markdown_text: str) -> dict:
    """Extract key-value pairs from the first markdown table (Metadata section)."""
    metadata = {}
    in_table = False
    for line in markdown_text.split("\n"):
        line = line.strip()
        if line.startswith("|") and "---" not in line:
            parts = [p.strip() for p in line.split("|")]
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                key = re.sub(r"\*\*", "", parts[0]).strip()
                val = re.sub(r"\*\*", "", parts[1]).strip()
                if key and val and key.lower() not in ("field", "item"):
                    metadata[key] = val
                    in_table = True
        elif in_table and not line.startswith("|"):
            break
    return metadata


def extract_section(markdown_text: str, section_name: str) -> str | None:
    """Extract content under a specific ## heading."""
    lines = markdown_text.split("\n")
    capturing = False
    result = []
    for line in lines:
        if line.strip().startswith("## ") and section_name.lower() in line.lower():
            capturing = True
            continue
        elif line.strip().startswith("## ") and capturing:
            break
        elif capturing:
            result.append(line)
    if result:
        return "\n".join(result).strip()
    return None


def extract_tldr(markdown_text: str) -> str | None:
    return extract_section(markdown_text, "TL;DR")


def parse_bibtex(bibtex_text: str) -> list[dict]:
    """Simple BibTeX parser returning list of entries."""
    entries = []
    pattern = re.compile(
        r"@(\w+)\{([^,]+),\s*(.*?)\}\s*(?=@|\Z)", re.DOTALL
    )
    for match in pattern.finditer(bibtex_text):
        entry_type = match.group(1)
        entry_key = match.group(2).strip()
        body = match.group(3)
        fields = {}
        field_pattern = re.compile(r"(\w+)\s*=\s*\{([^}]*)\}")
        for fm in field_pattern.finditer(body):
            fields[fm.group(1).lower()] = fm.group(2).strip()
        entries.append({
            "type": entry_type,
            "key": entry_key,
            **fields,
        })
    return entries
