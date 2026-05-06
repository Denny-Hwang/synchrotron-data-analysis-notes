"""Troubleshooter data layer — symptom-based noise/artifact diagnosis.

Loads ``09_noise_catalog/troubleshooter.yaml`` (the machine-readable
companion to the prose ``troubleshooter.md``) and exposes a small
typed surface that the Streamlit page (`5_Troubleshooter.py`) and the
unit tests both consume.

The 11 symptoms each carry a flat list of "cases" — a case is one
differential diagnosis, gated by an ordered sequence of
``conditions`` the user can match against their own data, plus a
``Diagnosis`` (name, severity, link to the full guide, and optional
recipe / before-after image references).

Severity vocabulary: ``critical``, ``major``, ``minor`` (matches
``summary_table.md``). Anything outside that set is logged and
treated as ``minor`` for visual styling.

Pure data layer — no Streamlit, no I/O at import time.

Ref: ADR-002 — Notes are the single source of truth (the prose
    troubleshooter.md remains canonical; the YAML is its structured
    companion).
Ref: ADR-008 — Section 10 Interactive Lab integration via the
    optional ``recipe`` field on each diagnosis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


VALID_SEVERITIES = ("critical", "major", "minor")


@dataclass(frozen=True)
class QuickCheck:
    """One Python diagnostic snippet attached to a symptom."""

    title: str
    language: str
    code: str


@dataclass(frozen=True)
class Diagnosis:
    """Terminal diagnosis at the end of a case's condition sequence."""

    name: str
    severity: str
    guide: str  # path relative to 09_noise_catalog/
    recipe: str | None = None  # recipe_id matching experiments/**/recipe.yaml
    image: str | None = None  # filename in 09_noise_catalog/images/


@dataclass(frozen=True)
class Case:
    """One differential diagnosis under a symptom."""

    conditions: tuple[str, ...]
    diagnosis: Diagnosis


@dataclass(frozen=True)
class Symptom:
    """One of the 11 top-level symptom categories."""

    id: str
    title: str
    summary: str
    quick_checks: tuple[QuickCheck, ...]
    cases: tuple[Case, ...]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _coerce_severity(raw: str, *, source: str) -> str:
    s = (raw or "minor").strip().lower()
    if s not in VALID_SEVERITIES:
        logger.warning(
            "Unknown severity %r in %s; defaulting to 'minor' for styling",
            raw,
            source,
        )
        return "minor"
    return s


def _parse_quick_check(d: dict) -> QuickCheck:
    return QuickCheck(
        title=str(d.get("title", "Quick check")),
        language=str(d.get("language", "python")),
        code=str(d.get("code", "")).rstrip(),
    )


def _parse_diagnosis(d: dict, *, source: str) -> Diagnosis:
    if "name" not in d:
        raise ValueError(f"diagnosis in {source} missing 'name'")
    return Diagnosis(
        name=str(d["name"]),
        severity=_coerce_severity(str(d.get("severity", "minor")), source=source),
        guide=str(d.get("guide", "")).strip(),
        recipe=str(d["recipe"]).strip() if d.get("recipe") else None,
        image=str(d["image"]).strip() if d.get("image") else None,
    )


def _parse_case(d: dict, *, source: str) -> Case:
    if "diagnosis" not in d:
        raise ValueError(f"case in {source} missing 'diagnosis'")
    conds = d.get("conditions") or []
    if not isinstance(conds, list):
        raise ValueError(f"case in {source}: 'conditions' must be a list")
    return Case(
        conditions=tuple(str(c) for c in conds),
        diagnosis=_parse_diagnosis(d["diagnosis"], source=source),
    )


def _parse_symptom(d: dict) -> Symptom:
    if "id" not in d or "title" not in d:
        raise ValueError(f"symptom missing required field: {d}")
    sid = str(d["id"])
    return Symptom(
        id=sid,
        title=str(d["title"]),
        summary=str(d.get("summary", "")).strip(),
        quick_checks=tuple(_parse_quick_check(qc) for qc in (d.get("quick_checks") or [])),
        cases=tuple(_parse_case(c, source=f"symptom {sid!r}") for c in (d.get("cases") or [])),
    )


@dataclass(frozen=True)
class Troubleshooter:
    """Aggregated view of all 11 symptoms loaded from the YAML."""

    symptoms: tuple[Symptom, ...]

    def by_id(self, symptom_id: str) -> Symptom | None:
        for s in self.symptoms:
            if s.id == symptom_id:
                return s
        return None

    def all_diagnoses(self) -> list[Diagnosis]:
        return [c.diagnosis for s in self.symptoms for c in s.cases]


def load_troubleshooter(repo_root: Path) -> Troubleshooter:
    """Load and parse ``09_noise_catalog/troubleshooter.yaml``.

    Bad symptoms / cases are logged and skipped so a single typo
    doesn't take down the whole page.
    """
    yaml_path = repo_root / "09_noise_catalog" / "troubleshooter.yaml"
    if not yaml_path.exists():
        logger.warning("troubleshooter.yaml not present at %s", yaml_path)
        return Troubleshooter(symptoms=())

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_symptoms = data.get("symptoms") or []
    parsed: list[Symptom] = []
    for raw in raw_symptoms:
        try:
            parsed.append(_parse_symptom(raw))
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping invalid symptom: %s", exc)
    return Troubleshooter(symptoms=tuple(parsed))


# ---------------------------------------------------------------------------
# Before/after image discovery
# ---------------------------------------------------------------------------


_IMAGE_DIR = Path("09_noise_catalog/images")


def list_before_after_images(repo_root: Path) -> dict[str, Path]:
    """Map ``"<stem>"`` (without ``_before_after.png`` suffix) → absolute path.

    e.g. ``ring_artifact_before_after.png`` → key ``ring_artifact``.
    Missing images are silently ignored — the troubleshooter page
    falls back to "no image available" text.
    """
    out: dict[str, Path] = {}
    folder = repo_root / _IMAGE_DIR
    if not folder.is_dir():
        return out
    for img in sorted(folder.glob("*_before_after.png")):
        stem = img.stem.removesuffix("_before_after")
        out[stem] = img
    return out


def severity_color(severity: str) -> str:
    """Stable CSS hex per severity for badges."""
    return {
        "critical": "#C0392B",
        "major": "#E67E22",
        "minor": "#3498DB",
    }.get(severity, "#7F8C8D")
