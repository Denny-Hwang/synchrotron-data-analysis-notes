"""Lightweight BibTeX parser for the Bibliography page (Phase R6).

The two `.bib` files in the repo (`08_references/bibliography.bib`
and `10_interactive_lab/CITATIONS.bib`) follow a permissive subset
of BibTeX. We don't need a full parser — we just want to extract:

- entry type (`@article`, `@inproceedings`, `@misc`, …)
- citation key
- key fields: ``title``, ``author``, ``year``, ``journal`` /
  ``booktitle`` / ``venue``, ``doi``

Anything more elaborate (cross-references, abbrevs, string macros)
is uncommon in this corpus and would require pulling in
`pybtex` / `bibtexparser` for a one-off page.

Pure data layer — no Streamlit, no I/O until the public
``parse_bib_file(path)`` is called.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Top-level entry: `@type{key,` … `}`. Tolerant of whitespace and
# comments-as-blank-lines between entries.
_ENTRY_RE = re.compile(
    r"@(?P<type>\w+)\s*\{\s*(?P<key>[^,\s]+)\s*,(?P<body>.*?)\}\s*(?=@|\Z)",
    re.DOTALL,
)
# Field inside an entry: `name = "value"` or `name = {value}` —
# values can contain balanced braces, escaped quotes, etc.
# We capture lazily and rely on the closing comma/brace to terminate.
_FIELD_RE = re.compile(
    r"(?P<name>\w+)\s*=\s*(?:"
    r'"(?P<dq>(?:[^"\\]|\\.)*)"'
    r"|"
    r"\{(?P<br>(?:[^{}]|\{[^{}]*\})*)\}"
    r"|"
    r"(?P<bare>[^,\n]+)"
    r")",
    re.DOTALL,
)

_AUTHOR_AND_RE = re.compile(r"\s+and\s+", re.IGNORECASE)


@dataclass(frozen=True)
class BibEntry:
    """One BibTeX record."""

    key: str
    entry_type: str
    title: str
    authors: tuple[str, ...]
    year: int | None = None
    venue: str = ""
    doi: str = ""
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def doi_url(self) -> str:
        return f"https://doi.org/{self.doi}" if self.doi else ""

    def render_apa_short(self) -> str:
        """Author1 et al. (Year). *Title*. Venue. DOI: …"""
        if not self.authors:
            attribution = self.key
        elif len(self.authors) == 1:
            attribution = self.authors[0]
        else:
            attribution = f"{self.authors[0]} et al."
        year = f"({self.year})" if self.year else ""
        title = self.title
        venue = f" *{self.venue}*." if self.venue else ""
        doi = f" DOI: {self.doi}" if self.doi else ""
        return f"{attribution} {year}. {title}.{venue}{doi}".strip()


# R12 B4 — BibTeX preserves accented characters via LaTeX escapes
# (``{\'e}`` for ``é``, ``{\^o}`` for ``ô``, ``{\"u}`` for ``ü`` …).
# The bibliography page used to render those literally, leaving
# ``J{\'e}r{\^o}me`` instead of ``Jérôme`` in author bylines.
# Mapping below covers every accent that appears in a typical
# scientific bibliography; non-listed escapes fall through unchanged.
# R12 B4 — also match a trailing space-separator after bare commands so
# ``\AA ngstr...`` yields ``Ångstr...`` (LaTeX consumes the separator
# that terminates the command name; Python regex doesn't, hence the
# explicit `[ \t]?` in the lone arm).
_LATEX_ACCENT_RE = re.compile(
    r"""
    \{\\
    (?P<accent>['`^"~=.uvHrco]|[a-zA-Z]+)   # accent or special name
    \s*
    \{?(?P<base>[A-Za-z]?)\}?               # base letter (optional)
    \}
    |
    \\(?P<lone>[a-zA-Z]+)[ \t]?             # bare commands like \aa, \ss
    """,
    re.VERBOSE,
)

# (accent, base) → unicode replacement. Populated explicitly so each
# entry is greppable and the table is easy to extend later.
_ACCENT_MAP: dict[tuple[str, str], str] = {
    # Acute (\')
    ("'", "a"): "á",
    ("'", "e"): "é",
    ("'", "i"): "í",
    ("'", "o"): "ó",
    ("'", "u"): "ú",
    ("'", "y"): "ý",
    ("'", "A"): "Á",
    ("'", "E"): "É",
    ("'", "I"): "Í",
    ("'", "O"): "Ó",
    ("'", "U"): "Ú",
    ("'", "Y"): "Ý",
    # Grave (\`)
    ("`", "a"): "à",
    ("`", "e"): "è",
    ("`", "i"): "ì",
    ("`", "o"): "ò",
    ("`", "u"): "ù",
    ("`", "A"): "À",
    ("`", "E"): "È",
    ("`", "I"): "Ì",
    ("`", "O"): "Ò",
    ("`", "U"): "Ù",
    # Circumflex (\^)
    ("^", "a"): "â",
    ("^", "e"): "ê",
    ("^", "i"): "î",
    ("^", "o"): "ô",
    ("^", "u"): "û",
    ("^", "A"): "Â",
    ("^", "E"): "Ê",
    ("^", "I"): "Î",
    ("^", "O"): "Ô",
    ("^", "U"): "Û",
    # Diaeresis / umlaut (\")
    ('"', "a"): "ä",
    ('"', "e"): "ë",
    ('"', "i"): "ï",
    ('"', "o"): "ö",
    ('"', "u"): "ü",
    ('"', "y"): "ÿ",
    ('"', "A"): "Ä",
    ('"', "E"): "Ë",
    ('"', "I"): "Ï",
    ('"', "O"): "Ö",
    ('"', "U"): "Ü",
    # Tilde (\~)
    ("~", "a"): "ã",
    ("~", "n"): "ñ",
    ("~", "o"): "õ",
    ("~", "A"): "Ã",
    ("~", "N"): "Ñ",
    ("~", "O"): "Õ",
    # Cedilla (\c)
    ("c", "c"): "ç",
    ("c", "C"): "Ç",
    # Caron / hacek (\v)
    ("v", "c"): "č",
    ("v", "s"): "š",
    ("v", "z"): "ž",
    ("v", "C"): "Č",
    ("v", "S"): "Š",
    ("v", "Z"): "Ž",
}

_LONE_LATEX_MAP: dict[str, str] = {
    "ss": "ß",
    "aa": "å",
    "AA": "Å",
    "o": "ø",
    "O": "Ø",
    "ae": "æ",
    "AE": "Æ",
    "oe": "œ",
    "OE": "Œ",
    "l": "ł",
    "L": "Ł",
}


def _decode_latex_accents(text: str) -> str:
    """Replace common LaTeX accent escapes with their Unicode equivalents."""
    if not text or "\\" not in text:
        return text

    def _sub(m: re.Match[str]) -> str:
        if m.group("lone"):
            cmd = m.group("lone")
            return _LONE_LATEX_MAP.get(cmd, "\\" + cmd)
        accent = m.group("accent") or ""
        base = m.group("base") or ""
        if (accent, base) in _ACCENT_MAP:
            return _ACCENT_MAP[(accent, base)]
        # ``{\ss}`` / ``{\aa}`` / ``{\oe}`` etc. — bare command inside
        # protective braces. Empty base + accent in the lone-command
        # map → use that.
        if not base and accent in _LONE_LATEX_MAP:
            return _LONE_LATEX_MAP[accent]
        # Unknown accent — strip LaTeX braces but keep the base
        # letter so the author name is still legible.
        return base

    return _LATEX_ACCENT_RE.sub(_sub, text)


def _clean(value: str) -> str:
    """Strip outer braces/whitespace, collapse whitespace, decode accents."""
    cleaned = re.sub(r"\s+", " ", value.strip().strip("{}").strip())
    return _decode_latex_accents(cleaned)


def _parse_authors(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    # Decode accents first so author splitting on `` and `` doesn't break
    # inside an accent escape.
    decoded = _decode_latex_accents(raw)
    parts = [_clean(p) for p in _AUTHOR_AND_RE.split(decoded)]
    return tuple(p for p in parts if p)


def _parse_year(raw: str) -> int | None:
    m = re.search(r"\d{4}", raw or "")
    return int(m.group()) if m else None


def parse_bib_text(text: str) -> list[BibEntry]:
    """Parse a BibTeX corpus into a list of :class:`BibEntry`."""
    out: list[BibEntry] = []
    for entry in _ENTRY_RE.finditer(text):
        body = entry.group("body")
        fields: dict[str, str] = {}
        for fm in _FIELD_RE.finditer(body):
            value = fm.group("dq") or fm.group("br") or fm.group("bare") or ""
            fields[fm.group("name").lower()] = _clean(value)

        title = fields.get("title", "")
        authors = _parse_authors(fields.get("author", ""))
        venue = (
            fields.get("journal")
            or fields.get("booktitle")
            or fields.get("venue")
            or fields.get("publisher")
            or ""
        )
        doi = fields.get("doi", "").strip()
        out.append(
            BibEntry(
                key=entry.group("key").strip(),
                entry_type=entry.group("type").lower(),
                title=title,
                authors=authors,
                year=_parse_year(fields.get("year", "")),
                venue=venue,
                doi=doi,
                raw=fields,
            )
        )
    return out


def parse_bib_file(path: Path) -> list[BibEntry]:
    """Parse a ``.bib`` file. Returns ``[]`` if the file is missing."""
    if not path.is_file():
        return []
    return parse_bib_text(path.read_text(encoding="utf-8"))


def collect_bibliography(repo_root: Path) -> list[BibEntry]:
    """Aggregate every ``.bib`` we know about into one sorted list."""
    out: list[BibEntry] = []
    for relative in (
        "08_references/bibliography.bib",
        "10_interactive_lab/CITATIONS.bib",
    ):
        out.extend(parse_bib_file(repo_root / relative))
    return sorted(out, key=lambda e: (-(e.year or 0), e.key))
