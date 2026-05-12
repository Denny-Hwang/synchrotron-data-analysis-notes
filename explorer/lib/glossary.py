"""Glossary auto-link — wrap known terms in rendered note HTML.

Parses ``08_references/glossary.md`` for ``**Term**: definition.`` lines
and provides :func:`annotate_html` which walks an HTML string token by
token and wraps the **first** occurrence of each known glossary term in
``<abbr class="eberlight-glossary" title="…">…</abbr>``. The function
deliberately skips text inside ``<code>``, ``<pre>``, ``<a>``,
``<abbr>``, and ``<h1>`` through ``<h6>`` so headings stay clean,
links don't nest, and code samples stay verbatim.

This is the senior-review action item that asked for
``08_references/glossary.md`` to actually show up in the explorer
instead of being a separate note nobody finds.

Ref: senior-review action item #9 (REL-E080).
"""

from __future__ import annotations

import re
from functools import lru_cache
from html import escape
from pathlib import Path

# Glossary line format: ``**Term**: definition sentence.``
# The term can contain spaces, hyphens, digits, slashes. The definition
# is the rest of the line; trailing parentheticals are kept.
_GLOSSARY_LINE = re.compile(r"^\*\*(?P<term>[^*\n]{2,80})\*\*:\s*(?P<defn>.+)$", re.MULTILINE)

# Token splitter: separates HTML into ``<tag>`` and text runs.
_HTML_TOKEN = re.compile(r"(<[^>]+>)")

# Detect whether a tag opens, closes, or is self-closing.
_OPEN_TAG = re.compile(r"<\s*([A-Za-z][A-Za-z0-9-]*)\b", re.ASCII)
_CLOSE_TAG = re.compile(r"<\s*/\s*([A-Za-z][A-Za-z0-9-]*)\s*>", re.ASCII)

# Elements whose contents should NOT receive glossary annotations.
# Headings stay clean (no clutter on H1/H2), code stays verbatim, and
# links / existing abbrs don't get nested anchors.
_SKIP_ELEMENTS: frozenset[str] = frozenset(
    {"a", "abbr", "code", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "kbd", "script", "style"}
)

# Minimum term length — below this the false-positive rate is too high
# (e.g. a single capital letter colliding with a chemical-formula token).
_MIN_TERM_LEN = 3


def _parse_glossary_text(text: str) -> dict[str, str]:
    """Parse glossary markdown into a ``{term: definition}`` mapping.

    The first sentence of the definition is kept verbatim (we strip
    trailing whitespace but preserve internal punctuation so chemistry
    notation and parenthetical citations round-trip). Duplicate terms
    (case-insensitively) keep the first occurrence.
    """
    glossary: dict[str, str] = {}
    for match in _GLOSSARY_LINE.finditer(text):
        term = match.group("term").strip()
        defn = match.group("defn").strip()
        if len(term) < _MIN_TERM_LEN:
            continue
        if term.lower() in {k.lower() for k in glossary}:
            continue  # de-dup, first-wins
        glossary[term] = defn
    return glossary


@lru_cache(maxsize=1)
def load_glossary(repo_root: Path) -> dict[str, str]:
    """Return ``{term: definition}`` parsed from ``08_references/glossary.md``.

    Cached on the absolute path so repeated calls across a Streamlit
    session pay the file IO + regex pass exactly once. ``maxsize=1`` —
    we only ever use a single repo root (REL-E081 M8).
    """
    path = Path(repo_root) / "08_references" / "glossary.md"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    return _parse_glossary_text(text)


@lru_cache(maxsize=4)
def _cached_match_regex(terms: tuple[str, ...]) -> re.Pattern[str]:
    """Cached version of :func:`_build_match_regex` keyed on the term tuple.

    REL-E081 B2 — the static-site builder calls ``annotate_html`` once per
    note (188 notes per build) and the Streamlit body renderer calls it
    once per markdown segment. Without caching the ~60-term alternation
    regex was recompiled each time; this brings it down to one compile
    per build (or per glossary edit during dev).
    """
    return _build_match_regex(list(terms))


def _build_match_regex(terms: list[str]) -> re.Pattern[str]:
    """Compile one alternation regex matching all terms, longest first.

    Sorting longest-first lets ``APS-U`` shadow ``APS`` and ``X-ray
    fluorescence`` shadow ``X-ray``; ``\\b`` word boundaries on each
    end keep ``APS`` from matching inside ``GAPS``. Case-insensitive
    so ``APS`` matches ``APS``, ``Aps``, ``aPS`` — common when authors
    re-cap the term mid-sentence.
    """
    if not terms:
        # Match nothing (negative lookahead trick) so callers can use
        # ``regex.sub`` unconditionally without a None check.
        return re.compile(r"(?!x)x")
    sorted_terms = sorted(terms, key=len, reverse=True)
    escaped = [re.escape(t) for t in sorted_terms]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


def annotate_html(
    html: str,
    glossary: dict[str, str],
    *,
    used: set[str] | None = None,
) -> str:
    """Wrap the first occurrence of each glossary term in an ``<abbr>``.

    Walks the input HTML token-by-token using a regex split, tracks
    the open-tag stack so text inside ``<code>``/``<a>``/``<h2>``/etc.
    is left untouched, and emits one ``<abbr>`` per term per document.
    Pure-Python; no BeautifulSoup dependency.

    Args:
        html: Rendered HTML body (post-markdown).
        glossary: ``{term: definition}`` mapping from :func:`load_glossary`.
        used: Optional **shared** "already-wrapped" set. When provided,
            the annotator will both read it (to skip terms already
            wrapped by a previous call) **and** mutate it (so later
            calls see the additions). Callers that render a single
            note body in multiple HTML segments (the Streamlit
            Mermaid-aware renderer splits at each ``‌```mermaid``
            block) should allocate one set and pass it through, so
            the "first occurrence **per document**" promise holds
            across the segment boundary. REL-E081 B1.

    Returns:
        HTML with the first occurrence of each glossary term wrapped in
        ``<abbr class="eberlight-glossary" tabindex="0" title="...">term</abbr>``.
        If ``glossary`` is empty or no matches occur the input is
        returned unchanged.
    """
    if not html or not glossary:
        return html

    regex = _cached_match_regex(tuple(glossary.keys()))
    glossary_lc = {k.lower(): v for k, v in glossary.items()}
    if used is None:
        used = set()

    # Token-by-token walk. Tags get appended verbatim and their
    # open/close kind toggles the skip-stack. Text segments outside
    # the skip-stack get glossary substitution.
    out: list[str] = []
    stack: list[str] = []
    tokens = _HTML_TOKEN.split(html)

    def _substitute(text: str) -> str:
        def _repl(match: re.Match[str]) -> str:
            term = match.group(1)
            lc = term.lower()
            if lc in used:
                return term
            defn = glossary_lc.get(lc)
            if not defn:
                return term
            used.add(lc)
            safe_defn = escape(defn, quote=True)
            # ``tabindex="0"`` makes the term keyboard-focusable so screen-
            # reader / keyboard users can land on it; pair with the CSS
            # ``:focus`` outline rule in styles.css for visual feedback.
            return (
                f'<abbr class="eberlight-glossary" tabindex="0" title="{safe_defn}">{term}</abbr>'
            )

        return regex.sub(_repl, text)

    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("<"):
            out.append(tok)
            close = _CLOSE_TAG.match(tok)
            if close:
                tag = close.group(1).lower()
                # Pop until we find the matching open-tag — handles
                # the not-quite-well-formed HTML that markdown sometimes
                # emits without raising.
                while stack and stack[-1] != tag:
                    stack.pop()
                if stack and stack[-1] == tag:
                    stack.pop()
                continue
            open_m = _OPEN_TAG.match(tok)
            if open_m:
                tag = open_m.group(1).lower()
                # Self-closing? ``<br/>``, ``<img ... />`` — no push.
                if tok.rstrip().endswith("/>"):
                    continue
                # ``void`` elements never have closing tags; do not push.
                if tag in {"br", "hr", "img", "input", "meta", "link", "source"}:
                    continue
                stack.append(tag)
            continue

        if any(t in _SKIP_ELEMENTS for t in stack):
            out.append(tok)
        else:
            out.append(_substitute(tok))

    return "".join(out)
