"""Progressive-disclosure helpers (Phase R5).

The legacy ``eberlight-explorer/`` shipped a four-step "Detail Level"
selector on every page: **L0 Overview**, **L1 Sections**, **L2 Details**,
**L3 Source**. Users could ramp up the depth from a one-liner all the
way to the raw markdown source. The new explorer keeps the same
four-step contract but derives every level from the **same markdown
body** — no separate L0/L1/L2 copies of each note (ADR-002 stays
intact).

Heuristic:

* **L0 Overview** — the first paragraph (intro), or the first ~300
  characters of body if the note has no leading paragraph break.
  Useful for one-glance scanning on cluster cards.
* **L1 Sections** — the H1/H2/H3 outline plus the first sentence
  under each heading. Gives a skimmable table of contents that's
  one click deeper than L0.
* **L2 Details** — the entire markdown body, rendered as today.
  This is the default level — clicking a note card lands here.
* **L3 Source** — the raw markdown text in a fenced code block.
  Useful when the user wants to copy/paste, fork, or audit
  formatting.

All four functions take a body string and return a string (markdown
or HTML — the page chooses how to render). The implementation is
pure: no Streamlit, no I/O.

Ref: ADR-002 — Notes are the SoT (no duplication).
Ref: FR-008 — 4-step zoom indicator (L0 → L3).
"""

from __future__ import annotations

import re

# Public canonical level vocabulary — order matters (rendered as a
# left-to-right radio in the sidebar). Display labels include short
# English summaries the legacy app used so users coming from there
# recognise the controls.
LEVELS: tuple[str, ...] = ("L0", "L1", "L2", "L3")

LEVEL_LABELS: dict[str, str] = {
    "L0": "🌍 Overview",
    "L1": "📋 Sections",
    "L2": "🔎 Details",
    "L3": "📄 Source",
}

LEVEL_HELP: dict[str, str] = {
    "L0": "Section summaries and one-glance bullets.",
    "L1": "Outline + first sentence per heading.",
    "L2": "Full content with images, tables, code, diagrams.",
    "L3": "Raw markdown source for copy / paste / fork.",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_frontmatter(body: str) -> str:
    """Remove leading YAML frontmatter (already stripped by ``load_notes``)."""
    return _FRONTMATTER_RE.sub("", body, count=1)


def _strip_top_h1(body: str) -> str:
    """Drop the leading ``# Title`` line if present.

    The note-detail page already renders the title above the body,
    so leaving it in would duplicate.
    """
    lines = body.lstrip().splitlines()
    if lines and lines[0].lstrip().startswith("# "):
        return "\n".join(lines[1:]).lstrip()
    return body


def _first_paragraph(body: str, *, max_chars: int = 600) -> str:
    """Return the first non-heading paragraph as plain markdown."""
    text = _strip_top_h1(_strip_frontmatter(body)).strip()
    # Pull paragraphs separated by a blank line, skip ones that start
    # with a heading marker (we want prose, not the next H2).
    for chunk in re.split(r"\n{2,}", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.lstrip().startswith("#"):
            continue
        if len(chunk) > max_chars:
            return chunk[:max_chars].rstrip() + "…"
        return chunk
    return text[:max_chars].rstrip()


def _outline(body: str) -> list[tuple[int, str, str]]:
    """Return a list of ``(depth, heading, first-sentence)`` triples.

    ``depth`` is the heading level (1 for H1, 2 for H2, …). The
    ``first-sentence`` is the first non-empty line under that heading
    that doesn't itself start with ``#`` or a fenced ``​```.
    """
    text = _strip_frontmatter(body)
    out: list[tuple[int, str, str]] = []
    matches = list(_HEADING_RE.finditer(text))
    for i, m in enumerate(matches):
        depth = len(m.group("hashes"))
        title = m.group("title").strip()
        # Slice between this heading and the next (or end of body).
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end]
        first_sentence = _first_paragraph(section, max_chars=180)
        # Strip code fences / list markers from the snippet so the
        # outline preview reads as prose.
        first_sentence = re.sub(r"^```.*$", "", first_sentence, flags=re.MULTILINE)
        first_sentence = first_sentence.strip().split("\n")[0]
        out.append((depth, title, first_sentence))
    return out


# --------------------------------------------------------------------------
# Public API: render one of the four levels as markdown
# --------------------------------------------------------------------------


def render_l0(body: str) -> str:
    """L0 — one-paragraph overview. Plain markdown."""
    para = _first_paragraph(body, max_chars=600)
    if not para:
        return "_(No prose preview available — open the full note.)_"
    return para


def render_l1(body: str) -> str:
    """L1 — outline + first sentence per heading, as nested markdown."""
    outline = _outline(body)
    if not outline:
        # No headings — degrade gracefully to the L0 paragraph.
        return render_l0(body)
    lines: list[str] = []
    for depth, heading, snippet in outline:
        # Skip H1 titles; the page already renders the title above.
        if depth == 1:
            continue
        indent = "  " * max(depth - 2, 0)
        lines.append(f"{indent}- **{heading}**")
        if snippet:
            lines.append(f"{indent}  {snippet}")
    if not lines:
        return render_l0(body)
    return "\n".join(lines)


def render_l2(body: str) -> str:
    """L2 — the body as-is. The default rendering depth."""
    return body


def render_l3(body: str) -> str:
    """L3 — raw markdown source in a fenced code block.

    Markdown viewers can choose to render the fence as syntax-
    highlighted text. The fence language is ``markdown`` so a
    pygments-aware renderer picks the right lexer.
    """
    # Make sure the user's content doesn't break our fence.
    safe = body.replace("```", "\\`\\`\\`")
    return f"```markdown\n{safe}\n```"


def render(level: str, body: str) -> str:
    """Dispatch to the right ``render_lN`` function.

    Unknown / malformed level strings fall back to L2 silently
    (the page decides whether to log a warning).
    """
    return {
        "L0": render_l0,
        "L1": render_l1,
        "L2": render_l2,
        "L3": render_l3,
    }.get(level.upper(), render_l2)(body)


def normalise_level(raw: str | None, *, default: str = "L2") -> str:
    """Map ``?level=…`` query-param values to the canonical vocabulary.

    Accepts the canonical id (``L0``..``L3``) or the legacy
    long-form labels the old explorer used (``Overview``, ``Sections``,
    ``Details``, ``Source``).
    """
    if not raw:
        return default
    s = raw.strip().lower()
    if s in {"l0", "overview", "summary"}:
        return "L0"
    if s in {"l1", "sections", "outline"}:
        return "L1"
    if s in {"l2", "details", "full"}:
        return "L2"
    if s in {"l3", "source", "raw", "markdown"}:
        return "L3"
    return default
