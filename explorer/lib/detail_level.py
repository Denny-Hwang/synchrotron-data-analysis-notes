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


_HEADING_LINE_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")
# Opening / closing of a fenced code block — ``` or ~~~, optionally
# preceded by whitespace (indented fences are tolerated by markdown).
_FENCE_RE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})")
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


def _strip_code_fences(body: str) -> str:
    """Remove every fenced code block from a markdown body.

    Used so a leading prose paragraph isn't mistaken for the contents
    of a ``# heading-like`` Python comment that lives inside a fenced
    block. Paragraph boundaries (blank lines) outside fences are
    preserved.
    """
    out_lines: list[str] = []
    in_fence = False
    fence_marker: str | None = None
    for line in body.splitlines():
        m = _FENCE_RE.match(line)
        if m:
            marker = m.group("fence")
            if not in_fence:
                in_fence = True
                fence_marker = marker
                continue
            if (
                fence_marker is not None
                and len(marker) >= len(fence_marker)
                and marker[0] == fence_marker[0]
            ):
                in_fence = False
                fence_marker = None
                continue
        if in_fence:
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _first_paragraph(body: str, *, max_chars: int = 600) -> str:
    """Return the first non-heading, non-fenced paragraph as plain markdown."""
    text = _strip_top_h1(_strip_frontmatter(body)).strip()
    text = _strip_code_fences(text).strip()
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
    """Return ``(depth, heading, first-sentence)`` triples for the body.

    Walks the body line by line tracking whether we're currently inside
    a fenced code block (``` or ~~~), so lines starting with ``#``
    inside a Python / shell code listing are NOT mistaken for section
    headings (Codex review of PR #45).
    """
    text = _strip_frontmatter(body)
    lines = text.splitlines()

    in_fence = False
    fence_marker: str | None = None
    headings: list[tuple[int, int, str]] = []  # (line_idx, depth, title)
    for line_idx, line in enumerate(lines):
        m = _FENCE_RE.match(line)
        if m:
            marker = m.group("fence")
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif (
                fence_marker is not None
                and len(marker) >= len(fence_marker)
                and marker[0] == fence_marker[0]
            ):
                in_fence = False
                fence_marker = None
            continue
        if in_fence:
            continue
        h = _HEADING_LINE_RE.match(line)
        if h:
            headings.append((line_idx, len(h.group("hashes")), h.group("title").strip()))

    out: list[tuple[int, str, str]] = []
    for i, (line_idx, depth, title) in enumerate(headings):
        next_line_idx = headings[i + 1][0] if i + 1 < len(headings) else len(lines)
        snippet = ""
        sub_in_fence = False
        sub_marker: str | None = None
        for sub_line in lines[line_idx + 1 : next_line_idx]:
            fm = _FENCE_RE.match(sub_line)
            if fm:
                marker = fm.group("fence")
                if not sub_in_fence:
                    sub_in_fence = True
                    sub_marker = marker
                elif (
                    sub_marker is not None
                    and len(marker) >= len(sub_marker)
                    and marker[0] == sub_marker[0]
                ):
                    sub_in_fence = False
                    sub_marker = None
                continue
            if sub_in_fence:
                continue
            stripped = sub_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            snippet = stripped[:180]
            break
        out.append((depth, title, snippet))
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


_BACKTICK_RUN_RE = re.compile(r"`+")


def render_l3(body: str) -> str:
    """L3 — raw markdown source in a fenced code block.

    The outer fence is **as long as needed** to contain any inner
    backtick run verbatim — this preserves embedded ``‌```mermaid`` /
    ``‌```python`` blocks so users can copy / paste / fork the page
    source without backslash-escapes leaking into their workflow.
    Per CommonMark §4.5: a fenced code block ends only on a fence
    of the same character class with at least as many backticks, so
    bumping the outer length by one always wins.
    """
    longest_run = max((len(m.group()) for m in _BACKTICK_RUN_RE.finditer(body)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}markdown\n{body}\n{fence}"


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


_ANCHOR_SAFE_RE = re.compile(r"[^a-z0-9-]+")


def _heading_anchor(title: str) -> str:
    """Convert a heading text into a GitHub-style markdown anchor slug."""
    slug = _ANCHOR_SAFE_RE.sub("-", title.lower()).strip("-")
    return slug or "section"


def extract_toc(body: str, *, max_depth: int = 3) -> list[tuple[int, str, str]]:
    """Return ``(depth, anchor, heading)`` triples for every H1..H``max_depth``.

    Used by the note-detail view to render an in-page table of
    contents. Headings inside fenced code blocks are skipped via the
    same line-by-line walker the L1 outline helper uses.
    """
    out = _outline(body)
    return [(d, _heading_anchor(t), t) for d, t, _ in out if 1 <= d <= max_depth]


def split_into_sections(body: str, *, level: int = 2) -> list[tuple[str, str]]:
    """Split a markdown body into ``(heading, body-without-heading)`` chunks.

    The split points are the H``level`` headings only — any H1 title
    above and any preamble is stitched together as a leading
    ``"Overview"`` section. Code-fence-aware (a ``# comment`` inside a
    Python fence does not start a new section). This powers the
    note-detail "Tabs" view that mirrors the legacy Publications
    page L2 (Background / Method / Key Results / …).
    """
    text = _strip_frontmatter(body)
    lines = text.splitlines()
    in_fence = False
    fence_marker: str | None = None

    sections: list[tuple[str, list[str]]] = []
    current_label = "Overview"
    current_lines: list[str] = []
    target_hashes = "#" * level

    for raw_line in lines:
        m = _FENCE_RE.match(raw_line)
        if m:
            marker = m.group("fence")
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif (
                fence_marker is not None
                and len(marker) >= len(fence_marker)
                and marker[0] == fence_marker[0]
            ):
                in_fence = False
                fence_marker = None
            current_lines.append(raw_line)
            continue

        if not in_fence:
            stripped = raw_line.lstrip()
            if stripped.startswith(target_hashes + " "):
                heading = stripped[len(target_hashes) :].strip()
                # Flush the in-progress section (skip if first heading
                # found and current section is empty preamble — keeps
                # output free of empty Overview tabs).
                if sections or any(s.strip() for s in current_lines):
                    sections.append((current_label, current_lines))
                current_label = heading
                current_lines = []
                continue
            # Strip the H1 title line entirely — the page renders it.
            if stripped.startswith("# ") and not sections and not current_lines:
                continue

        current_lines.append(raw_line)

    # Always flush the trailing section once any heading has been seen — an
    # empty body under the last H2 is still a valid (empty) section.
    if sections or any(s.strip() for s in current_lines):
        sections.append((current_label, current_lines))

    return [(label, "\n".join(body_lines).strip()) for label, body_lines in sections]
