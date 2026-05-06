"""Search + Bibliography (Phase R6).

Two surfaces in one page:

1. **Full-text search** over the loaded notes — title-boosted
   TF-IDF approximation, snippet around the first match, deep
   links into the same ``?note=…`` routing the cluster pages use.
   Supports ``?q=<query>`` for shareable URLs.
2. **Bibliography** — every BibTeX entry from
   ``08_references/bibliography.bib`` and
   ``10_interactive_lab/CITATIONS.bib``, with author / year filter
   and clickable DOI links.

Ref: FR-009 — global search bar.
Ref: ADR-002 — notes are the SoT (search index built at runtime).
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote, unquote

import streamlit as st

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from components.footer import render_footer
from components.header import render_header
from lib.bibliography import BibEntry, collect_bibliography
from lib.search import Index, SearchHit, index_from_repo, search

st.set_page_config(page_title="Search — eBERlight", page_icon="🔍", layout="wide")

_REPO_ROOT = _EXPLORER_DIR.parent
_CSS_PATH = _EXPLORER_DIR / "assets" / "styles.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource
def _cached_index() -> Index:
    return index_from_repo(_REPO_ROOT)


@st.cache_resource
def _cached_bibliography() -> list[BibEntry]:
    return collect_bibliography(_REPO_ROOT)


def _query_param(name: str) -> str | None:
    raw = st.query_params.get(name)
    if raw is None:
        return None
    if isinstance(raw, list):
        return unquote(raw[0]) if raw else None
    return unquote(str(raw))


idx = _cached_index()
bibliography = _cached_bibliography()

render_header(active_cluster=None)
st.markdown(
    '<h1 style="color:#0033A0;">🔍 Search</h1>'
    '<p style="color:#555;font-size:16px;margin-bottom:16px;">'
    f"Full-text search over <b>{len(idx)}</b> notes plus a unified "
    f"bibliography of <b>{len(bibliography)}</b> BibTeX entries.</p>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Search — query box + results
# ---------------------------------------------------------------------------


initial_q = _query_param("q") or ""
query = st.text_input(
    "Type a query — terms are AND-ed implicitly; titles are boosted ×2",
    value=initial_q,
    key="search_q",
    label_visibility="visible",
)

if query:
    hits: list[SearchHit] = search(idx, query, limit=30)
    st.markdown(f"### Results — {len(hits)} match{'es' if len(hits) != 1 else ''}")
    if not hits:
        st.info("No matches. Try fewer / shorter terms.")
    for hit in hits:
        href = f"?note={quote(hit.note.url_id(_REPO_ROOT), safe='/')}"
        terms_html = " · ".join(
            f'<code style="background:#FFF4E0;padding:1px 6px;border-radius:4px;">{t}</code>'
            for t in hit.matched_terms[:6]
        )
        folder_label = hit.note.folder.split("_", 1)[1].replace("_", " ").title()
        st.markdown(
            f'<div class="eberlight-card" style="margin-bottom:12px;">'
            f'<h4 style="margin:0 0 6px 0;">'
            f'<a href="{href}" target="_self" '
            f'style="color:#0033A0;text-decoration:none;">{hit.note.title}</a>'
            f"</h4>"
            f'<p style="font-size:12px;color:#888;margin:0 0 6px 0;">'
            f"{folder_label} · score {hit.score:.2f}</p>"
            f'<p style="font-size:14px;color:#333;margin:0 0 8px 0;'
            f'line-height:1.5;">{hit.snippet}</p>'
            f'<p style="font-size:12px;color:#666;margin:0;">'
            f"Matched: {terms_html}</p>"
            "</div>",
            unsafe_allow_html=True,
        )
else:
    st.caption(
        "Tip: try keywords like `tomography`, `noise2noise`, `XANES`, "
        "or a tool name like `tomopy`, `topaz`."
    )


# ---------------------------------------------------------------------------
# Bibliography
# ---------------------------------------------------------------------------


st.markdown("---")
st.markdown("### Bibliography")
if not bibliography:
    st.info("No `.bib` files found. Drop one at `08_references/bibliography.bib`.")
else:
    cols = st.columns([2, 1, 1])
    with cols[0]:
        bib_query = st.text_input(
            "Filter by title / author / key",
            value="",
            key="bib_q",
            placeholder="e.g. tomogan, vo, 2018",
        )
    with cols[1]:
        years = sorted({e.year for e in bibliography if e.year is not None}, reverse=True)
        year_filter = st.multiselect("Year", options=years, default=[], key="bib_year")
    with cols[2]:
        types = sorted({e.entry_type for e in bibliography})
        type_filter = st.multiselect("Entry type", options=types, default=[], key="bib_type")

    def _match(e: BibEntry) -> bool:
        if year_filter and e.year not in year_filter:
            return False
        if type_filter and e.entry_type not in type_filter:
            return False
        if not bib_query:
            return True
        q = bib_query.lower()
        haystack = " ".join(
            [
                e.title or "",
                " ".join(e.authors),
                e.key or "",
                e.venue or "",
                str(e.year or ""),
            ]
        ).lower()
        return q in haystack

    visible = [e for e in bibliography if _match(e)]
    st.caption(f"{len(visible)} of {len(bibliography)} entries.")

    for e in visible:
        doi_link = (
            f' &nbsp;·&nbsp; <a href="{e.doi_url}" target="_blank" rel="noopener" '
            f'style="color:#0033A0;">DOI ↗</a>'
            if e.doi
            else ""
        )
        author_str = ", ".join(e.authors) if e.authors else f"<em>(no author — key {e.key})</em>"
        st.markdown(
            f'<div class="eberlight-card" style="margin-bottom:8px;">'
            f'<div style="font-size:14px;color:#1A1A1A;font-weight:600;'
            f'margin-bottom:4px;">{e.title or e.key}</div>'
            f'<div style="font-size:13px;color:#555;">'
            f"{author_str} · "
            f'<span style="color:#888;">{e.year or "—"}</span> · '
            f"<em>{e.venue or e.entry_type}</em>"
            f"{doi_link}"
            "</div>"
            f'<div style="font-size:11px;color:#888;margin-top:4px;'
            f'font-family:monospace;">{e.key}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

render_footer()
