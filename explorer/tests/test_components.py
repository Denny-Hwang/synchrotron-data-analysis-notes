"""Smoke tests for eBERlight Explorer UI components.

Each render function is tested to ensure it runs without error and
produces non-empty output via st.markdown.

Ref: TST-001 (test_plan.md) — Component smoke tests.
"""

from unittest.mock import MagicMock, patch


@patch("streamlit.markdown")
def test_render_header(mock_markdown: MagicMock) -> None:
    """render_header() calls st.markdown with non-empty HTML."""
    from components.header import render_header

    render_header()

    mock_markdown.assert_called_once()
    html_output = mock_markdown.call_args[0][0]
    assert len(html_output.strip()) > 0
    assert "eBERlight Explorer" in html_output
    assert "eberlight-header" in html_output


@patch("streamlit.markdown")
def test_render_header_includes_search_form_by_default(mock_markdown: MagicMock) -> None:
    """FR-009 — every page that calls render_header() gets the search form."""
    from components.header import render_header

    render_header()
    html_output = mock_markdown.call_args[0][0]
    assert 'action="/Search"' in html_output
    assert 'name="q"' in html_output
    assert "eberlight-header-search" in html_output


@patch("streamlit.markdown")
def test_render_header_can_suppress_search(mock_markdown: MagicMock) -> None:
    """The Search page itself opts out via show_search=False."""
    from components.header import render_header

    render_header(show_search=False)
    html_output = mock_markdown.call_args[0][0]
    assert "eberlight-header-search" not in html_output


@patch("streamlit.markdown")
def test_render_header_pre_fills_search_input(mock_markdown: MagicMock) -> None:
    """initial_query is URL-encoded into the input value attribute."""
    from components.header import render_header

    render_header(initial_query='ring "vo 2018"')
    html_output = mock_markdown.call_args[0][0]
    # Plus-form encoding for spaces is acceptable; check the safe payload.
    assert "ring" in html_output
    assert 'value="' in html_output


@patch("streamlit.markdown")
def test_render_breadcrumb_with_items(mock_markdown: MagicMock) -> None:
    """render_breadcrumb() renders linked and current items."""
    from components.breadcrumb import render_breadcrumb

    render_breadcrumb([("Home", "/"), ("Explore", "/explore"), ("TomoGAN", None)])

    mock_markdown.assert_called_once()
    html_output = mock_markdown.call_args[0][0]
    assert "Home" in html_output
    assert "Explore" in html_output
    assert "TomoGAN" in html_output
    assert 'class="current"' in html_output
    assert "eberlight-breadcrumb" in html_output


@patch("streamlit.markdown")
def test_render_breadcrumb_empty(mock_markdown: MagicMock) -> None:
    """render_breadcrumb() with empty list produces no output."""
    from components.breadcrumb import render_breadcrumb

    render_breadcrumb([])

    mock_markdown.assert_not_called()


@patch("streamlit.markdown")
def test_render_footer(mock_markdown: MagicMock) -> None:
    """render_footer() includes DOE acknowledgment and contract number."""
    from components.footer import render_footer

    render_footer()

    mock_markdown.assert_called_once()
    html_output = mock_markdown.call_args[0][0]
    assert len(html_output.strip()) > 0
    assert "DE-AC02-06CH11357" in html_output
    assert "eberlight-footer" in html_output
    assert "Advanced Photon Source" in html_output
    assert "eBERlight" in html_output


# ---------------------------------------------------------------------------
# R14 regression: ``[object Object]`` leak in code blocks (note_view._md_to_html)
# ---------------------------------------------------------------------------


def test_md_to_html_does_not_emit_pygments_class_spans() -> None:
    """Streamlit's React frontend renders ``<pre><code>`` HTML via its
    native ``stCode`` component, which expects a *string* child. If
    ``_md_to_html`` runs codehilite, the code block becomes a tree of
    ``<span class="kn">…</span>`` Pygments tokens and React stringifies
    them as ``[object Object]`` — the bug R14 hotfixed.

    This test asserts the rendered HTML for a Python fenced code block
    contains **no** Pygments class spans (``<span class="kn">``,
    ``<span class="nn">``, etc.) and **no** ``<div class="highlight">``
    wrapper. If a future change re-introduces codehilite, this guard
    fires immediately.
    """
    from components.note_view import _md_to_html

    body = (
        "## Quick Diagnosis\n\n"
        "```python\n"
        "import numpy as np\n"
        "col_std = np.std(sinogram, axis=0)\n"
        'print(f"Suspicious columns: {outlier_cols}")\n'
        "```\n"
    )
    html = _md_to_html(body)

    # The fenced_code extension's normal output: no class wrapper, no
    # span tree — just the raw code as text inside <pre><code>.
    assert "<pre><code" in html
    assert "import numpy as np" in html
    assert '<div class="highlight">' not in html, (
        "codehilite is back — Streamlit will render code blocks as "
        "[object Object]. Drop it from _md_to_html's extensions list."
    )
    assert '<span class="kn"' not in html
    assert '<span class="nn"' not in html
    assert '<span class="o"' not in html


def test_md_to_html_emits_language_class_for_prism() -> None:
    """The ``fenced_code`` extension must keep emitting ``class="language-X"``
    on fenced blocks — Streamlit's native code component highlights via
    Prism.js keyed on this class."""
    from components.note_view import _md_to_html

    html = _md_to_html("```python\nx = 1\n```\n")
    assert 'class="language-python"' in html


def test_render_body_with_mermaid_signature_no_highlight_css() -> None:
    """R14 — ``_render_body_with_mermaid`` must NOT take a
    ``highlight_css`` argument. Re-introducing it would mean
    re-introducing the Pygments css path, which only exists to
    style the codehilite span tree that the bug originated from.
    """
    import inspect

    from components.note_view import _render_body_with_mermaid, _render_section_tabs

    body_sig = inspect.signature(_render_body_with_mermaid)
    assert "highlight_css" not in body_sig.parameters
    tabs_sig = inspect.signature(_render_section_tabs)
    assert "highlight_css" not in tabs_sig.parameters


# ---------------------------------------------------------------------------
# R14 regression: vis.js tooltip rendered raw <b>...</b> as text
# ---------------------------------------------------------------------------


def test_visjs_graph_converts_title_to_html_element() -> None:
    """vis-network 9.x renders ``node.title`` strings via
    ``document.createTextNode`` — so passing ``"<b>X</b>"`` shows up
    as the literal text ``<b>X</b>``. The component must convert the
    HTML string to an ``HTMLElement`` client-side before the
    ``vis.Network`` constructor sees it.

    This test reads the rendered iframe HTML produced by
    ``render_visjs_graph`` and asserts the JS path contains the
    ``htmlToElement`` helper *and* the ``titleHtml`` indirection that
    survives ``JSON.parse(JSON.stringify(...))`` deep-cloning.
    """
    from unittest.mock import patch

    from components.visjs_graph import render_visjs_graph

    nodes = [
        {
            "id": "a",
            "label": "Electron Microscopy",
            "title": "<b>Electron Microscopy</b><br><i>modality</i>",
            "color": "#0033A0",
            "size": 30,
        }
    ]
    edges: list[dict] = []

    with patch("streamlit.components.v1.html") as mock_html:
        render_visjs_graph(nodes, edges, height=400)

    mock_html.assert_called_once()
    iframe_src = mock_html.call_args[0][0]

    # The HTML string is JSON-encoded in the iframe — that's expected.
    assert '"<b>Electron Microscopy</b><br><i>modality</i>"' in iframe_src

    # The fix: a JS helper that wraps the string in a div via innerHTML
    # so vis.js sees an HTMLElement and renders the markup.
    assert "htmlToElement" in iframe_src
    assert "wrap.innerHTML" in iframe_src

    # The clone-safe indirection: titles are stashed on ``titleHtml`` so
    # ``JSON.parse(JSON.stringify(...))`` doesn't blow them away (it
    # would replace the HTMLElement with ``{}``).
    assert "titleHtml" in iframe_src
    assert "n.title = htmlToElement(n.titleHtml)" in iframe_src


def test_visjs_graph_no_title_means_no_tooltip_object() -> None:
    """When a node has no ``title``, the iframe JS must not try to
    convert ``undefined`` — it should leave the property absent."""
    from unittest.mock import patch

    from components.visjs_graph import render_visjs_graph

    nodes = [{"id": "a", "label": "Plain", "color": "#0033A0", "size": 14}]
    with patch("streamlit.components.v1.html") as mock_html:
        render_visjs_graph(nodes, [], height=300)

    iframe_src = mock_html.call_args[0][0]
    # The titleHtml branch must guard against missing titles.
    assert "if (n.titleHtml)" in iframe_src
    assert "delete n.title" in iframe_src
