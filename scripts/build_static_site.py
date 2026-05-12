"""Static site generator for eBERlight Explorer — GitHub Pages mirror.

Reads the same note folders and IA mapping the Streamlit explorer uses
(`explorer/lib/ia.py`, `explorer/lib/notes.py`) and emits a fully static
HTML site that mirrors the Streamlit app's pages 1:1:

- Landing (hero + 3 cluster cards + 4 feature CTAs)  ← explorer/app.py
- Discover cluster page                              ← explorer/pages/1_Discover.py
- Explore cluster page (grouped)                     ← explorer/pages/2_Explore.py
- Build cluster page (grouped)                       ← explorer/pages/3_Build.py
- Knowledge Graph stub (interactive surface)         ← explorer/pages/0_Knowledge_Graph.py
- Interactive Lab stub (interactive surface)         ← explorer/pages/4_Experiment.py
- Troubleshooter stub (interactive surface)          ← explorer/pages/5_Troubleshooter.py
- Search stub (interactive surface)                  ← explorer/pages/6_Search.py
- Note detail pages (markdown + aside)               ← explorer/components/note_view.py

Pages whose value is in interactive controls (Plotly graph, parameter
sliders, search box, decision tree) cannot be rendered usefully as
flat HTML. The generator emits a stub page for each so the static
site keeps the same URL surface area as Streamlit, with a banner that
points readers to ``streamlit run explorer/app.py``. This is the same
pattern FR-022 introduced for the Recipe Gallery.

The generator also copies the design wireframes under /wireframes/ so
the existing wireframe preview keeps working.

Ref: ADR-007 — Static site mirror for GitHub Pages.
Ref: DS-001 — Design system tokens (reuses explorer/assets/styles.css).
Ref: IA-001, ADR-004 — 3-cluster IA reused via explorer.lib.ia.
Ref: CLAUDE.md invariant #9 — Pages MUST mirror the Streamlit explorer.
"""

from __future__ import annotations

import argparse
import html as html_escape_mod
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from itertools import groupby
from pathlib import Path
from urllib.parse import quote

import markdown
from pygments.formatters import HtmlFormatter

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPLORER_DIR = _REPO_ROOT / "explorer"
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from lib.experiments import Recipe, load_recipes
from lib.glossary import annotate_html as _glossary_annotate
from lib.glossary import load_glossary
from lib.ia import CLUSTER_META, FOLDER_TO_CLUSTER, get_folders_for_cluster
from lib.notes import Note, load_notes

logger = logging.getLogger("build_static_site")

# REL-E081 M2 — mirror of explorer/lib/cluster_page._GITHUB_BLOB_PREFIX so
# the static site's "see this note on GitHub" links honour the same
# ``EBERLIGHT_GITHUB_BLOB_PREFIX`` override.
_GITHUB_REPO_URL = os.environ.get(
    "EBERLIGHT_GITHUB_REPO_URL",
    "https://github.com/Denny-Hwang/synchrotron-data-analysis-notes",
)

# Cluster → file slug used in URLs.
CLUSTER_SLUG = {"discover": "discover", "explore": "explore", "build": "build"}

# Cluster → target page file path (relative to site root).
CLUSTER_PAGE = {cid: f"clusters/{slug}.html" for cid, slug in CLUSTER_SLUG.items()}

# Cluster → display order on the landing page.
CLUSTER_ORDER = ["discover", "explore", "build"]


# Interactive surfaces that ship as full Streamlit pages but degrade to
# read-only stubs on the static mirror. Order matters — it drives both
# the landing CTA grid and the loop in ``build()`` that emits the stubs.
# Each entry is keyed by the URL slug used in the static site (e.g.
# ``knowledge-graph.html``); ``streamlit_path`` is the path served by
# ``streamlit run explorer/app.py`` (Streamlit derives it from the
# pages/ filename).
INTERACTIVE_PAGES: tuple[dict[str, str], ...] = (
    {
        "slug": "knowledge-graph",
        "title": "Knowledge Graph",
        "icon": "🧠",
        "color_cluster": "discover",
        "streamlit_path": "/Knowledge_Graph",
        "summary": (
            "Cross-reference network of every modality, AI/ML method, paper, tool, "
            "Interactive-Lab recipe, and noise/artifact in one interactive view. "
            "Hover for details, click to navigate."
        ),
        "stat_line": ("100+ entities · 120+ edges · auto-extracted from notes + recipe.yaml."),
        "what_static_shows": (
            "The graph layout, layer toggles, and hover tooltips are Plotly-driven "
            "and need a running Python kernel. The static site cannot replay them."
        ),
    },
    {
        "slug": "experiment",
        "title": "Interactive Lab",
        "icon": "🧪",
        "color_cluster": "build",
        "streamlit_path": "/Experiment",
        "summary": (
            "Replay noise mitigation techniques from prior research on real bundled "
            "data — tune parameters, compare before/after, see PSNR/SSIM against a "
            "clean reference."
        ),
        "stat_line": (
            "14 recipes · 90+ real samples · Vo 2018 / Munch 2009 / Liu 2020 (TomoGAN) / "
            "Herraez 2002 / Donoho 1994 / Chambolle 2004 / Buades 2005 / van Dokkum 2001."
        ),
        "what_static_shows": (
            "The recipe pipelines (Sarepy stripe removal, wavelet-Fourier filter, "
            "TomoGAN-baseline wavelet shrinkage, 2-D phase unwrapping, TV / NLM / "
            "bilateral denoising, biharmonic inpainting, beam-hardening polynomial, "
            "L.A.Cosmic) execute server-side on user-tuned parameters; the recipe "
            "gallery on the Build cluster page is the static-site read-only summary."
        ),
    },
    {
        "slug": "troubleshooter",
        "title": "Troubleshooter",
        "icon": "🩺",
        "color_cluster": "explore",
        "streamlit_path": "/Troubleshooter",
        "summary": (
            "Symptom-driven decision tree over the noise catalog. Pick what you see "
            "in the data; get differential diagnoses with severity, conditions, and "
            "a one-click jump to the matching Lab recipe."
        ),
        "stat_line": ("11 symptom categories · 35 differential cases · before/after comparisons."),
        "what_static_shows": (
            "The decision tree, modality + severity filters, and ``?symptom=`` deep "
            "links rely on Streamlit query-param routing."
        ),
    },
    {
        "slug": "search",
        "title": "Search & Bibliography",
        "icon": "🔎",
        "color_cluster": "discover",
        "streamlit_path": "/Search",
        "summary": (
            "Global full-text search across every note plus a filterable BibTeX "
            "bibliography. Title-boosted relevance, prefix matching, deep links."
        ),
        "stat_line": ("<10 ms typical query · TF-IDF approx · 19 + 20 BibTeX entries indexed."),
        "what_static_shows": (
            "Live search needs a running index; the static mirror cannot host it. "
            "GitHub's repository search is a serviceable fallback while the app is "
            "offline."
        ),
    },
)

SITE_LAYOUT_CSS = """
/* === Static site layout (GitHub Pages mirror) ===
   REL-E080: design tokens are now CSS custom properties so this file
   and explorer/assets/styles.css speak the same vocabulary. A single
   palette change ripples to both surfaces.
   REL-E081 S5: ``prefers-color-scheme: dark`` re-binds the palette
   for OS-level dark mode; ``max-width: 1024px`` adds a tablet
   breakpoint between the existing 720px mobile and the desktop
   layout. */
:root {
    --color-primary: #0033A0;
    --color-primary-hover: #002270;
    --color-secondary: #0085C0;
    --color-accent: #D86510;
    --color-surface: #F5F5F5;
    --color-surface-alt: #FFFFFF;
    --color-surface-banner: #E8EEF6;
    --color-surface-hover: #F0F4FB;
    --color-text: #1A1A1A;
    --color-text-secondary: #555555;
    --color-text-muted: #888888;
    --color-text-inverse: #FFFFFF;
    --color-border: #E0E0E0;
    --color-border-soft: #E0E5EE;
    --color-border-strong: #C8D2E2;
    --color-success: #2E7D32;
    --color-success-bg: #E6F4EA;
    --color-success-fg: #1E6B33;
    --color-warning: #F57F17;
    --color-warning-bg: #FFF7DB;
    --color-warning-fg: #7A5A00;
    --color-error: #C62828;
    --color-error-bg: #FDECEA;
    --color-error-fg: #A82618;
    --radius-sm: 4px;
    --radius-md: 8px;
    --radius-lg: 16px;
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
        scroll-behavior: auto !important;
    }
}
@media (prefers-color-scheme: dark) {
    :root {
        --color-surface: #121821;
        --color-surface-alt: #1A2230;
        --color-surface-banner: #1B2A44;
        --color-surface-hover: #243047;
        --color-text: #E8ECF1;
        --color-text-secondary: #9DA8B8;
        --color-text-muted: #6F7B8A;
        --color-text-inverse: #FFFFFF;
        --color-border: #2A3447;
        --color-border-soft: #2A3447;
        --color-border-strong: #3A4860;
        --color-primary: #6FA0FF;
        --color-primary-hover: #8FB8FF;
        --color-secondary: #5DC4FF;
        --color-accent: #FF8A4C;
        --color-success: #6CC07A;
        --color-success-bg: #1F3A28;
        --color-success-fg: #B5E8BE;
        --color-warning: #F2B450;
        --color-warning-bg: #3A2E18;
        --color-warning-fg: #F8DCA0;
        --color-error: #FF6B65;
        --color-error-bg: #3A1B19;
        --color-error-fg: #FFC0BC;
    }
}
html, body { margin: 0; padding: 0; background: var(--color-surface); }
body {
    font-family: 'Source Sans 3', system-ui, -apple-system, 'Segoe UI',
                 Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: var(--color-text);
    line-height: 1.6;
}
a { color: var(--color-primary); }
.site-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 24px 48px 24px;
}
.site-container.narrow { max-width: 960px; }

/* (R11 I4 — styles.css now ships readable nav colors directly; the
   legacy ``pointer-events:none`` override is no longer needed.) */

/* Hero on the landing page */
.hero { text-align: center; padding: 48px 0; }
.hero h1 {
    color: #0033A0;
    font-size: 36px;
    margin: 0 0 12px 0;
    font-weight: 700;
}
.hero p {
    color: #555555;
    font-size: 18px;
    max-width: 720px;
    margin: 0 auto;
}

/* Cluster card grid on the landing page */
.cluster-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
    margin-top: 16px;
}
@media (max-width: 900px) {
    .cluster-grid { grid-template-columns: 1fr; }
}
.cluster-card {
    background: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 24px;
    min-height: 200px;
    border-top-width: 4px;
    border-top-style: solid;
    display: flex;
    flex-direction: column;
    text-decoration: none;
    color: inherit;
    transition: box-shadow 0.2s, transform 0.2s;
}
.cluster-card:hover {
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    transform: translateY(-2px);
}
.cluster-card h4 { margin: 0 0 12px 0; font-size: 20px; font-weight: 700; }
.cluster-card p { font-size: 14px; color: #555555; margin: 0 0 16px 0; flex: 1; }
.cluster-card .enter { font-weight: 600; font-size: 15px; }

/* Feature CTA cards on the landing — link out to interactive surfaces
   (Knowledge Graph, Lab, Troubleshooter, Search) that ship as static
   stubs on the Pages mirror. */
.feature-cta-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-top: 32px;
}
@media (max-width: 900px) {
    .feature-cta-grid { grid-template-columns: 1fr; }
}
.feature-cta {
    background: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-left-width: 4px;
    border-left-style: solid;
    border-radius: 8px;
    padding: 20px;
    text-decoration: none;
    color: inherit;
    transition: box-shadow 0.2s, transform 0.2s;
}
.feature-cta:hover {
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    transform: translateY(-2px);
}
.feature-cta h4 { margin: 0 0 8px 0; font-size: 18px; font-weight: 700; }
.feature-cta p { font-size: 14px; color: #555555; margin: 0 0 8px 0; }
.feature-cta p.stat { font-size: 13px; color: #888; margin-bottom: 0; }

/* Cluster / section hero heading */
.cluster-heading h1 { margin: 0 0 8px 0; font-size: 32px; }
.cluster-heading p {
    color: #555;
    font-size: 16px;
    margin: 0 0 24px 0;
    max-width: 800px;
}

/* Card grid on cluster pages */
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
}
.card-grid .eberlight-card {
    margin-bottom: 0;
    display: flex;
    flex-direction: column;
}
.card-grid .eberlight-card h4 a { color: #1A1A1A; text-decoration: none; }
.card-grid .eberlight-card h4 a:hover { color: #0033A0; text-decoration: underline; }
.card-grid .eberlight-card p {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.folder-section h2 {
    margin: 32px 0 12px 0;
    font-size: 20px;
    font-weight: 700;
    color: #1A1A1A;
    padding-bottom: 6px;
    border-bottom: 1px solid #E0E0E0;
}

/* Note detail two-column layout */
.note-layout {
    display: grid;
    grid-template-columns: minmax(0, 3fr) minmax(0, 1fr);
    gap: 32px;
    align-items: start;
}
@media (max-width: 900px) {
    .note-layout { grid-template-columns: 1fr; }
}
.note-main h1 {
    font-size: 32px;
    margin: 0 0 16px 0;
    color: #1A1A1A;
}
.note-main h2 { font-size: 24px; margin-top: 28px; }
.note-main h3 { font-size: 20px; margin-top: 24px; }
.note-main p, .note-main li { font-size: 16px; }
.note-main pre {
    padding: 12px 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.5;
}
.note-main code {
    font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.92em;
}
.note-main table {
    border-collapse: collapse;
    margin: 16px 0;
    width: 100%;
    font-size: 14px;
}
.note-main th, .note-main td {
    border: 1px solid #E0E0E0;
    padding: 8px 12px;
    text-align: left;
}
.note-main th { background: #F5F5F5; font-weight: 600; }
.note-main img { max-width: 100%; height: auto; }
.note-main blockquote {
    border-left: 4px solid #00A3E0;
    margin: 16px 0;
    padding: 4px 16px;
    color: #555;
    background: #F5F5F5;
}

/* Empty state banner (like Streamlit st.info) */
.info-box {
    background: var(--color-surface-banner);
    border-left: 4px solid var(--color-primary);
    padding: 12px 16px;
    border-radius: var(--radius-sm);
    margin: 16px 0;
    color: var(--color-text);
}

/* REL-E080 — disclaimer banner under the landing hero. */
.disclaimer-banner {
    max-width: 760px;
    margin: 8px auto 24px auto;
    padding: 10px 16px;
    background: var(--color-surface-banner);
    border-left: 4px solid var(--color-primary);
    border-radius: var(--radius-sm);
    color: var(--color-text);
    font-size: 13px;
    line-height: 1.5;
    text-align: left;
}
.disclaimer-banner b { color: var(--color-primary); }

/* REL-E080 — onboarding scenario picker on the landing. */
.eberlight-onboarding {
    background: var(--color-surface-alt);
    border: 1px solid var(--color-border);
    border-left: 4px solid var(--color-primary);
    border-radius: var(--radius-md);
    padding: 20px 24px;
    margin: 24px 0;
}
.eberlight-onboarding h3 { font-size: 16px; margin: 0 0 4px 0; color: var(--color-primary); }
.eberlight-onboarding p.sub {
    font-size: 13px;
    color: var(--color-text-secondary);
    margin: 0 0 12px 0;
}
.eberlight-onboarding .scenarios {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}
.eberlight-onboarding .scenario-card {
    display: block;
    text-decoration: none;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: 12px 14px;
    color: var(--color-text);
    transition: border-color 0.15s, background 0.15s, transform 0.15s;
}
.eberlight-onboarding .scenario-card:hover,
.eberlight-onboarding .scenario-card:focus-visible {
    border-color: var(--color-primary);
    background: var(--color-surface-banner);
    transform: translateY(-1px);
    outline: none;
}
.eberlight-onboarding .scenario-card .icon { font-size: 18px; margin-bottom: 6px; }
.eberlight-onboarding .scenario-card .title {
    font-size: 14px; font-weight: 700;
    color: var(--color-primary); margin-bottom: 4px;
}
.eberlight-onboarding .scenario-card .body {
    font-size: 12px; color: var(--color-text-secondary); line-height: 1.4;
}
@media (max-width: 720px) {
    .eberlight-onboarding .scenarios { grid-template-columns: 1fr; }
}

/* REL-E080 — cluster orientation stats line + tagline + first steps. */
.cluster-heading .stats-line {
    color: var(--color-text-muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 12px 0;
}
.cluster-heading .tagline {
    color: var(--color-text-secondary);
    font-size: 14px;
    margin: 0 0 8px 0;
}
.cluster-heading .first-steps {
    color: var(--color-text-muted);
    font-size: 13px;
    margin: 0 0 16px 0;
}

/* REL-E080 — glossary auto-link (mirrors explorer/assets/styles.css).
   REL-E081 B3 — focus styles match the Streamlit side. */
abbr.eberlight-glossary {
    text-decoration: underline dotted var(--color-text-muted);
    text-underline-offset: 3px;
    cursor: help;
    border: 0;
}
abbr.eberlight-glossary:focus,
abbr.eberlight-glossary:focus-visible {
    outline: 2px solid var(--color-primary);
    outline-offset: 2px;
    background: var(--color-surface-banner);
    border-radius: 2px;
}

/* REL-E081 B4 — small "(needs local Streamlit)" suffix on landing
   scenario cards that link to interactive stubs. */
.scenario-card .needs-streamlit {
    display: inline-block;
    margin-left: 6px;
    font-size: 11px;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* REL-E081 S2 — table/cards layout toggle on cluster pages. */
.eberlight-layout-toggle {
    display: flex;
    gap: 8px;
    margin: 0 0 16px 0;
}

/* REL-E081 S5 — tablet breakpoint. Halve the 3-col onboarding grid. */
@media (max-width: 1024px) and (min-width: 901px) {
    .eberlight-onboarding .scenarios { grid-template-columns: 1fr 1fr; }
}
""".strip()


def _rel(from_path: str, to_path: str) -> str:
    """Compute a relative URL from one site-relative path to another.

    Both arguments are POSIX-style paths relative to the site root (no
    leading slash).  Returns a relative URL suitable for an href.
    """
    from_parts = from_path.split("/")[:-1]  # strip filename
    to_parts = to_path.split("/")
    # Common prefix length
    i = 0
    while i < len(from_parts) and i < len(to_parts) - 1 and from_parts[i] == to_parts[i]:
        i += 1
    ups = [".."] * (len(from_parts) - i)
    tail = to_parts[i:]
    return "/".join(ups + tail) if ups or tail else "."


def _note_output_path(note: Note) -> str:
    """Return the site-relative HTML path for a note (POSIX-style)."""
    rel = note.path.relative_to(_REPO_ROOT).with_suffix(".html")
    return "notes/" + rel.as_posix()


def _md_link_rewrite(body_html: str) -> str:
    """Rewrite href="...md" links to href="...html" inside rendered HTML.

    Conservative: only rewrites href attributes that end in .md (with or
    without a fragment), and never touches absolute URLs that would confuse
    the rewrite.  Absolute URLs with .md in them are rare, and we still
    want them rewritten when they point to the repo.
    """
    return re.sub(
        r'(href="[^"]*?)\.md((?:#[^"]*)?")',
        r"\1.html\2",
        body_html,
    )


def _git_iso_date() -> str:
    """Best-effort HEAD commit date (YYYY-MM-DD)."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=_REPO_ROOT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()[:10]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return datetime.now().strftime("%Y-%m-%d")


def _folder_label(folder: str) -> str:
    """Human-friendly folder label, e.g. '02_xray_modalities' → 'Xray Modalities'."""
    return folder.split("_", 1)[1].replace("_", " ").title() if "_" in folder else folder


# ---------------------------------------------------------------------------
# HTML fragments
# ---------------------------------------------------------------------------


def _header_html(page_path: str, active_cluster: str | None = None) -> str:
    """Site header with logo + 3 cluster links. Mirrors explorer/components/header.py."""

    def link(cid: str, label: str) -> str:
        href = _rel(page_path, CLUSTER_PAGE[cid])
        cls = "active" if active_cluster == cid else ""
        return f'<a href="{href}" class="{cls}">{label}</a>'

    home_href = _rel(page_path, "index.html")
    return f"""
<div class="eberlight-header">
    <div class="eberlight-header-brand">
        <div class="eberlight-header-logo">eB</div>
        <a href="{home_href}" style="text-decoration:none;">
            <span class="eberlight-header-title">eBERlight Explorer</span>
        </a>
    </div>
    <nav class="eberlight-header-nav" aria-label="Main navigation">
        {link("discover", "Discover")}
        {link("explore", "Explore")}
        {link("build", "Build")}
    </nav>
</div>
""".strip()


def _breadcrumb_html(page_path: str, items: list[tuple[str, str | None]]) -> str:
    """Mirrors explorer/components/breadcrumb.py but with real anchor targets."""
    parts: list[str] = []
    for label, target in items:
        esc = html_escape_mod.escape(label)
        if target is None:
            parts.append(f'<span class="current">{esc}</span>')
        else:
            href = _rel(page_path, target) if not target.startswith("http") else target
            parts.append(f'<a href="{href}">{esc}</a>')
    sep = '<span class="separator">&gt;</span>'
    return f'<nav class="eberlight-breadcrumb" aria-label="Breadcrumb">{sep.join(parts)}</nav>'


def _footer_html(last_updated: str) -> str:
    """Mirrors explorer/components/footer.py.

    REL-E080: footer copy reframed for the personal-research project.
    The static mirror previously implied institutional affiliation
    ("This research used resources…"); the corrected text acknowledges
    the upstream data sources without claiming sponsorship.
    """
    return f"""
<div class="eberlight-footer">
    <p>
        <b>Personal research / learning project — not affiliated with
        ANL, APS, or DOE.</b> The bundled sample data is redistributed
        under the upstream permissive licenses preserved verbatim in
        <code>10_interactive_lab/LICENSES/</code>; please do not
        deploy this app publicly without first contacting the original
        data sources.
    </p>
    <p>
        The notes use synchrotron X-ray data analysis as a representative
        case study, with the eBERlight program at the Advanced Photon
        Source as one source-material reference. The visual style is
        ANL/APS-inspired but no official branding is claimed.
    </p>
    <div class="eberlight-footer-links">
        <a href="https://www.aps.anl.gov/" target="_blank" rel="noopener">APS (reference)</a>
        <a href="https://eberlight.aps.anl.gov/" target="_blank" rel="noopener">eBERlight (reference)</a>
        <a href="{_GITHUB_REPO_URL}" target="_blank" rel="noopener">Repository</a>
    </div>
    <div class="eberlight-footer-updated">Last updated: {last_updated}</div>
</div>
""".strip()


def _card_html(title: str, summary: str, tags: list[str], href: str) -> str:
    """Mirrors explorer/components/card.py."""
    tags_html = "".join(
        f'<span class="eberlight-tag">{html_escape_mod.escape(t)}</span>' for t in tags[:5]
    )
    return f"""
<div class="eberlight-card">
    <h4><a href="{href}">{html_escape_mod.escape(title)}</a></h4>
    <p>{html_escape_mod.escape(summary)}</p>
    <div>{tags_html}</div>
</div>
""".strip()


def _related_views_html(page_path: str, note: Note) -> str:
    """Render the "Related views" aside on a note page (REL-E081 S1).

    Mirrors :func:`explorer.lib.cluster_page._build_related_views`:
    every note links to KG + Troubleshooter stubs and to a modality-
    filtered cluster view, so a reader doesn't have to bounce home
    to reach the power surfaces.
    """
    items: list[tuple[str, str]] = [
        ("🧠 Knowledge Graph", _rel(page_path, _interactive_stub_page_path("knowledge-graph"))),
        ("🩺 Troubleshooter", _rel(page_path, _interactive_stub_page_path("troubleshooter"))),
    ]
    if note.modality:
        cluster_href = (
            _rel(page_path, CLUSTER_PAGE[note.cluster]) + f"?tag={quote(note.modality, safe='')}"
        )
        items.append(
            (
                f"📚 Other {html_escape_mod.escape(note.modality.replace('_', ' '))} notes",
                cluster_href,
            )
        )
    items.append(
        ("🔎 Search across all notes", _rel(page_path, _interactive_stub_page_path("search")))
    )

    rows = "".join(
        f'<div style="font-size:13px;margin:4px 0;">'
        f'<a href="{href}" style="color:var(--color-primary);text-decoration:none;">'
        f"{label}</a></div>"
        for label, href in items
    )
    return (
        '<aside aria-label="Related views" '
        'style="background:var(--color-surface-alt);'
        "border:1px solid var(--color-border);"
        'border-radius:var(--radius-md);padding:16px;margin-top:16px;">'
        '<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        'letter-spacing:0.5px;color:var(--color-text-secondary);margin-bottom:8px;">'
        "Related views</div>"
        f"{rows}"
        "</aside>"
    )


def _metadata_panel_html(note: Note) -> str:
    """Mirrors explorer/components/note_view.py._render_metadata_panel."""
    sections: list[str] = []

    def section(label: str, body: str) -> str:
        return (
            '<div style="margin-bottom:20px;">'
            '<div style="font-size:12px;font-weight:600;text-transform:uppercase;'
            'letter-spacing:0.5px;color:#555;margin-bottom:8px;">'
            f"{label}</div>{body}</div>"
        )

    if note.beamline:
        badges = " ".join(
            f'<span style="background:#0033A0;color:white;padding:4px 12px;'
            f'border-radius:12px;font-size:12px;font-weight:600;">'
            f"{html_escape_mod.escape(bl)}</span>"
            for bl in note.beamline
        )
        sections.append(
            section(
                "Beamlines", f'<div style="display:flex;gap:6px;flex-wrap:wrap;">{badges}</div>'
            )
        )

    if note.modality:
        sections.append(
            section(
                "Modality",
                f'<span class="eberlight-tag">{html_escape_mod.escape(note.modality)}</span>',
            )
        )

    if note.tags:
        tags_html = " ".join(
            f'<span class="eberlight-tag">{html_escape_mod.escape(t)}</span>' for t in note.tags
        )
        sections.append(section("Tags", tags_html))

    if note.related_publications:
        links = "".join(
            f'<div style="font-size:14px;color:#0033A0;margin-bottom:4px;">'
            f"{html_escape_mod.escape(p)}</div>"
            for p in note.related_publications
        )
        sections.append(section("Publications", links))

    if note.related_tools:
        links = "".join(
            f'<div style="font-size:14px;color:#0033A0;margin-bottom:4px;">'
            f"{html_escape_mod.escape(t)}</div>"
            for t in note.related_tools
        )
        sections.append(section("Related Tools", links))

    # Always surface cluster + source path so users can jump back to the repo.
    cluster_meta = CLUSTER_META.get(note.cluster)
    if cluster_meta:
        sections.append(
            section(
                "Cluster",
                f'<span class="eberlight-tag" style="background:{cluster_meta["color"]}1A;'
                f'color:{cluster_meta["color"]};border-color:{cluster_meta["color"]}33;">'
                f"{html_escape_mod.escape(cluster_meta['name'])}</span>",
            )
        )

    rel_source = note.path.relative_to(_REPO_ROOT).as_posix()
    sections.append(
        section(
            "Source",
            f'<a href="https://github.com/Denny-Hwang/synchrotron-data-analysis-notes/'
            f'blob/main/{rel_source}" target="_blank" rel="noopener" '
            f'style="font-size:13px;word-break:break-all;">{html_escape_mod.escape(rel_source)}</a>',
        )
    )

    if not sections:
        return ""
    return (
        '<aside aria-label="Note metadata" style="background:#FFFFFF;'
        'border:1px solid #E0E0E0;border-radius:8px;padding:24px;">'
        + "".join(sections)
        + "</aside>"
    )


# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------


def _page_shell(
    page_path: str,
    title: str,
    body: str,
    *,
    active_cluster: str | None = None,
    extra_head: str = "",
    narrow: bool = False,
) -> str:
    css_href = _rel(page_path, "assets/styles.css")
    container_cls = "site-container narrow" if narrow else "site-container"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_escape_mod.escape(title)}</title>
<meta name="description" content="eBERlight Explorer — static mirror of the Streamlit app.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{css_href}">
{extra_head}
</head>
<body>
<div class="{container_cls}">
    {_header_html(page_path, active_cluster=active_cluster)}
    <main>
{body}
    </main>
    {_footer_html(_git_iso_date())}
</div>
</body>
</html>
"""


def _interactive_stub_page_path(slug: str) -> str:
    """URL path for an interactive-stub page (e.g. ``knowledge-graph.html``)."""
    return f"{slug}.html"


def _interactive_cta_card_html(page_path: str, entry: dict[str, str]) -> str:
    """One CTA card on the landing pointing to an interactive-stub page."""
    color = CLUSTER_META[entry["color_cluster"]]["color"]
    href = _rel(page_path, _interactive_stub_page_path(entry["slug"]))
    return (
        f'<a class="feature-cta" href="{href}" '
        f'style="border-left-color: {color};">'
        f'<h4 style="color: {color};">'
        f"{entry['icon']} {html_escape_mod.escape(entry['title'])}"
        f"</h4>"
        f"<p>{html_escape_mod.escape(entry['summary'])}</p>"
        f'<p class="stat">{html_escape_mod.escape(entry["stat_line"])}</p>'
        f"</a>"
    )


def _render_interactive_stub(out_dir: Path, entry: dict[str, str]) -> None:
    """Emit a read-only stub for an interactive page (FR-022 pattern).

    The stub keeps the static site URL surface area aligned with the
    Streamlit app so external links and the header nav resolve. The
    body reproduces the page's value proposition and tells the reader
    how to launch the live version.
    """
    page_path = _interactive_stub_page_path(entry["slug"])
    color = CLUSTER_META[entry["color_cluster"]]["color"]
    title_html = f"{entry['icon']} {html_escape_mod.escape(entry['title'])}"
    body = f"""
    {_breadcrumb_html(page_path, [("Home", "index.html"), (entry["title"], None)])}
    <section class="cluster-heading">
        <h1 style="color: {color};">{title_html}</h1>
        <p>{html_escape_mod.escape(entry["summary"])}</p>
    </section>
    <div class="info-box" style="border-left: 4px solid {color};">
        <p><b>Interactive feature.</b>
        {html_escape_mod.escape(entry["what_static_shows"])}</p>
        <p style="margin-bottom:0;">
        Run the Streamlit app to use the live version:</p>
        <pre><code>git clone https://github.com/Denny-Hwang/synchrotron-data-analysis-notes.git
cd synchrotron-data-analysis-notes
pip install -r explorer/requirements.txt
streamlit run explorer/app.py
# then open {html_escape_mod.escape(entry["streamlit_path"])}</code></pre>
    </div>
    <section class="folder-section">
        <h2>What it does</h2>
        <p>{html_escape_mod.escape(entry["stat_line"])}</p>
    </section>
"""
    html = _page_shell(
        page_path,
        f"{entry['title']} — eBERlight Explorer",
        body,
        narrow=True,
    )
    target = out_dir / page_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")


_LANDING_SCENARIOS = (
    {
        "icon": "🔬",
        "title": "I have a sample to analyse",
        "body": (
            "Map sample → modality → method → tool. Best starting point when you "
            "know what you're imaging but not how."
        ),
        "target_cluster": "explore",
    },
    {
        "icon": "🩺",
        "title": "I see something weird in my data",
        "body": (
            "Symptom-driven troubleshooter walks the differential diagnoses for "
            "ring patterns, streaks, noise, blur, dead pixels, and more."
        ),
        "target_slug": "troubleshooter",
    },
    {
        "icon": "🧪",
        "title": "I want to try a noise-mitigation method hands-on",
        "body": (
            "Replay 14 bundled recipes on real samples — slide parameters, watch "
            "PSNR/SSIM move, see the |Δ| diff panel."
        ),
        "target_slug": "experiment",
    },
)


def _scenario_card_html(page_path: str, scenario: dict[str, str]) -> str:
    """One onboarding scenario card on the landing (REL-E080).

    REL-E081 B4: scenarios that route to an interactive-only stub
    (Troubleshooter, Lab) carry a "(needs local Streamlit)" suffix so
    the static-site reader isn't surprised when they click and get the
    "run Streamlit locally" page.
    """
    if "target_cluster" in scenario:
        href = _rel(page_path, CLUSTER_PAGE[scenario["target_cluster"]])
        suffix = ""
    else:
        href = _rel(page_path, _interactive_stub_page_path(scenario["target_slug"]))
        suffix = '<span class="needs-streamlit">(needs local Streamlit)</span>'
    return (
        f'<a class="scenario-card" href="{href}">'
        f'<div class="icon" aria-hidden="true">{scenario["icon"]}</div>'
        f'<div class="title">{html_escape_mod.escape(scenario["title"])} {suffix}</div>'
        f'<div class="body">{html_escape_mod.escape(scenario["body"])}</div>'
        f"</a>"
    )


def _render_landing(out_dir: Path) -> None:
    page_path = "index.html"
    cards: list[str] = []
    for cid in CLUSTER_ORDER:
        meta = CLUSTER_META[cid]
        href = _rel(page_path, CLUSTER_PAGE[cid])
        cards.append(
            f'<a class="cluster-card" href="{href}" '
            f'style="border-top-color: {meta["color"]};">'
            f'<h4 style="color: {meta["color"]};">{html_escape_mod.escape(meta["name"])}</h4>'
            f"<p>{html_escape_mod.escape(meta['description'])}</p>"
            f'<span class="enter" style="color: {meta["color"]};">Enter →</span>'
            f"</a>"
        )
    feature_cards = "\n".join(
        _interactive_cta_card_html(page_path, entry) for entry in INTERACTIVE_PAGES
    )
    scenario_cards = "\n".join(_scenario_card_html(page_path, s) for s in _LANDING_SCENARIOS)
    body = f"""
    {_breadcrumb_html(page_path, [("Home", None)])}
    <section class="hero">
        <h1>eBERlight Research Explorer</h1>
        <p>Personal study notes on synchrotron data analysis</p>
    </section>
    <div class="disclaimer-banner">
        <b>Personal research / learning workspace</b> — ANL/APS-inspired but
        <b>not affiliated with</b> ANL, APS, or DOE. The bundled sample data is
        redistributed under upstream permissive licenses; please do not host
        this app publicly without contacting the original data sources first.
    </div>
    <section class="eberlight-onboarding" aria-label="Choose your scenario">
        <h3>New here? Pick your scenario</h3>
        <p class="sub">Three high-leverage entry points — or browse the three
        clusters below if you already know where you're heading.</p>
        <div class="scenarios">{scenario_cards}</div>
    </section>
    <section class="cluster-grid">
        {"".join(cards)}
    </section>
    <section class="feature-cta-grid">
        {feature_cards}
    </section>
"""
    html = _page_shell(page_path, "eBERlight Explorer", body)
    (out_dir / page_path).write_text(html, encoding="utf-8")


def _recipe_card_html(recipe: Recipe) -> str:
    """Render one recipe.yaml as a card on the Build cluster page.

    The static site cannot run pipelines — the card is a read-only summary
    pointing readers to the Streamlit Lab to actually execute.
    """
    desc_first_para = recipe.description.split("\n\n", 1)[0] if recipe.description else ""
    # Strip markdown emphasis from the excerpt for cleaner display.
    desc_clean = re.sub(r"[*_`]+", "", desc_first_para).strip()
    if len(desc_clean) > 280:
        desc_clean = desc_clean[:280].rstrip() + "…"

    primary_ref = recipe.references[0] if recipe.references else None
    ref_html = ""
    if primary_ref:
        cite = (
            f"{html_escape_mod.escape(primary_ref.authors)} "
            f"({primary_ref.year}). "
            f"{html_escape_mod.escape(primary_ref.title)}."
        )
        if primary_ref.doi:
            cite += (
                f' <a href="https://doi.org/{html_escape_mod.escape(primary_ref.doi)}"'
                ' target="_blank" rel="noopener">DOI</a>'
            )
        ref_html = f'<p class="recipe-cite" style="font-size:12px;color:#888;">{cite}</p>'

    badge_color = CLUSTER_META["build"]["color"]
    return (
        '<div class="eberlight-card" '
        f'style="border-left: 3px solid {badge_color};">'
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'gap:8px;margin-bottom:8px;">'
        f'<h4 style="margin:0;">{html_escape_mod.escape(recipe.title)}</h4>'
        f'<span style="font-size:11px;color:#fff;background:{badge_color};'
        'padding:2px 8px;border-radius:10px;white-space:nowrap;">'
        f"{html_escape_mod.escape(recipe.modality)}</span>"
        "</div>"
        f'<p style="font-size:13px;color:#555;margin:0 0 8px 0;">{html_escape_mod.escape(desc_clean)}</p>'
        '<p style="font-size:12px;color:#666;margin:0 0 4px 0;">'
        f"<b>{len(recipe.samples)}</b> bundled sample"
        f"{'s' if len(recipe.samples) != 1 else ''} · "
        f"<b>{len(recipe.parameters)}</b> tunable parameter"
        f"{'s' if len(recipe.parameters) != 1 else ''} · "
        f"<code>{html_escape_mod.escape(recipe.recipe_id)}</code>"
        "</p>"
        f"{ref_html}"
        "</div>"
    )


def _recipe_gallery_html() -> str:
    """Render the full recipe gallery shown on the Build cluster page."""
    experiments_root = _REPO_ROOT / "experiments"
    if not experiments_root.exists():
        return ""
    recipes = load_recipes(experiments_root)
    if not recipes:
        return ""
    cards = "\n".join(_recipe_card_html(r) for r in recipes)
    return (
        '<section class="folder-section">'
        "<h2>Interactive Lab — Recipes</h2>"
        '<p style="color:#666;font-size:14px;margin:-8px 0 12px 0;">'
        f"{len(recipes)} bundled noise-mitigation recipe"
        f"{'s' if len(recipes) != 1 else ''}. "
        "Run interactively in the Streamlit Explorer "
        "(see <code>10_interactive_lab/README.md</code>); the static site "
        "cannot execute pipelines."
        "</p>"
        f'<div class="card-grid">{cards}</div>'
        "</section>"
    )


# REL-E080 — per-cluster orientation copy. Mirrors
# explorer/lib/cluster_page.py::_CLUSTER_TAGLINE so a static-site
# visitor lands on the same "what's here / when to use it / where to
# start" framing as the Streamlit visitor.
_CLUSTER_TAGLINE: dict[str, dict[str, str]] = {
    "discover": {
        "tagline": (
            "Start here when you want context on the program, its facilities and partners, "
            "or you're hunting for a glossary term or a paper reference."
        ),
        "good_first_steps": (
            "Try the <b>Program Overview</b> folder for the BER mission and APS facility, "
            "or <b>References</b> for the glossary + bibliography."
        ),
    },
    "explore": {
        "tagline": (
            "Start here when you have a sample and need to choose a modality, "
            "compare AI/ML methods, or diagnose a weird-looking image."
        ),
        "good_first_steps": (
            "Try <b>X-Ray Modalities</b> to map sample → technique, "
            "<b>AI/ML Methods</b> for method-by-method tradeoffs, or "
            "<b>Noise Catalog</b> when the artefact is already visible in your data."
        ),
    },
    "build": {
        "tagline": (
            "Start here when you have data in hand and need a tool, a schema, "
            "or a recipe to apply to it."
        ),
        "good_first_steps": (
            "Try <b>Tools and Code</b> for reverse-engineered tool internals, "
            "<b>Data Structures</b> for HDF5 schemas, or jump straight to the "
            "<b>Interactive Lab</b> to replay a noise-mitigation recipe."
        ),
    },
}


def _cluster_orientation_html(cluster_id: str, cluster_notes: list[Note]) -> str:
    """Render the static-site cluster header: h1 + stats + tagline + first-steps.

    Mirrors explorer/lib/cluster_page.py::_render_cluster_orientation so
    the static and Streamlit views feel identical.
    """
    meta = CLUSTER_META[cluster_id]
    extra = _CLUSTER_TAGLINE.get(cluster_id, {})
    folders = {n.folder for n in cluster_notes}
    note_count = len(cluster_notes)
    folder_count = len(folders)
    last_updated = max(
        (n.last_reviewed for n in cluster_notes if getattr(n, "last_reviewed", None)),
        default=None,
    )
    stats_bits = [
        f"<b>{note_count}</b> note{'s' if note_count != 1 else ''}",
        f"<b>{folder_count}</b> folder{'s' if folder_count != 1 else ''}",
    ]
    if last_updated:
        stats_bits.append(f"last reviewed <b>{html_escape_mod.escape(last_updated)}</b>")
    stats_html = " · ".join(stats_bits)

    parts: list[str] = [
        f'<h1 style="color: {meta["color"]};">{html_escape_mod.escape(meta["name"])}</h1>',
        f'<p class="stats-line">{stats_html}</p>',
        f"<p>{html_escape_mod.escape(meta['description'])}</p>",
    ]
    if extra.get("tagline"):
        parts.append(f'<p class="tagline">{extra["tagline"]}</p>')
    if extra.get("good_first_steps"):
        parts.append(f'<p class="first-steps">💡 {extra["good_first_steps"]}</p>')
    return '<section class="cluster-heading">' + "".join(parts) + "</section>"


def _cluster_layout_toggle_html(page_path: str, active: str) -> str:
    """Render the cluster-page 📋 Table / 🃏 Cards pill row (REL-E081 S2).

    On the static site we generate **two output files** per cluster
    (``discover.html`` + ``discover-cards.html``) so the toggle is a
    plain link — no JS required. The active pill is solid, the other
    outlined; both use the existing ``.eberlight-chip`` styles.
    """
    base_name = page_path.rsplit("/", 1)[-1]
    if base_name.endswith("-cards.html"):
        table_filename = base_name.replace("-cards.html", ".html")
        cards_filename = base_name
    else:
        table_filename = base_name
        cards_filename = base_name.replace(".html", "-cards.html")

    def _pill(label: str, href: str, is_active: bool) -> str:
        cls = "eberlight-chip active" if is_active else "eberlight-chip"
        aria = "true" if is_active else "false"
        return f'<a href="{href}" class="{cls}" aria-pressed="{aria}">{label}</a>'

    return (
        '<div class="eberlight-layout-toggle" role="tablist" aria-label="View layout">'
        + _pill("📋 Table", table_filename, active == "table")
        + _pill("🃏 Cards", cards_filename, active == "cards")
        + "</div>"
    )


def _render_cluster(
    out_dir: Path,
    cluster_id: str,
    notes: list[Note],
    *,
    group_by_folder: bool,
) -> None:
    page_path = CLUSTER_PAGE[cluster_id]
    meta = CLUSTER_META[cluster_id]
    cluster_notes = [n for n in notes if n.folder in set(get_folders_for_cluster(cluster_id))]

    def card_for(note: Note) -> str:
        summary = note.description or note.body[:200].strip().replace("\n", " ")
        href = _rel(page_path, _note_output_path(note))
        return _card_html(note.title, summary, note.tags, href)

    if not cluster_notes:
        content = '<div class="info-box">No notes found in this cluster.</div>'
    elif group_by_folder:
        blocks: list[str] = []
        for folder, folder_notes_iter in groupby(cluster_notes, key=lambda n: n.folder):
            folder_notes = list(folder_notes_iter)
            cards = "\n".join(card_for(n) for n in folder_notes)
            blocks.append(
                f'<section class="folder-section">'
                f"<h2>{html_escape_mod.escape(_folder_label(folder))}</h2>"
                f'<div class="card-grid">{cards}</div>'
                f"</section>"
            )
        content = "\n".join(blocks)
    else:
        cards = "\n".join(card_for(n) for n in cluster_notes)
        content = f'<div class="card-grid">{cards}</div>'

    # Build cluster gets an extra "Interactive Lab — Recipes" gallery (ADR-008).
    if cluster_id == "build":
        gallery = _recipe_gallery_html()
        if gallery:
            content = gallery + "\n" + content

    toggle = _cluster_layout_toggle_html(page_path, active="table")
    body = f"""
    {_breadcrumb_html(page_path, [("Home", "index.html"), (meta["name"], None)])}
    {_cluster_orientation_html(cluster_id, cluster_notes)}
    {toggle}
    {content}
"""
    html = _page_shell(
        page_path,
        f"{meta['name']} — eBERlight Explorer",
        body,
        active_cluster=cluster_id,
    )
    target = out_dir / page_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")

    # REL-E081 S2 — also emit a sibling cards layout. Reuses the cluster
    # orientation header + recipe gallery (Build only) and renders notes
    # as a card grid instead of the dataframe-style compare table.
    _render_cluster_cards(out_dir, cluster_id, cluster_notes, meta)


def _render_cluster_cards(
    out_dir: Path,
    cluster_id: str,
    cluster_notes: list[Note],
    meta: dict[str, str],
) -> None:
    """Emit the cards-layout sibling page (e.g. ``clusters/discover-cards.html``)."""
    page_path = CLUSTER_PAGE[cluster_id].replace(".html", "-cards.html")

    def card_for(note: Note) -> str:
        summary = note.description or note.body[:200].strip().replace("\n", " ")
        href = _rel(page_path, _note_output_path(note))
        return _card_html(note.title, summary, note.tags, href)

    if not cluster_notes:
        content = '<div class="info-box">No notes found in this cluster.</div>'
    else:
        cards = "\n".join(card_for(n) for n in cluster_notes)
        content = f'<div class="card-grid">{cards}</div>'

    if cluster_id == "build":
        gallery = _recipe_gallery_html()
        if gallery:
            content = gallery + "\n" + content

    toggle = _cluster_layout_toggle_html(page_path, active="cards")
    body = f"""
    {_breadcrumb_html(page_path, [("Home", "index.html"), (meta["name"], None)])}
    {_cluster_orientation_html(cluster_id, cluster_notes)}
    {toggle}
    {content}
"""
    html = _page_shell(
        page_path,
        f"{meta['name']} (cards) — eBERlight Explorer",
        body,
        active_cluster=cluster_id,
    )
    target = out_dir / page_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")


# Match ``‌```mermaid`` fenced blocks in the **raw markdown body** (not in
# the rendered HTML — codehilite syntax-highlights every fenced block,
# which would mangle the diagram source).
_MERMAID_MD_BLOCK = re.compile(
    r"```mermaid[ \t]*\n(?P<code>.*?)\n```",
    re.DOTALL,
)


def _replace_mermaid_blocks(body_html: str) -> tuple[str, bool]:
    """Convert pre-rendered Mermaid placeholder blocks to live ``<div>``s.

    The render path is split into three steps to avoid codehilite
    rewriting the diagram source as syntax-highlighted code:

    1. :func:`_extract_mermaid_blocks` lifts every fenced
       ``‌```mermaid`` block out of the raw markdown body and replaces
       each with an HTML-safe placeholder (``<!--MERMAID:N-->``).
    2. The cleaned body goes through ``markdown.markdown(...)`` as
       usual.
    3. This function swaps every placeholder for a real
       ``<div class="mermaid">…</div>`` carrying the original source,
       and signals via the second return value whether the page needs
       the runtime ``<script>`` tag.
    """
    placeholder_re = re.compile(r"<!--MERMAID:(?P<idx>\d+):(?P<code>[^-]+(?:-(?!-)[^-]*)*)-->")
    has_mermaid = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal has_mermaid
        has_mermaid = True
        from base64 import b64decode

        code = b64decode(match.group("code")).decode("utf-8")
        return f'<div class="mermaid">{code}</div>'

    return placeholder_re.sub(_replace, body_html), has_mermaid


def _extract_mermaid_blocks(body: str) -> str:
    """Pre-process the **raw markdown body** to lift Mermaid blocks aside.

    Replaces every ``‌```mermaid`` fenced block with a base64-encoded
    HTML comment placeholder. The caller renders the cleaned body via
    ``markdown.markdown(...)``, then runs :func:`_replace_mermaid_blocks`
    on the rendered HTML to swap each placeholder for a live
    ``<div class="mermaid">``. Base64 is used because the diagram
    source can contain ``-->`` arrows that would otherwise close an
    HTML comment prematurely.
    """
    from base64 import b64encode

    counter = [0]

    def _swap(match: re.Match[str]) -> str:
        encoded = b64encode(match.group("code").encode("utf-8")).decode("ascii")
        idx = counter[0]
        counter[0] += 1
        # Two newlines isolate the placeholder so markdown treats it as
        # its own block, not part of the surrounding paragraph.
        return f"\n\n<!--MERMAID:{idx}:{encoded}-->\n\n"

    return _MERMAID_MD_BLOCK.sub(_swap, body)


_MERMAID_HEAD = """
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  document.addEventListener('DOMContentLoaded', () => {
    if (window.mermaid) {
      mermaid.initialize({
        startOnLoad: true,
        theme: 'default',
        securityLevel: 'loose',
        flowchart: { htmlLabels: true, curve: 'basis' }
      });
    }
  });
</script>
"""


def _render_note(out_dir: Path, note: Note, highlight_css: str) -> None:
    page_path = _note_output_path(note)
    cluster_meta = CLUSTER_META.get(note.cluster, CLUSTER_META["explore"])

    # Step 1 — lift Mermaid blocks aside before codehilite mangles them.
    body_with_placeholders = _extract_mermaid_blocks(note.body)
    body_html = markdown.markdown(
        body_with_placeholders,
        extensions=["fenced_code", "tables", "toc", "codehilite"],
        extension_configs={
            "codehilite": {
                "css_class": "highlight",
                "linenums": False,
                # R11 I1 — same fix as note_view._md_to_html: never guess
                # a lexer for unlabeled code blocks, otherwise ASCII tree
                # diagrams become a span soup.
                "guess_lang": False,
            }
        },
    )
    body_html = _md_link_rewrite(body_html)
    # Step 2 — swap each placeholder for a real ``<div class="mermaid">``.
    body_html, has_mermaid = _replace_mermaid_blocks(body_html)
    # Step 3 — glossary auto-link (REL-E080). Wraps the first occurrence of
    # each known glossary term in an <abbr> so static-site readers get the
    # same tooltips Streamlit shows. ``load_glossary`` is lru_cached so the
    # ~60-entry file is parsed exactly once per build.
    _glossary = load_glossary(_REPO_ROOT)
    if _glossary:
        body_html = _glossary_annotate(body_html, _glossary)

    breadcrumb = _breadcrumb_html(
        page_path,
        [
            ("Home", "index.html"),
            (cluster_meta["name"], CLUSTER_PAGE[note.cluster]),
            (note.title, None),
        ],
    )

    aside = _metadata_panel_html(note)
    related = _related_views_html(page_path, note)

    body = f"""
    {breadcrumb}
    <div class="note-layout">
        <article class="note-main">
            <h1>{html_escape_mod.escape(note.title)}</h1>
            {body_html}
        </article>
        <div>{aside}{related}</div>
    </div>
"""
    extra_head = f"<style>{highlight_css}</style>"
    if has_mermaid:
        extra_head += _MERMAID_HEAD
    html = _page_shell(
        page_path,
        f"{note.title} — eBERlight Explorer",
        body,
        active_cluster=note.cluster,
        extra_head=extra_head,
        narrow=False,
    )
    target = out_dir / page_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")


def _render_404(out_dir: Path) -> None:
    page_path = "404.html"
    body = """
    <section class="hero">
        <h1>404 — Not Found</h1>
        <p>The page you requested does not exist. Try the <a href="index.html">home page</a>.</p>
    </section>
"""
    html = _page_shell(page_path, "Not Found — eBERlight Explorer", body, narrow=True)
    (out_dir / page_path).write_text(html, encoding="utf-8")


def _render_wireframe_index(out_dir: Path) -> None:
    page_path = "wireframes/index.html"
    body = """
    <section class="cluster-heading">
        <h1>Design Wireframes</h1>
        <p>Static HTML mockups from <code>docs/02_design/wireframes/html/</code>.
        These are design references produced before the Streamlit app and kept
        here for continuity.</p>
    </section>
    <ul>
        <li><a href="landing_v0.1.html">Landing page (v0.1)</a></li>
        <li><a href="section_v0.1.html">Section page (v0.1)</a></li>
        <li><a href="tool_v0.1.html">Tool detail page (v0.1)</a></li>
    </ul>
"""
    target = out_dir / page_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _page_shell(page_path, "Wireframes — eBERlight Explorer", body, narrow=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------


def _write_styles(out_dir: Path) -> None:
    """Concatenate explorer CSS + site-layout CSS → site/assets/styles.css."""
    explorer_css = (_EXPLORER_DIR / "assets" / "styles.css").read_text(encoding="utf-8")
    target = out_dir / "assets" / "styles.css"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(explorer_css + "\n\n" + SITE_LAYOUT_CSS + "\n", encoding="utf-8")


def _copy_note_assets(out_dir: Path) -> None:
    """Copy non-markdown files (images etc.) alongside notes into site/notes/.

    We only mirror the files the Streamlit app and markdown links actually
    reference: anything inside a note folder that is not a .md file.
    """
    exts = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".pdf"}
    for folder in FOLDER_TO_CLUSTER:
        src = _REPO_ROOT / folder
        if not src.is_dir():
            continue
        for path in src.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                rel = path.relative_to(_REPO_ROOT)
                dst = out_dir / "notes" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dst)


def _copy_wireframes(out_dir: Path) -> None:
    src = _REPO_ROOT / "docs" / "02_design" / "wireframes" / "html"
    if not src.is_dir():
        return
    dst = out_dir / "wireframes"
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file() and path.suffix.lower() == ".html":
            shutil.copy2(path, dst / path.name)


def _write_nojekyll(out_dir: Path) -> None:
    """Disable Jekyll so filenames with underscores are served as-is."""
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build(out_dir: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    notes = load_notes(_REPO_ROOT)
    logger.info("Loaded %d notes", len(notes))

    _write_styles(out_dir)
    _write_nojekyll(out_dir)

    _render_landing(out_dir)
    _render_cluster(out_dir, "discover", notes, group_by_folder=False)
    _render_cluster(out_dir, "explore", notes, group_by_folder=True)
    _render_cluster(out_dir, "build", notes, group_by_folder=True)
    for entry in INTERACTIVE_PAGES:
        _render_interactive_stub(out_dir, entry)
    _render_404(out_dir)

    highlight_css = HtmlFormatter(style="monokai", noclasses=False).get_style_defs(".highlight")
    for note in notes:
        _render_note(out_dir, note, highlight_css)

    _copy_note_assets(out_dir)
    _copy_wireframes(out_dir)
    _render_wireframe_index(out_dir)

    logger.info("Static site written to %s", out_dir)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build the eBERlight static site.")
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "site",
        help="Output directory (default: ./site)",
    )
    args = parser.parse_args()
    build(args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(_main())
