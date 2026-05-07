"""Full-text search over the loaded notes (Phase R6).

Lightweight in-memory inverted index. Built on top of the existing
:func:`lib.notes.load_notes` so the index stays in lockstep with
the notes the rest of the explorer renders.

Why not Whoosh / lunr / Elasticsearch?

* The corpus is small (~200 notes, ~5 MB total).  At that size a
  Python `dict[str, set[int]]` answers every query in <10 ms on a
  cold CPU and uses <50 MB RAM.
* Whoosh adds a 2 MB dependency, requires writing an index to disk,
  and in tests its query parser was fussier than just substring +
  fuzzy. Lunr would force a JS bundle on the static-site mirror.
* The simpler approach also lets the static-site generator emit
  the same index as a JSON blob for client-side filtering later
  without another build step.

Pure data layer — no Streamlit. Designed to be unit-tested.

Ref: FR-009 — global search bar with relevance ranking.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .notes import Note

# Tokenizer: lowercase, split on non-alphanumeric. Tokens must
# start AND end with an alphanumeric so trailing punctuation (".",
# ",", "-") doesn't create distinct keys. Single-char tokens are
# dropped to suppress `, ;` noise.
_TOKEN_RE = re.compile(r"[a-z0-9](?:[a-z0-9_+.-]*[a-z0-9])?")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2]


@dataclass
class Index:
    """Inverted-index keyed by token → set of note indices."""

    notes: list[Note] = field(default_factory=list)
    inverted: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    note_token_counts: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.notes)

    def suggest(self, query: str, *, limit: int = 6) -> list[str]:
        """Return up to ``limit`` indexed terms close to the user's query.

        Used by the Search page when a query produced zero hits — we
        offer "did you mean" links to indexed tokens that share a
        non-trivial common prefix with one of the query tokens. No
        external dependency (no Levenshtein library); we lean on the
        plain inverted-index keys plus length/prefix heuristics.

        Picks the longer of the query tokens (more discriminating),
        keeps tokens whose first 3 characters match, ranks by
        document-frequency descending so the most popular candidate
        comes first.
        """
        q_tokens = sorted(set(_tokenize(query)), key=len, reverse=True)
        if not q_tokens:
            return []
        candidates: dict[str, int] = {}
        for q_tok in q_tokens:
            prefix = q_tok[: max(3, min(4, len(q_tok)))]
            for term, postings in self.inverted.items():
                if term == q_tok:
                    continue
                if term.startswith(prefix) or (
                    len(term) >= 4 and len(q_tok) >= 4 and term[:4] == q_tok[:4]
                ):
                    candidates[term] = max(candidates.get(term, 0), len(postings))
        ranked = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
        return [term for term, _ in ranked[:limit]]


def build_index(notes: list[Note]) -> Index:
    """Build the in-memory index from a sequence of parsed notes.

    Each note is broken into tokens from its title + body + tags +
    folder + modality so the search picks up metadata as well as
    content. Token positions are not retained — the page uses a
    cheap "snippet around first match in body" instead.
    """
    idx = Index(notes=list(notes))
    for note_idx, note in enumerate(idx.notes):
        text = " ".join(
            [
                note.title or "",
                note.body or "",
                " ".join(note.tags or []),
                note.modality or "",
                note.folder or "",
                " ".join(note.beamline or []),
            ]
        )
        tokens = _tokenize(text)
        idx.note_token_counts.append(len(tokens))
        for tok in tokens:
            idx.inverted[tok].add(note_idx)
    return idx


@dataclass(frozen=True)
class SearchHit:
    """One hit returned by :func:`search`."""

    note: Note
    score: float
    matched_terms: tuple[str, ...]
    snippet: str  # ~160-char excerpt around the first match


def _snippet(body: str, query_terms: list[str], width: int = 160) -> str:
    """Return a short excerpt around the first query-term occurrence."""
    if not body:
        return ""
    lowered = body.lower()
    earliest = -1
    for term in query_terms:
        idx = lowered.find(term.lower())
        if idx >= 0 and (earliest < 0 or idx < earliest):
            earliest = idx
    if earliest < 0:
        return body[:width].strip().replace("\n", " ")
    start = max(0, earliest - width // 2)
    end = min(len(body), earliest + width // 2)
    excerpt = body[start:end].strip().replace("\n", " ")
    if start > 0:
        excerpt = "…" + excerpt
    if end < len(body):
        excerpt = excerpt + "…"
    return excerpt


def search(idx: Index, query: str, *, limit: int = 30) -> list[SearchHit]:
    """Run a full-text search over the index.

    Scoring is a tiny TF-IDF approximation:
    ``score = sum_for_term(matches_in_note / sqrt(note_token_count + 1))``,
    boosted ×2 if the term hits the title verbatim. Results are
    sorted score-descending, ties broken by note path for
    determinism.
    """
    terms = _tokenize(query)
    if not terms:
        return []

    note_scores: dict[int, float] = defaultdict(float)
    note_terms: dict[int, set[str]] = defaultdict(set)

    for term in terms:
        # Direct match.
        for ni in idx.inverted.get(term, ()):
            note_scores[ni] += 1.0 / max(1.0, (idx.note_token_counts[ni] + 1) ** 0.5)
            note_terms[ni].add(term)
        # Prefix match — catches plurals / inflections cheaply.
        if len(term) >= 4:
            for tok, ids in idx.inverted.items():
                if tok.startswith(term) and tok != term:
                    for ni in ids:
                        note_scores[ni] += 0.4 / max(1.0, (idx.note_token_counts[ni] + 1) ** 0.5)
                        note_terms[ni].add(tok)

    # Title boost — case-insensitive substring of the original query.
    q_lower = query.lower()
    for ni, note in enumerate(idx.notes):
        if q_lower in (note.title or "").lower():
            note_scores[ni] = note_scores.get(ni, 0.0) * 2.0 or 0.5

    hits: list[SearchHit] = []
    for ni, score in note_scores.items():
        note = idx.notes[ni]
        hits.append(
            SearchHit(
                note=note,
                score=score,
                matched_terms=tuple(sorted(note_terms[ni])),
                snippet=_snippet(note.body, terms),
            )
        )
    hits.sort(key=lambda h: (-h.score, str(h.note.path)))
    return hits[:limit]


def index_from_repo(repo_root: Path) -> Index:
    """Convenience: load every note + return a fresh index."""
    from .notes import load_notes

    return build_index(load_notes(repo_root))
