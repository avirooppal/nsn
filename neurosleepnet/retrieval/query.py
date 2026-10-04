"""
FTS5 BM25 Query Compiler.
Compiles natural-language questions and deliberate phrases into safe,
deterministic SQLite FTS5 query expressions.
"""
import re
from typing import List, Tuple

STOP_WORDS = {
    "what", "is", "are", "the", "a", "an", "how", "why", "when", "where",
    "who", "which", "does", "do", "did", "can", "could", "would", "should",
    "using", "use", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "about", "me", "tell", "show", "please", "my", "your", "our",
}


class FTSQueryCompiler:
    """
    Compiles search queries for SQLite FTS5.
    Distinguishes deliberate phrase searches from natural-language keyword questions.
    """

    @staticmethod
    def compile(query: str) -> Tuple[str, List[str], bool]:
        """
        Compiles a query string into an FTS5 search expression.
        Returns:
            (fts_expression, extracted_terms, is_phrase)
        """
        if not query or not query.strip():
            return "", [], False

        raw = query.strip()

        # Deliberate phrase mode: quoted with "..." or '...'
        if (raw.startswith('"') and raw.endswith('"') and len(raw) > 2) or \
           (raw.startswith("'") and raw.endswith("'") and len(raw) > 2):
            phrase = raw[1:-1].strip()
            # Clean internal double-quotes
            safe_phrase = phrase.replace('"', '""')
            return f'"{safe_phrase}"', [phrase], True

        # Extract tokens and identifiers (e.g. 'auth', 'production-port', '8080', 'db.internal')
        # Preserve alphanumerics, hyphens, underscores, dots, colons
        raw_tokens = re.findall(r'[A-Za-z0-9_\-\.:]+', raw)
        if not raw_tokens:
            return "", [], False

        # Filter out question/filler stopwords
        filtered_terms = [
            t for t in raw_tokens
            if t.lower() not in STOP_WORDS and len(t) > 1 or t.isdigit()
        ]

        # If all terms were filtered, fall back to non-empty raw tokens
        active_terms = filtered_terms if filtered_terms else raw_tokens

        # Escape each token by quoting it in FTS5
        fts_tokens = []
        for term in active_terms:
            safe_t = term.replace('"', '""')
            fts_tokens.append(f'"{safe_t}"')

        # Combine with OR to allow term-based BM25 matching
        fts_query = " OR ".join(fts_tokens)
        return fts_query, active_terms, False
