import re
import unicodedata

ARTICLES = {"the", "a", "an"}
_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)
_NONALNUM = re.compile(r"[\W_]+", re.UNICODE)


def nfkc(s):
    return unicodedata.normalize("NFKC", s)


def norm_alnum(s):
    """Lowercase and keep only letters/digits. Used for verbatim-quote checks, because
    Re-DocRED tokens are re-joined with spaces ('Schneider ( born') and LLMs re-space text."""
    return _NONALNUM.sub("", nfkc(s).lower())


def norm_tokens(s):
    """Token tuple for entity-name comparison (lowercase, no punctuation, no articles)."""
    toks = _PUNCT.sub(" ", nfkc(s).lower()).split()
    return tuple(t for t in toks if t not in ARTICLES)
