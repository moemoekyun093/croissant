"""Text normalization shared by every corpus builder."""


def clean_text(x) -> str:
    """Stringify, collapse all whitespace runs to single spaces, and strip.

    ``None`` becomes the empty string. This is the one canonical
    normalization so web, BIRD and SynSQL tables look identical in the corpus.
    """
    if x is None:
        return ""
    return " ".join(str(x).split()).strip()
