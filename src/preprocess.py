"""Text cleaning shared by training, evaluation and the API."""
import re

_WS = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Normalise a complaint narrative.

    CFPB already redacts personal data as ``XXXX`` or ``{$amount}``; we collapse those to
    short placeholder tokens (``redacted`` / ``amount``) so they do not dominate n-grams,
    then squeeze whitespace. Case is kept for the transformer's uncased tokenizer to handle.
    """
    if not text:
        return ""
    text = text.replace(" ", " ")
    text = re.sub(r"\{\$[\d.,]+\}", " amount ", text)
    text = re.sub(r"X{2,}(?:/X{2,})*", " redacted ", text)
    return _WS.sub(" ", text).strip()


def truncate(text: str, n: int = 300) -> str:
    """Shorten text for display in reports."""
    text = _WS.sub(" ", text).strip()
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."
