from src.config import LABELS, PRODUCT_MAP
from src.preprocess import clean_text, truncate


def test_redactions_become_placeholders():
    out = clean_text("On XX/XX/XXXX XXXX charged me {$25.00} twice.")
    assert "XX" not in out
    assert "redacted" in out and "amount" in out


def test_whitespace_collapsed_and_stripped():
    assert clean_text("  hello \n\n  world\t ") == "hello world"


def test_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_truncate_adds_ellipsis():
    out = truncate("word " * 200, 50)
    assert len(out) <= 50 and out.endswith("...")
    assert truncate("short") == "short"


def test_product_map_merges_into_expected_classes():
    assert PRODUCT_MAP["Prepaid card"] == "credit_card"
    assert PRODUCT_MAP["Credit reporting"] == "credit_reporting"
    assert len(LABELS) == len(set(LABELS)) == 9
