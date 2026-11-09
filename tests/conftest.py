import csv
from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from api.predictor import BaselinePredictor
from src.preprocess import clean_text

FIXTURE = Path(__file__).parent / "fixtures" / "sample_complaints.csv"


@pytest.fixture(scope="session")
def fixture_rows():
    with FIXTURE.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="session")
def tiny_predictor(fixture_rows):
    """A small baseline fit on the synthetic fixture. Real models are not needed to test the API."""
    labels = sorted({r["label"] for r in fixture_rows})
    X = [clean_text(r["narrative"]) for r in fixture_rows]
    y = [labels.index(r["label"]) for r in fixture_rows]
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                     ("clf", LogisticRegression(C=10, max_iter=500))])
    pipe.fit(X, y)
    return BaselinePredictor(pipe, labels, "test-fixture")
