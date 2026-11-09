def test_fixture_is_balanced_and_nonempty(fixture_rows):
    from collections import Counter

    counts = Counter(r["label"] for r in fixture_rows)
    assert len(counts) == 4 and set(counts.values()) == {6}
    assert all(len(r["narrative"].split()) >= 10 for r in fixture_rows)
