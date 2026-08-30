"""Tests for suffix_maps.py — run with `uv run pytest`."""

import pandas as pd

from suffix_maps import abbreviate, expand


def test_abbreviate_basic():
    assert abbreviate("main street") == "main st"
    assert abbreviate("winding trail rd") == "winding trl rd"
    assert abbreviate("interlake avenue n") == "interlake ave n"


def test_expand_basic():
    assert expand("main st") == "main street"
    assert expand("mallory ln") == "mallory lane"
    assert expand("interlake ave n") == "interlake avenue n"


def test_round_trip_on_clean_suffixes():
    for raw in ["main st", "main street", "oak avenue", "oak ave"]:
        assert abbreviate(expand(raw)) == abbreviate(raw)
        assert expand(abbreviate(raw)) == expand(raw)


def test_typod_suffix_untouched():
    assert abbreviate("main stree") == "main stree"
    assert expand("main stree") == "main stree"


def test_saint_collision_is_real():
    # the naive expander corrupts Saint-style names — by design, this is
    # the failure mode the discussion warns about
    assert expand("st clair ave") == "street clair avenue"
    assert abbreviate("st clair ave") == "st clair ave"


def test_period_stripped_on_match_only():
    assert expand("main st.") == "main street"
    assert expand("st.hwy 320") == "st.hwy 320"


def test_non_string_passthrough():
    assert pd.isna(abbreviate(float("nan")))
    assert pd.isna(expand(float("nan")))
