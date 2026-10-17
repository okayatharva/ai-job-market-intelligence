
"""Tests for text-cleaning and column-matching helpers."""

import pandas as pd

from src.text_utils import clean, skills_list, find_column


def test_clean_normalizes_whitespace():
    text = " Python  " + chr(10) + " SQL "
    assert clean(text) == "Python SQL"


def test_clean_handles_missing_values():
    assert clean(None) == ""


def test_skills_list_removes_case_insensitive_duplicates():
    assert skills_list("Python, SQL, python, SQL") == [
        "Python",
        "SQL",
    ]


def test_skills_list_handles_missing_values():
    assert skills_list(None) == []


def test_find_column_matches_case_insensitively():
    df = pd.DataFrame(columns=["Job Title", "Company"])

    assert find_column(df, ["job title"]) == "Job Title"


def test_find_column_returns_none_when_not_found():
    df = pd.DataFrame(columns=["Job Title", "Company"])

    assert find_column(df, ["salary"]) is None


def test_find_column_checks_candidates_in_order():
    df = pd.DataFrame(columns=["Title", "Job Title"])

    assert find_column(df, ["Job Title", "Title"]) == "Job Title"


def test_find_column_handles_extra_whitespace():
    df = pd.DataFrame(columns=["Job Title"])

    assert find_column(df, ["  job   title  "]) == "Job Title"
