"""Tests for text-cleaning and column-matching helpers."""

import pandas as pd

from src.text_utils import clean, skills_list, find_column


def test_clean_normalizes_whitespace():
    text = " Python  " + "\\n" + " SQL "
    assert clean(text) == "Python SQL"


def test_clean_handles_missing_values():
    assert clean(None) == ""


def test_skills_list_removes_case_insensitive_duplicates():
    assert skills_list("Python, SQL, python") == ["Python", "SQL"]


def test_find_column_matches_alternative_formatting():
    frame = pd.DataFrame({"Job Title": ["Data Analyst"]})
    assert find_column(frame, ["job_title"]) == "Job Title"
