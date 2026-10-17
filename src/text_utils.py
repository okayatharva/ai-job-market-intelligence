"""Text cleaning and dataframe column-matching utilities."""

import re


def clean(value):
    """Normalize text, whitespace, and missing values."""
    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        return ", ".join(
            str(item).strip()
            for item in value
            if item is not None
        )

    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass

    if type(value).__name__ in {"NAType", "NaTType"}:
        return ""

    text = str(value)

    # Handle literal newline and tab escape sequences.
    text = text.replace("\\n", " ")
    text = text.replace("\\t", " ")

    # Normalize actual whitespace characters.
    return " ".join(text.split())


def skills_list(value):
    """Split comma-separated skills and remove case-insensitive duplicates."""
    text = clean(value)

    if not text:
        return []

    skills = []
    seen = set()

    for item in text.split(","):
        skill = clean(item)
        key = skill.casefold()

        if skill and key not in seen:
            skills.append(skill)
            seen.add(key)

    return skills


def find_column(df, candidates):
    """Find a column despite differences in case, spaces, or separators."""

    def normalize(name):
        return re.sub(r"[^a-z0-9]+", " ", clean(name).casefold()).strip()

    normalized_columns = {
        normalize(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = normalize(candidate)

        if key in normalized_columns:
            return normalized_columns[key]

    return None
