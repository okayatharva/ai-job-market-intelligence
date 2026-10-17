"""Load and prepare the existing Hugging Face job dataset."""
import pandas as pd
import streamlit as st
from datasets import load_dataset
from src.config import DATASET
from src.text_utils import clean, find_column

@st.cache_data(show_spinner="Loading the job dataset...")
def load_jobs():
    dataset = load_dataset(DATASET)

    if hasattr(dataset, "keys"):
        split_name = "train" if "train" in dataset else list(dataset.keys())[0]
        source = dataset[split_name]
    else:
        source = dataset

    df = source.to_pandas()

    category_col = find_column(
        df, ["category", "job_category", "role_category", "job_role"]
    )
    title_col = find_column(
        df, ["job_title", "title", "position", "role", "job"]
    )
    description_col = find_column(
        df, ["job_description", "description", "job_desc", "details"]
    )
    skills_col = find_column(
        df, ["job_skill_set", "skills", "required_skills", "job_skills", "skill_set"]
    )

    missing = []
    if category_col is None:
        missing.append("category")
    if title_col is None:
        missing.append("job title")
    if description_col is None:
        missing.append("job description")

    if missing:
        raise ValueError(
            "The downloaded dataset is missing required fields: "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(map(str, df.columns))
        )

    result = pd.DataFrame()
    result["category"] = df[category_col].map(clean)
    result["job_title"] = df[title_col].map(clean)
    result["job_description"] = df[description_col].map(clean)

    if skills_col is not None:
        result["job_skill_set"] = df[skills_col].map(clean)
    else:
        result["job_skill_set"] = ""

    result = result.drop_duplicates(
        subset=["category", "job_title", "job_description"]
    )

    result = result[
        (result["category"] != "")
        & (result["job_title"] != "")
        & (result["job_description"] != "")
    ].copy()

    # Very rare categories are excluded to make held-out evaluation possible.
    category_counts = result["category"].value_counts()
    valid_categories = category_counts[category_counts >= 4].index
    result = result[result["category"].isin(valid_categories)].reset_index(drop=True)

    if result.empty:
        raise ValueError(
            "No usable records remain after cleaning. Check the dataset columns "
            "and category counts."
        )

    result["combined_text"] = (
        result["job_title"]
        + " "
        + result["job_description"]
        + " "
        + result["job_skill_set"]
    ).str.strip()

    return result
