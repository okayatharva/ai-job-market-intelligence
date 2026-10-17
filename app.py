
import ast
import os
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------



# The reusable business logic lives in src/ modules.
from src.config import DATASET, EMBEDDING_MODEL, GEMINI_MODEL, OLLAMA_MODEL
from src.text_utils import clean, skills_list, find_column
from src.data_loader import load_jobs
from src.ml_model import train_model
from src.semantic_search import (
    get_embedding_model,
    create_embeddings,
    semantic_search,
)
from src.ai_assistant import get_gemini_api_key, ai_answer

st.set_page_config(
    page_title="AI Job Market Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Job Market Intelligence & Skill Gap Assistant")
st.caption(
    "Explore job skills, predict job categories, search similar roles, "
    "and compare your résumé with a job description."
)


# ---------------------------------------------------------
# TEXT HELPERS
# ---------------------------------------------------------







# ---------------------------------------------------------
# LOAD AND PREPARE DATA
# ---------------------------------------------------------



# ---------------------------------------------------------
# MACHINE-LEARNING CLASSIFIER
# ---------------------------------------------------------



# ---------------------------------------------------------
# SEMANTIC SEARCH
# ---------------------------------------------------------







# ---------------------------------------------------------
# GENERATIVE AI: GEMINI OR LOCAL OLLAMA
# ---------------------------------------------------------





# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

try:
    jobs = load_jobs()
except Exception as exc:
    st.error(f"Could not load the job dataset: {exc}")
    st.info(
        "Check your internet connection and the dataset page: "
        "https://huggingface.co/datasets/batuhanmtl/job-skill-set"
    )
    st.stop()


# ---------------------------------------------------------
# TOP-LEVEL METRICS
# ---------------------------------------------------------

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Usable job records", f"{len(jobs):,}")
metric2.metric("Job categories", f"{jobs['category'].nunique():,}")
metric3.metric(
    "Records with skill information",
    f"{(jobs['job_skill_set'].str.strip() != '').sum():,}",
)

analytics, predictor, search_tab, gap_tab = st.tabs(
    [
        "📈 Analytics",
        "🤖 ML Predictor",
        "🔎 Semantic Search",
        "🧩 Résumé Skill Gap",
    ]
)


# ---------------------------------------------------------
# TAB 1: ANALYTICS
# ---------------------------------------------------------

with analytics:
    st.subheader("Job-category distribution")

    category_count = jobs["category"].nunique()
    max_categories = min(20, category_count)

    # Slider min and max must not be equal.
    if max_categories > 1:
        top_n = st.slider(
            "Categories to display",
            min_value=1,
            max_value=max_categories,
            value=min(5, max_categories),
        )
    else:
        top_n = 1

    counts = jobs["category"].value_counts().head(top_n)
    st.bar_chart(counts)

    st.subheader("Frequently mentioned skills")

    all_skills = []
    for skill_text in jobs["job_skill_set"]:
        all_skills.extend(skills_list(skill_text))

    if all_skills:
        skill_counts = (
            pd.Series(all_skills)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .value_counts()
            .head(15)
        )
        st.bar_chart(skill_counts)
    else:
        st.info(
            "This dataset does not provide skills in a format that can be "
            "counted reliably."
        )

    st.subheader("Dataset preview")
    st.dataframe(
        jobs[["category", "job_title", "job_skill_set", "job_description"]].head(20)
    )


# ---------------------------------------------------------
# TAB 2: ML PREDICTOR
# ---------------------------------------------------------

with predictor:
    st.subheader("Predict a job category")
    st.write(
        "Enter a job title and description. The TF-IDF and Logistic Regression "
        "model predicts the most likely category from the dataset."
    )

    job_title_input = st.text_input(
        "Job title",
        placeholder="e.g. Junior Data Analyst",
        key="predict_title",
    )
    job_description_input = st.text_area(
        "Job description",
        placeholder="Describe the responsibilities, tools, and required skills...",
        height=180,
        key="predict_description",
    )

    if st.button("Predict category", key="predict_button"):
        input_text = f"{job_title_input} {job_description_input}".strip()

        if not input_text:
            st.warning("Enter a job title or job description first.")
        else:
            try:
                vectorizer, model, metrics = train_model(
                    jobs["combined_text"].tolist(),
                    jobs["category"].tolist(),
                )
                vectorized_input = vectorizer.transform([input_text])
                predicted_category = model.predict(vectorized_input)[0]
                probabilities = model.predict_proba(vectorized_input)[0]
                class_order = model.classes_
                ranked = np.argsort(probabilities)[::-1][:3]

                st.success(f"Predicted category: **{predicted_category}**")

                st.caption(
                    "The model is trained on this fixed dataset. Predictions "
                    "are estimates, not guarantees."
                )

                st.subheader("Held-out evaluation")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                col2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
                col3.metric("Test records", metrics["test_size"])

                st.subheader("Top category scores")
                score_table = pd.DataFrame(
                    {
                        "Category": class_order[ranked],
                        "Model score": probabilities[ranked],
                    }
                )
                score_table["Model score"] = score_table["Model score"].map(
                    lambda value: f"{value:.1%}"
                )
                st.dataframe(score_table)

                with st.expander("View classification report"):
                    report_df = pd.DataFrame(metrics["report"]).T
                    st.dataframe(report_df)

            except Exception as exc:
                st.error(f"The classifier could not run: {exc}")


# ---------------------------------------------------------
# TAB 3: SEMANTIC SEARCH
# ---------------------------------------------------------

with search_tab:
    st.subheader("Find semantically similar jobs")
    st.write(
        "This search uses sentence embeddings to find descriptions with related "
        "meaning, even when they do not use exactly the same keywords."
    )

    search_query = st.text_area(
        "Describe the role or skills you want",
        placeholder=(
            "e.g. A junior role using Python, SQL, dashboards, "
            "and basic machine learning"
        ),
        height=100,
        key="semantic_query",
    )
    result_count = st.slider(
        "Number of results",
        min_value=3,
        max_value=10,
        value=5,
        key="semantic_result_count",
    )

    if st.button("Search similar jobs", key="semantic_search_button"):
        if not search_query.strip():
            st.warning("Enter a role or skill description first.")
        else:
            try:
                with st.spinner("Searching job descriptions..."):
                    embeddings = create_embeddings(jobs["combined_text"].tolist())
                    results = semantic_search(
                        search_query,
                        jobs,
                        embeddings,
                        result_count,
                    )

                for _, result in results.iterrows():
                    with st.container(border=True):
                        st.markdown(f"**{result['job_title']}**")
                        st.caption(f"Category: {result['category']}")
                        st.caption(
                            f"Similarity score: "
                            f"{result['similarity_score']:.1%}"
                        )
                        st.write(
                            clean(result["job_description"])[:900]
                            + (
                                "..."
                                if len(clean(result["job_description"])) > 900
                                else ""
                            )
                        )
                        if clean(result["job_skill_set"]):
                            st.write(
                                "**Listed skills:** "
                                + clean(result["job_skill_set"])[:500]
                            )

            except Exception as exc:
                st.error(f"Semantic search failed: {exc}")


# ---------------------------------------------------------
# TAB 4: RÉSUMÉ SKILL GAP
# ---------------------------------------------------------

with gap_tab:
    st.subheader("Compare your résumé with a job description")
    st.write(
        "Paste your résumé text and a target job description. The assistant "
        "will identify overlapping skills, possible gaps, and practical next steps."
    )

    resume_text = st.text_area(
        "Résumé text",
        placeholder="Paste the relevant text from your résumé here...",
        height=220,
        key="resume_text",
    )
    target_job = st.text_area(
        "Target job description",
        placeholder="Paste the job description here...",
        height=220,
        key="target_job",
    )

    if st.button("Analyse skill gap", key="analyse_gap_button"):
        if not resume_text.strip() or not target_job.strip():
            st.warning("Please provide both résumé text and a job description.")
        else:
            prompt = f"""
You are a careful career and skills analysis assistant.

Compare the candidate's résumé with the target job description.

Use these sections:
1. Short overall match summary.
2. Skills and experience explicitly supported by the résumé.
3. Requirements in the job description that are not clearly evidenced in the résumé.
4. Three practical learning or portfolio priorities.
5. Suggestions for describing relevant existing experience honestly.

Important rules:
- Do not invent experience, qualifications, projects, or skills.
- Distinguish "not mentioned in the résumé" from "the candidate does not have this skill".
- Do not guarantee interviews or employment.
- Be specific, concise, constructive, and beginner-friendly.

RÉSUMÉ:
{resume_text[:12000]}

TARGET JOB DESCRIPTION:
{target_job[:12000]}
"""
            with st.spinner("Analysing the résumé and job description..."):
                answer = ai_answer(prompt)

            st.markdown(answer)

st.divider()
st.caption(
    "Educational portfolio project. The dataset is a fixed sample, not a "
    "live feed of current vacancies. Model predictions and AI suggestions "
    "should be reviewed by a human."
)
