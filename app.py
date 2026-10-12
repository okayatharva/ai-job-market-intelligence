
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

DATASET = "batuhanmtl/job-skill-set"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
OLLAMA_MODEL = "llama3.2:1b"

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

def clean(value):
    """Convert text, lists, and missing values into clean strings."""
    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if item is not None)

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return re.sub(r"\s+", " ", str(value)).strip()


def skills_list(value):
    """Convert a skill field into a list of readable skills."""
    if isinstance(value, (list, tuple, set)):
        raw_skills = list(value)
    else:
        text = clean(value)

        if not text:
            return []

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                raw_skills = list(parsed)
            else:
                raw_skills = re.split(r"[,;\n|]", text)
        except (ValueError, SyntaxError):
            raw_skills = re.split(r"[,;\n|]", text)

    result = []
    seen = set()

    for skill in raw_skills:
        item = clean(skill).strip(" []{}'\"")
        if item and item.lower() not in seen:
            result.append(item)
            seen.add(item.lower())

    return result


def find_column(df, candidates):
    """Find a dataset column using common alternative names."""
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): column
        for column in df.columns
    }

    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]

    return None


# ---------------------------------------------------------
# LOAD AND PREPARE DATA
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# MACHINE-LEARNING CLASSIFIER
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Training and evaluating the classifier...")
def train_model(texts, labels):
    texts = list(texts)
    labels = list(labels)

    number_of_classes = len(set(labels))

    if number_of_classes < 2:
        raise ValueError(
            "At least two job categories are required to train a classifier."
        )

    counts = pd.Series(labels).value_counts()
    if counts.min() < 4:
        raise ValueError(
            "Each category needs at least four records for this evaluation setup."
        )

    test_count = max(int(np.ceil(len(labels) * 0.25)), number_of_classes)
    test_count = min(test_count, len(labels) - number_of_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_count,
        random_state=42,
        stratify=labels,
    )

    evaluation_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=1,
        stop_words="english",
        sublinear_tf=True,
    )

    X_train_tfidf = evaluation_vectorizer.fit_transform(X_train)
    X_test_tfidf = evaluation_vectorizer.transform(X_test)

    evaluation_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    evaluation_model.fit(X_train_tfidf, y_train)

    predictions = evaluation_model.predict(X_test_tfidf)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "test_size": len(y_test),
    }

    # Train the final predictor on all available records after evaluation.
    final_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=1,
        stop_words="english",
        sublinear_tf=True,
    )
    all_tfidf = final_vectorizer.fit_transform(texts)

    final_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    final_model.fit(all_tfidf, labels)

    return final_vectorizer, final_model, metrics


# ---------------------------------------------------------
# SEMANTIC SEARCH
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading the semantic search model...")
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_data(show_spinner="Creating job-description embeddings...")
def create_embeddings(texts):
    model = get_embedding_model()
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return normalize(embeddings)


def semantic_search(query, jobs, embeddings, top_k=5):
    model = get_embedding_model()
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    query_embedding = normalize(query_embedding)

    scores = cosine_similarity(query_embedding, embeddings)[0]
    indices = np.argsort(scores)[::-1][:top_k]

    results = jobs.iloc[indices][
        ["job_title", "category", "job_description", "job_skill_set"]
    ].copy()
    results["similarity_score"] = scores[indices]
    return results


# ---------------------------------------------------------
# GENERATIVE AI: GEMINI OR LOCAL OLLAMA
# ---------------------------------------------------------

def get_gemini_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY", "")


def ai_answer(prompt):
    """Use Gemini when configured; otherwise try a local Ollama model."""
    api_key = get_gemini_api_key()
    errors = []

    if api_key:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent"
        )

        try:
            response = requests.post(
                url,
                params={"key": api_key},
                json={
                    "contents": [
                        {"parts": [{"text": prompt}]}
                    ],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 1200,
                    },
                },
                timeout=60,
            )

            if response.ok:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    answer = "\n".join(
                        part.get("text", "")
                        for part in parts
                        if part.get("text")
                    )
                    if answer:
                        return answer

                errors.append("Gemini returned no text in its response.")
            else:
                errors.append(
                    f"Gemini API returned HTTP {response.status_code}: "
                    f"{response.text[:400]}"
                )

        except requests.RequestException as exc:
            errors.append(f"Could not connect to Gemini: {exc}")

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        if response.ok:
            answer = response.json().get("response", "").strip()
            if answer:
                return answer

        errors.append(
            f"Ollama returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    except requests.RequestException:
        errors.append(
            "Local Ollama is unavailable. Start Ollama and ensure the "
            f"'{OLLAMA_MODEL}' model is installed, or configure GEMINI_API_KEY."
        )

    return "AI request could not be completed.\n\n" + "\n\n".join(errors)


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
