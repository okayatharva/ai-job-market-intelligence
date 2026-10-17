"""Load sentence embeddings and search similar job descriptions."""
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from src.config import EMBEDDING_MODEL

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
