"""Train and evaluate the existing job-category classifier."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

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
