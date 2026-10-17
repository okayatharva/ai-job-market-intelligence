"""Provide Gemini and local Ollama AI assistance."""
import os
import requests
import streamlit as st
from src.config import GEMINI_MODEL, OLLAMA_MODEL

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
