# AI Job Market Intelligence & Skill Gap Assistant

An interactive Python application that combines job-market analytics, machine learning, semantic search, and generative AI to help users explore job categories and compare résumés with job descriptions.

## Features

- Job category and skill analytics
- Job-category prediction using TF-IDF and Logistic Regression
- Held-out model evaluation using accuracy and macro F1
- Semantic search using Sentence Transformers
- AI-assisted résumé and job-description comparison
- Interactive Streamlit dashboard

## Technology Stack

Python, Streamlit, Pandas, NumPy, Scikit-learn, Hugging Face Datasets, Sentence Transformers, Google Gemini API, and optionally Ollama for local AI inference.

## Dataset

The application uses the Hugging Face job-skill-set dataset:

https://huggingface.co/datasets/batuhanmtl/job-skill-set

This is a fixed dataset, not a live feed of current job vacancies. Check the dataset card and license before redistributing or using the data commercially.

## Run Locally

Clone the repository and open its directory. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` with your own Gemini API key:

```toml
GEMINI_API_KEY = "YOUR_API_KEY"
```

Keep the file private. Never commit API keys to GitHub.

Start the application:

```bash
python -m streamlit run app.py
```

## Evaluation

Record the actual accuracy and macro F1 results displayed by the application. Do not report unmeasured performance figures.

## Limitations

- The dataset does not represent live job-market conditions.
- Rare job categories may be excluded from classification.
- Predictions depend on the training data.
- Semantic similarity does not guarantee that a job is suitable.
- AI-generated recommendations can be incomplete or incorrect and require human review.

## Future Improvements

- Compare different classification models.
- Add cross-validation and error analysis.
- Evaluate semantic-search relevance using reviewed examples.
- Add automated tests and continuous integration.

## Disclaimer

This is an educational portfolio project. It does not guarantee job suitability, interviews, or employment.

## Live Demo
https://ai-job-market-intelligence-hkxkzcnofdcuzkxiyptsp9.streamlit.app


## Project structure

- `app.py`: Streamlit interface and dashboard tabs.
- `src/config.py`: shared dataset and model settings.
- `src/text_utils.py`: text cleaning and skill parsing.
- `src/data_loader.py`: dataset loading and preparation.
- `src/ml_model.py`: TF-IDF and Logistic Regression.
- `src/semantic_search.py`: sentence embeddings and similarity search.
- `src/ai_assistant.py`: Gemini and Ollama integration.
- `tests/`: automated helper-function tests.

## Run locally

```bash
python -m streamlit run app.py
```

## Run tests

```bash
python -m pytest -q
```
