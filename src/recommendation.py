import pickle
import pandas as pd
import numpy as np
import streamlit as st

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple

from src.preprocessing import preprocess_text
from src.embedding import document_vector

# Paths to resources
MODEL_PATH = "models/word2vec.model"
DATASET_PATH = "models/career_dataset.pkl"


@st.cache_resource
def load_resources() -> Tuple[Word2Vec, pd.DataFrame]:
    """
    Load Word2Vec model and career dataset from disk.
    Uses Streamlit caching to prevent multiple disk reads and optimize performance.
    """
    word2vec_model = Word2Vec.load(MODEL_PATH)

    with open(DATASET_PATH, "rb") as f:
        career_df = pickle.load(f)

    return word2vec_model, career_df


def get_user_vector(user_input: str) -> Tuple[str, np.ndarray]:
    """
    Process the user input text and return the cleaned text along with its vector representation.

    Validates that:
    1. Input is not completely empty.
    2. Input is not just numbers, symbols, or whitespaces (checks length of cleaned text).
    3. Input contains at least one valid word present in the Word2Vec vocabulary.
    """
    if not user_input.strip():
        raise ValueError("Input cannot be empty. Please describe yourself.")

    processed_text = preprocess_text(user_input)

    # Check if the text only contained numbers or symbols that were stripped away
    if not processed_text.strip():
        raise ValueError(
            "Your input only contains invalid characters (numbers, symbols, or stop words). Please provide a descriptive text about your skills and interests."
        )

    word2vec_model, _ = load_resources()

    # Verify if at least one word exists in the vocabulary
    words = processed_text.split()
    valid_words = [word for word in words if word in word2vec_model.wv]

    if not valid_words:
        raise ValueError(
            "None of the words in your input match our system vocabulary. Try using common keywords related to skills, fields, or jobs."
        )

    # Generate the embedding vector
    user_vector = document_vector(
        processed_text,
        word2vec_model
    )

    return processed_text, user_vector


def recommend_career(user_input: str, top_k: int = 3) -> pd.DataFrame:
    """
    Recommend top careers based on the semantic similarity between the user's description
    and the job descriptions in the dataset using Cosine Similarity.
    """
    processed_text, user_vector = get_user_vector(user_input)

    _, career_df = load_resources()

    # Calculate cosine similarity
    similarities = cosine_similarity(
        [user_vector],
        list(career_df["career_vector"])
    )[0]

    result = career_df.copy()

    # Add similarity percentage
    result["Similarity (%)"] = (similarities * 100).round(2)

    # Sort results by descending similarity
    result = result.sort_values(
        by="Similarity (%)",
        ascending=False
    )

    # Pick top k
    result = result.head(top_k).reset_index(drop=True)

    result.index += 1
    result.index.name = "Rank"

    return result[
        [
            "Job Title",
            "Job Field",
            "Job Description",
            "Required Skills & Qualifications",
            "Similarity (%)"
        ]
    ]