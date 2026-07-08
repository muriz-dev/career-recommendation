# 🎯 Career Recommendation System

## Project Overview
The **Career Recommendation System** is a machine learning-based application designed to help users discover career paths that best align with their skills, experiences, and interests. By utilizing **Natural Language Processing (NLP)** and the **Word2Vec** embedding model, the system analyzes user descriptions and compares them with a diverse dataset of career profiles to find the optimal match.

## Features
- **Intelligent Career Matching:** Get personalized career recommendations based on descriptive text.
- **Natural Language Processing:** Comprehensive text preprocessing including cleaning, tokenization, stopword removal, and lemmatization.
- **Word Embedding Representation:** Transforms textual data into continuous vector spaces using a pretrained Word2Vec model.
- **Similarity Scoring:** Uses Cosine Similarity to calculate the exact percentage match between the user profile and various job roles.
- **Professional User Interface:** Clean, responsive, and intuitive interface built using Streamlit, featuring real-time progress bars, similarity indicators, and detailed job descriptions.
- **Graceful Error Handling:** Provides user-friendly alerts for invalid inputs or out-of-vocabulary terms.

## Dataset
The recommendation model was trained on a comprehensive dataset consisting of **581 unique careers**.
For each career, the dataset includes:
- **Job Title:** The official name of the role.
- **Job Field:** The industry or category the role belongs to.
- **Job Description:** A detailed overview of the daily tasks and responsibilities.
- **Required Skills & Qualifications:** The technical and soft skills necessary for the role.

## Methodology
The recommendation pipeline follows these steps:
1. **User Input:** The user provides a text description of themselves (skills, interests, past experiences).
2. **Text Preprocessing:** The input text is cleaned (lowercased, stripped of special characters), tokenized, stripped of stopwords, and lemmatized.
3. **Vectorization:** The preprocessed words are mapped to their corresponding Word2Vec embeddings. The system computes the average embedding to represent the entire document/profile as a single vector.
4. **Similarity Calculation:** The system calculates the **Cosine Similarity** between the user's document vector and all the pre-computed career vectors in the dataset.
5. **Ranking:** The careers are sorted in descending order of their similarity scores, presenting the highest matching jobs to the user.

## Folder Structure
```text
Career-Recommendation/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
│
├── dataset/
│   └── job_description.csv     # Raw dataset
│
├── models/
│   ├── word2vec.model          # Pretrained Word2Vec model
│   └── career_dataset.pkl      # Serialized dataframe with pre-computed vectors
│
├── notebooks/
│   └── uas-ai03-kelompok2.ipynb # Exploratory Data Analysis & Model Training
│
└── src/
    ├── __init__.py
    ├── preprocessing.py        # Text preprocessing functions
    ├── embedding.py            # Document vectorization logic
    └── recommendation.py       # Core recommendation engine
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd Career-Recommendation
   ```

2. **Create a virtual environment (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK resources**
   The necessary NLTK data (punkt, stopwords, wordnet) will be automatically downloaded the first time you run the application.

## Running the Application
To start the Streamlit application, run the following command in your terminal:
```bash
streamlit run app.py
```
This will open the application in your default web browser (usually at `http://localhost:8501`).

## Technologies
- **Python 3.x**
- **Streamlit** (Web Framework)
- **Gensim** (Word2Vec)
- **Scikit-Learn** (Cosine Similarity)
- **NLTK** (Natural Language Processing)
- **Pandas & NumPy** (Data Manipulation)

## Example Output
When providing a description such as:
> *"I enjoy building web applications. I work with Python, SQL, Docker, and REST APIs. I like cloud computing and backend development."*

**The system will output:**
- **Best Career Match:** Backend Developer (🔥 Excellent Match: ~92.50%)
- **Other Recommendations:**
  - Cloud Architect (✨ Good Match: ~85.30%)
  - Full Stack Developer (✨ Good Match: ~82.10%)
