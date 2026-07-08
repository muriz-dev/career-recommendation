import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Preprocess text for Word2Vec.

    Steps:
    1. Lowercase
    2. Replace hyphen with space
    3. Remove special characters
    4. Remove extra whitespace
    5. Tokenization
    6. Stopword removal
    7. Lemmatization

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Cleaned text.
    """

    # Lowercase
    text = text.lower()

    # Replace hyphen
    text = text.replace("-", " ")

    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal
    tokens = [
        token
        for token in tokens
        if token not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]

    return " ".join(tokens)