import numpy as np


def document_vector(text: str, model):
    """
    Convert a document into a vector using the average
    Word2Vec embedding.
    """

    words = text.split()

    vectors = [
        model.wv[word]
        for word in words
        if word in model.wv
    ]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)