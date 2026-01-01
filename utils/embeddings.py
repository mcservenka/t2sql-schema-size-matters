import numpy as np
from numpy.linalg import norm


def embed_word(word: str, embedding_model):
    return embedding_model.get_word_vector(word)


def embed_words(words: list[str]) -> np.ndarray:
    return np.stack([embed_word(w) for w in words])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))

