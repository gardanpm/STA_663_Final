import numpy as np


def get_top_n_words(phi_matrix, n, unique_words):
    topic_top_words = {}

    for k in range(phi_matrix.shape[0]):
        top_10_idx = np.argsort(phi_matrix[k, :])[::-1][:n]
        top_10_words = [unique_words[i] for i in top_10_idx]
        topic_top_words[k + 1] = top_10_words

    return topic_top_words
