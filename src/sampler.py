import numpy as np
from random import choices


def LatentDirichletAllocation(iden_to_tokens, K, alpha, beta, niter):
    ''' Perform collapsed Gibbs sampling to discover latent topics in corpus

    :param iden_to_tokens: A dictionary that maps unique identifiers to their contents
    :param K: Number of topics for LDA to discover
    :param alpha: Determines sparsity of topic distributions per document
    :param beta: Determines sparsity of word distributions per topic
    :param niter: Number of iterations to run the Gibbs sampler for
    :return: A (K x W) Nu
    '''

    titles = iden_to_tokens.keys()
    document_word_topics = {title: [] for title in titles}  # Contains the ordered list of topics for each document
    # Dict of lists

    # Counts of each topic per document (dict of dicts)
    document_topic_counts = {title: dict.fromkeys(range(1, K + 1), 0) for title in titles}
    word_topic_counts = {word: dict.fromkeys(range(1, K + 1), 0)
                         for word in unique_words}  # Counts of each topic per word (dict of dicts)
    total_topic_counts = dict.fromkeys(range(1, K + 1), 0)  # Counts of each topic across all documents

    for _ in range(niter):  # One iteration of Gibbs sampler
        for doc, words in iden_to_tokens.items():  # For every document
            for i, word in enumerate(words):  # For every word
                densities = np.zeros(K)
                curr_topic = document_word_topics[doc][i]
                for k in range(1, K + 1):
                    N_kj = document_topic_counts[doc].get(k, 0)
                    N_wk = word_topic_counts[word].get(k, 0)
                    N_k = total_topic_counts.get(k, 0)

                    # New draw is conditioned on everything BUT this observation
                    if curr_topic == k:
                        N_kj -= 1
                        N_wk -= 1
                        N_k -= 1

                    # Eq. 1
                    a_kj = N_kj + alpha
                    b_wk = (N_wk + beta) / (N_k + W * beta)

                    densities[k - 1] = a_kj * b_wk

                # Draw a new topic
                densities /= np.sum(densities)  # Normalize
                new_topic = choices(range(1, K + 1), densities)[0]

                if new_topic == curr_topic:
                    continue

                # Update counts
                document_word_topics[doc][i] = new_topic

                document_topic_counts[doc][curr_topic] -= 1
                document_topic_counts[doc][new_topic] += 1

                word_topic_counts[word][curr_topic] -= 1
                word_topic_counts[word][new_topic] += 1

                total_topic_counts[curr_topic] -= 1
                total_topic_counts[new_topic] += 1