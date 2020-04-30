# Implement gensim LDA approximation in parallele on the Reuter dataset
from src.utility import *
from sklearn.datasets import fetch_20newsgroups
# ! pip install gensim
import gensim
import gensim.corpora as corpora
from Comparisons.func_gensim import *
from pprint import pprint
from time import time

if __name__ == '__main__':
    K = 10
    n_top_words = 10

    ###### First get the data ready similarly to our implemented example
    with open('../Data/reuters21578/reut2-000.sgm') as f:
            data = f.read()

    titles_to_articles = parse_sgm_file(data)
    titles_to_tokens = {title: tokenize_doc(doc) for title, doc in titles_to_articles.items()}

    # Remove articles whose content is 'blah blah blah'
    extra_words = ['reuter', 'said', 'also', 'would']
    titles_to_tokens = {title: remove_stop_words(tokens, extra_words=extra_words)
                        for title, tokens in titles_to_tokens.items() if 'blah' not in tokens}
    titles_to_tokens_stem = {title: stem_tokens(tokens) for title, tokens in titles_to_tokens.items()}

    docs_gen = list(titles_to_tokens_stem) # list of document titles
    data_gen = list(titles_to_tokens_stem.values()) # list of words for each docs
    
    ##### Second Run the algorithm
    t0 = time()
    lda_model, corpus = gensim_lda(K, data_gen)
    doc_n = 0
    topic_spec_doc_gen(lda_model, corpus, doc_n)
    
    doc_lda = lda_model[corpus]
    print("done in %0.3fs." % (time() - t0))
    
    # Print the Keyword in each topics
    pprint(lda_model.print_topics())
    
    print("\nTopics proportions in Doc %s :" % docs_gen[doc_n])
    topic_spec_doc_gen(lda_model,corpus, doc_n)
