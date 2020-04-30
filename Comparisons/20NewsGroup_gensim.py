# Implement gensim LDA approximation in parallele on the 20NewsGroup dataset
from src.utility import *
from sklearn.datasets import fetch_20newsgroups
# ! pip install gensim
import gensim
import gensim.corpora as corpora
from Comparisons.func_gensim import *
from pprint import pprint
from time import time

if __name__ == '__main__':

    K=20
    dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

    data = dataset['data']
    # Putting each doc in an ordered dictionnary
    title_docs = {}
    for i in range(len(data)):
            title_docs[i] = data[i]
            
    ##### First get the data ready similarly to our implemented example          
    titles_to_tokens = {title: tokenize_doc(doc) for title, doc in title_docs.items()}

    # Remove articles whose content is 'blah blah blah'
    extra_words = ['maxaxaxaxaxaxaxaxaxaxaxaxaxaxax', 'said', 'also', 'would', 'get', 'say', 'go', 'do', 'one']
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
    topic_spec_doc_gen(lda_model,corpus_gen, doc_n)
