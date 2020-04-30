# Implement SKLEARN LatentDirichletAllocation with varitional Bayes on the Reuter dataset
# Ref: Olivier Grisel 
#      Lars Buitinck
#      Chyi-Kwei Yau 
from src.utility import *
from Comparisons.print_sklearn import*
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from time import time

if __name__ == '__main__':
    n_topics = 10
    n_top_words = 10
    n_samples = 10

    ###### First get the data ready similarly to our implemented example
    with open('C:/Users/pgard/STA_663_Final_para/Data/reuters21578/reut2-000.sgm') as f:
            data = f.read()

    title_docs = parse_sgm_file(data)
    titles_to_tokens = {title: tokenize_doc(doc) for title, doc in title_docs.items()}
    # Remove articles whose content is 'blah blah blah'
    extra_words = ['reuter', 'said', 'also', 'would']
    titles_to_tokens = {title: remove_stop_words(tokens, extra_words=extra_words) 
                        for title, tokens in titles_to_tokens.items() if 'blah' not in tokens}
    titles_to_tokens_stem = {title: stem_tokens(tokens) for title, tokens in titles_to_tokens.items()}

    # Transforming the data to a list of texts according to the required format for the count vectorizer
    data_skl = list(titles_to_tokens_stem.values())
    for i in range(len(data_skl)):
        data_skl[i] = ' '.join((  # note double parens, join() takes an iterable
            data_skl[i]
        ))
    # Getting list of doc titles 
    docs_skl = list(titles_to_tokens_stem)
    
    ##### Second run the algorithm
    t0 = time()
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer()

    tf = tf_vectorizer.fit_transform(data_skl)
    print()

    print("Fitting LDA models with tf features, "
          "n_samples=%d ..."
          % (n_samples))

    lda_skl = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=0)
    lda_skl.fit(tf) #fititng model
    print("done in %0.3fs." % (time() - t0))
    
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda_skl, tf_feature_names, n_top_words) #Printing n top words

    doc_n = 0 # get topics for a given doc:
    print("\nTopics in Doc %s :" % docs_skl[doc_n])
    topics_spec_doc(lda_skl, tf, n_topics, doc_n)
