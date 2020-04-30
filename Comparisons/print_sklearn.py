# Function to print the m pop words
def print_top_words(model, feature_names, n_top_words):
    ''' Print the n to words of a lda model from sklearn package
    : param model: lda model obtained from LatentDirichletAllocation function
    : param feature_names: Names obtained from tf_vectorizer.get_feature_names()
    : param n_top words: number of top words desired
    '''
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

# Function to print proportion of each topics for doc n
def topics_spec_doc(model, data, n_topics, doc_n): # specify the doc you want to look at by inuting its row
    ''' Print the proportion of each topics for a particular doc n with sklearn
    : param model: lda model obtained from LatentDirichletAllocation function
    : param data: dataset used to run the model
    : param n_topics: number of topics desired
    : doc_n: Placement number of the document desired in dataset
    '''
    for topic in range(n_topics):
        message = "Topic #%d: " % topic
        message += str(round(model.transform(data)[doc_n, topic], 4))
        print(message)
    print()