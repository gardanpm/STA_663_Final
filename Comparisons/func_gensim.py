import gensim
import gensim.corpora as corpora

def gensim_lda(K, texts): # Create Dictionary
    
    id2word = corpora.Dictionary(texts)
    # Create Corpus 
    
    # Term Document Frequency
    corpus_gen = [id2word.doc2bow(text) for text in texts]
               
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_gen, id2word=id2word, num_topics=K, random_state=100,
                                               update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    return lda_model, corpus_gen
    
def topic_spec_doc_gen(lda_model, corpus, doc_n): # print topic proportions for each documents
    for i, row in enumerate(lda_model[corpus]):
        if i == doc_n :
            prop = row[0] # topic proportion for doc n is in first argument of row
        else:
            break
    print(prop)