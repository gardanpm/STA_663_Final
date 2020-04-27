from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def remove_stop_words(iden_to_tokens, extra_words=None):
    if extra_words is None:
        extra_words = []

    stop_words = stopwords.words('English')
    stop_words += extra_words

    no_stop = {iden: [token for token in tokens if token not in stopwords]
               for iden, tokens in iden_to_tokens.items()}
    return no_stop


def stem_words(iden_to_tokens):
    ps = PorterStemmer
    iden_to_tokens_stemmed = {iden: [ps.stem(w) for w in tokens]
                              for iden, tokens in iden_to_tokens.items()}
    return iden_to_tokens_stemmed