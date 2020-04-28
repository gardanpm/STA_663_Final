from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import string


def parse_sgm_file(sgm_data):
    '''
    Returns a dictionary with titles + articles of an SGM file
    '''

    soup = BeautifulSoup(sgm_data, features="html5lib")
    texts = soup.find_all('text')
    title_docs = {}
    for text in texts:
        title = text.findChild('title')

        # Title is non-existent for a few articles
        if title:
            # Use contents because the text has no name; always the last element
            title_docs[title.string.strip()] = text.contents[-1]

    return title_docs


def tokenize_doc(doc):
    doc = doc.lower()
    whitespace = string.whitespace + '\x03'  # End of file char
    trans = str.maketrans(whitespace, ' ' * len(whitespace), string.punctuation)
    doc_no_punc = doc.translate(trans)
    return doc_no_punc.split()


def remove_stop_words(tokens, remove_numbers=True, tokens_have_quotes=False, extra_words=None):
    if extra_words is None:
        extra_words = []

    stop_words = stopwords.words('English')
    stop_words += extra_words

    if not tokens_have_quotes:
        stop_words = set([word.replace('\'', '') for word in stop_words])

    tokens_no_stop = [token for token in tokens if token not in stop_words]
    if remove_numbers:
        tokens_no_stop = [token for token in tokens_no_stop if not token.isnumeric()]
    return tokens_no_stop


def stem_tokens(tokens):
    ps = PorterStemmer()
    tokens_stemmed = [ps.stem(token) for token in tokens]
    return tokens_stemmed
