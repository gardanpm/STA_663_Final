from src.utility import *

if __name__ == '__main__':

    with open('../Data/reuters21578/reut2-000.sgm') as f:
        data = f.read()

    titles_to_articles = parse_sgm_file(data)
    titles_to_tokens = {title: tokenize_doc(doc) for title, doc in titles_to_articles.items()}

    # Remove articles whose content is 'blah blah blah'
    titles_to_tokens = {title: remove_stop_words(tokens, extra_words='reuter')
                        for title, tokens in titles_to_tokens.items() if 'blah' not in tokens}
    titles_to_tokens_stem = {title: stem_tokens(tokens) for title, tokens in titles_to_tokens.items()}