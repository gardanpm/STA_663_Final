# This file tests test some of the functions we made to make sure they return what is expected of them

from src.sampler import *
from src.utility import *
from src.inference import *


if __name__ == "__main__":

    # Testing unique words
    test_list1 = get_unique_words([['a', 'b'], ['c','a','d'], ['w', 'd','b'],'i'])
    test_list2 = ['a','b','c','d','w','i']

    # sorting both the lists
    test_list1.sort()
    test_list2.sort()

    # using == to check if
    # lists are equal
    if test_list1 == test_list2:
        print ("The lists 1 and 2 are identical")
    else :
        print ("The lists 1 and 2 are not identical")


    test_list2 = ['a','b','c','d','w','i']
    matrix1 = np.array([[0, 0, 0, .05, 1, .95], [1, .95,0,0,0,.05], [0,.05,1,.95,0,0]])
    # We can read the first row as the probability of word 'w' to be assigned to topic 1 is ,
    # the probability of 'i is .95 and the probability of 'd is .05

    # We expect a dictionary in order of the probabilities of the words for topc k
    assert get_top_n_words(matrix1, 2, test_list2) == {1: ['w', 'i'], 2: ['a', 'b'], 3: ['c', 'd']}, 'Does not return the right words'

    # Checking 'tokenize_doc'
    string_test = "We EAT on the coffee-table! U.S.A. '56'"
    assert tokenize_doc(string_test) == ['we', 'eat', 'on', 'the', 'coffeetable', 'usa', '56'], "Sould return ['we', 'eat', 'on', 'the', 'coffeetable', 'usa', '56']"
    tokens = tokenize_doc(string_test)

    # Checking 'remove_stop_words'
    extra_words = ['usa']
    # stopwords remove words such as "to", "in", "on"
    # we don't want numbers nor 'usa'
    tokens = remove_stop_words(tokens, remove_numbers=True, tokens_have_quotes=False, extra_words=extra_words)
    assert tokens == ['eat', 'coffeetable'], "should return ['eat', 'coffeetable']"

    # Checking 'stem_tokens'
    # remove some endings such as able, ing, ...
    assert stem_tokens(tokens) == ['eat', 'coffeet'], "Should return ['eat', 'coffeet']"

    print("All the tests are conclusive!")