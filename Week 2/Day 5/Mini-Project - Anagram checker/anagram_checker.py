'''

We will create a program that will ask the user for a word.
It will check if the word is a valid English word, and then find all possible anagrams for that word.

'''

from itertools import permutations

class AnagramChecker:
    
    def __init__(self):
        # reading the file
        fh = open('sowpods.txt')
        self.english = fh.read().lower().split("\n")  # read the file, make all words lowercase, and split all lines (words) into a list.
        fh.close()

    def is_valid_word(self, word):
        # should check if the given word (ie. the word of the user) is a valid word.
        return word in self.english  # simply check if the word is in the list (the english dictionary)

    # get ALL possible anagrams, even if it's not a valid english word
    def get_anagrams(self, word):
        # find all anagrams for the given text.
        # use a python built-in func 'permutations'.
        return set([''.join(permutation) for permutation in permutations(word)])


