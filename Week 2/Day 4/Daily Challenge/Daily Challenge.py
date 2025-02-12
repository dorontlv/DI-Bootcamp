'''

Daily challenge : Text Analysis

What You will learn :
OOP
Modules


'''


class Text:
    
    def __init__(self, sentence:str):
        
        self.sentence = sentence

        self.sentence = self.sentence.replace(".", "")      # remove period
        self.sentence = self.sentence.replace(",", "")      # remove comma
        self.sentence = self.sentence.replace("\n", " ")    # remove newline
        
        #  we need to remove all doublespaces - over and over again - until there are no doublespaces
        new_sentence = self.sentence.replace("  ", " ")
        while new_sentence != self.sentence:
            self.sentence = new_sentence
            new_sentence = self.sentence.replace("  ", " ")    # remove doublespace - put only one space instead

        # split the entire text into a list of words
        list_of_words = self.sentence.split(" ")

        self.word_frequencies = {}  # a dict of {word, frequency}

        # add the words into the dict, with the correct frequency
        for item in list_of_words:
            if item not in self.word_frequencies:
                self.word_frequencies[item] = 1  # first time we're adding the word to the dict - so it's frequency 1
            else:
                self.word_frequencies[item] += 1  # increase the frequency


    def frequency(self, word):

        if word in self.word_frequencies:
            return self.word_frequencies[word]  # return the frequency - it's in the dict
        
        return 0


    # the most common word is the word with the maximum frequency
    def most_common(self):
        maximum = 0
        the_word = ""
        for word,count in self.word_frequencies.items():
            if count > maximum:  # look for the maximum frequenct
                the_word = word
                maximum = count

        return the_word


    # all the words that have a frequency of 1 (unique words)
    def unique(self):
        return [word for word,count in self.word_frequencies.items() if count==1]


    @classmethod
    def from_file(cls, filename:str):
        # read the file
        f = open(filename)
        data = f.read()
        f.close()
        return Text(data)



sentence = "A good book would sometimes cost as much as a good house."


x = Text(sentence)
print (x.frequency("as"))
print(x.most_common())
print(x.unique())

print ("")

# reading from a file
y = Text.from_file("the_stranger.txt")
print (y.frequency("the"))
print(y.most_common())
print(y.unique())

