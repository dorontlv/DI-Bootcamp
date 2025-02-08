'''
Daily challenge: Challenges

'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
Challenge 1 : Sorting

Write a program that accepts a comma separated sequence of words as input and prints the words in a comma-separated sequence after sorting them alphabetically.
Use List Comprehension

'''

sentence = input("Please enter a comma separated sequence of words: ")

words = [word for word in sorted(sentence.split(","))]

print (",".join(words))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
Challenge 2 : Longest Word

Write a function that finds the longest word in a sentence.

'''

def longest_word(string):
    
    words = string.split(" ")

    longest_so_far = 0
    the_longest_word = ""  # the longest word in the string
    current_word_len = 0

    for word in words:
        current_word_len = len(word)
        if  current_word_len > longest_so_far:
            longest_so_far = current_word_len
            the_longest_word = word

    return the_longest_word


print (longest_word("Margaret's toy is a pretty doll."))
print (longest_word("A thing of beauty is a joy forever."))
print (longest_word("Forgetfulness is by all means powerless!"))

