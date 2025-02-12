'''

Exercises XP

What you will learn
Working with files

'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
Exercise 1 – Random Sentence Generator

'''

import random

words = []

def get_words_from_file() -> list :

        # reading the file
        fh = open('sowpods.txt')
        data = fh.read()
        fh.close()

        return data.split("\n")  # split into words.  return a ist of words


def get_random_sentence(length):
        global words
        some_words = []
        for _ in range(length):
            some_words.append(words[random.randint(0,len(words)-1)])  # randomly choose words of the entire list
        return some_words
            
            
def main():
       
        global words

        print("This program lets you choose how many words will be randomly chosen - and then creates a sentence.")

        count = int (input("Choosea number of words - between 2 - 20 : "))

        # validate the input
        if (count<2) or (count>20):
            print("You entered a wrong number!  The program will end now.")
            return
        
        words = get_words_from_file()

        several_words = get_random_sentence(count)  # gets some random words of the entire list of words

        random.shuffle(several_words)  # shuffle them randomly
        print (" ".join(several_words).lower())  # print the sentence

        


main()



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''

Exercise 2: Working with JSON

'''

import json

sampleJson = { 
   "company":{ 
      "employee":{ 
         "name":"emma",
         "payable":{ 
            "salary":7000,
            "bonus":800
         }
      }
   }
}

print (sampleJson['company']['employee']['payable']['salary'])

sampleJson['company']['employee']['birth_date'] = "12.2.2025"

# writing into a file
jason_file = "my_file.json"
fh = open(jason_file, "w")
json.dump(sampleJson, fh, indent=5)  # write the dict into the json file
fh.close()


