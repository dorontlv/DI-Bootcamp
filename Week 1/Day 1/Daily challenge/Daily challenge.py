""" 
Daily challenge: Build up a string


What You will learn:
Python Basics
Conditionals
Loops

"""

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

"""
1.
Using the input function, ask the user for a string. The string must be 10 characters long.
If it’s less than 10 characters, print a message which states “string not long enough”.
If it’s more than 10 characters, print a message which states “string too long”.
If it’s 10 characters, print a message which states “perfect string” and continue the exercise.
"""

my_string = input("1. Please enter a 10 chars long string: ")
if len(my_string) < 10:
    print ("string not long enough")
elif len(my_string) > 10:
    print ("string too long")
else:
    print ("perfect string")

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

""" 
2.
Then, print the first and last characters of the given text.
The user enters "HelloWorld"
Then you have to print 
H
d
"""

my_string = input("2. Please enter a sentence: ")
print (my_string[0])  # first char
print (my_string[len(my_string)-1])  # last char

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

""" 
3.
Using a for loop, construct the string character by character: Print the first character, then the second, then the third, until the full string is printed. For example:

The user enters "HelloWorld"
Then, you have to construct the string character by character
H
He
Hel
... etc
HelloWorld
"""

my_string = input("3. Please enter a sentence: ")
new_string = ""

for chr in my_string:
    new_string += chr
    print(new_string)

print ("\n")

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

""" 
4.
Bonus: Swap some characters around then print the newly jumbled string (hint: look into the shuffle method). For example:

Hlrolelwod
"""

import random

my_string = input("4. Please enter a sentence: ")
string_converted_to_list = list(my_string)      # Convert the string to a list of characters
random.shuffle(string_converted_to_list)        # shuffle it
my_string = ''.join(string_converted_to_list)   # Join the shuffled list back into a string
print (my_string)

