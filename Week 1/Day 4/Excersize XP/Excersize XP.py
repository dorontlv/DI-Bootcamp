'''
Exercises XP

'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 1 : What are you learning ?")


def display_message():
    print ("I'm stuying python now.")

display_message()



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 2: What’s your favorite book ?")


'''
Write a function called favorite_book() that accepts one parameter called title.
The function should print a message, such as "One of my favorite books is <title>".

'''


def favorite_book(title):
    print (f"One of my favorite books is {title}")

favorite_book("Alice in Wonderland")



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 3 : Some Geography")


'''
Write a function called describe_city() that accepts the name of a city and its country as parameters.
The function should print a simple sentence, such as "<city> is in <country>".

'''

def describe_city(city, country = "Israel"):
    print (f"{city} is in {country}")

describe_city("Tel Aviv", "UK")
describe_city("Tel Aviv")



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 4 : Random")


'''
Create a function that accepts a number between 1 and 100 and generates another number randomly between 1 and 100. Use the random module.
Compare the two numbers, if it’s the same number, display a success message, otherwise show a fail message and display both numbers.

'''

import random

def generate_number(my_number):

    # a random number between 1 and 100
    num = random.randint(1, 100)

    if (num == my_number):
        print ("Success - it's the same number.")
    else:
        print ("You guessed it wrong.")
        print (f"Random number is {num}")
        print (f"Your guess is {my_number}")


num = int(input("Choose a number between 1 and 100: "))
generate_number(num)


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 5 : Let’s create some personalized shirts !")


'''
Write a function called make_shirt() that accepts a size and the text of a message that should be printed on the shirt.
The function should print a sentence summarizing the size of the shirt and the message printed on it, such as "The size of the shirt is <size> and the text is <text>"

'''

# Modify the make_shirt() function so that shirts are large by default with a message that reads “I love Python” by default.
def make_shirt(size = "Large", text = "I love Python"):
    print ("The size of the shirt is {size} and the text is {text}")


# Call the function make_shirt().
make_shirt("Small", "This is my message on the shirt")

# Call the function, in order to make a large shirt with the default message
make_shirt()

# Make medium shirt with the default message
make_shirt("Medium")

# Make a shirt of any size with a different message.
make_shirt("Any size that I want", "Any message that I want")

# Bonus: Call the function make_shirt() using keyword arguments.
make_shirt(size="Any size", text="My message")



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 6 : Magicians …")


magician_names = ['Harry Houdini', 'David Blaine', 'Criss Angel']

print ("\n\n*** Juliana said that this excersize should be implemented without passing arguments to the functions. ***\n")


# prints the name of each magician in the list.
def show_magicians():
    for name in magician_names:
        print (name)

show_magicians()


# modify the original list of magicians by adding the phrase "the Great" to each magician’s name.
def make_great():
    # we are using the enumerate and the index so that we can change the list
    for i, name in enumerate(magician_names):
        magician_names[i] += " the Great"

# Call the function make_great().
make_great()

# Call the function show_magicians() to see that the list has actually been modified.
show_magicians()



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 7 : Temperature Advice")


import random

def get_random_temp(season = ""):

    # random.randint()  # this is how you get a random integer

    if season == "spring":
        return round((random.random()*4 + 21), 1) #  this is a float between 21 to 26

    if season == "summer":
        return round((random.random()*12 + 27), 1) #  this is a float between 27 to 39
    
    if season == "autumn":
        return round((random.random()*3 + 17), 1) #  this is a float between 17 to 20
    
    if season == "winter":
        return round((random.random()*26 - 10), 1) #  this is a float between -10 to 16
    
    # if no season is selected
    return round((random.random()*49 - 10), 1) #  this is a float between -10 to 39



print (get_random_temp())


def main():

    # the user will choose a season name
    season_choise = input("Please choose a season: ")

    temperature = get_random_temp(season_choise)
    
    print (f"The temperature right now is {temperature} degrees Celsius.")

    # print a message according to the temperature.
    if temperature < 0:
        print ("Brrr, that’s freezing! Wear some extra layers today.")
    elif temperature>=0 and temperature<16:
        print ("Quite chilly! Don’t forget your coat.")
    elif temperature>=16 and temperature<23:
        print ("A bit better.")
    elif temperature>=23 and temperature<32:
        print ("Even warmer.")
    elif temperature>=32 and temperature<40:
        print ("Very hot")

    # the user will choose a month numer instead of a season
    month = int(input("Please choose a month numner [1-12]: "))

    if month >=4 and month <=6:
        season_choise = "spring"
    elif month >=7 and month <=8:
        season_choise = "summer"
    elif month >=9 and month <=11:
        season_choise = "autumn"
    else:
        season_choise = "winter"

    temperature = get_random_temp(season_choise)
    
    print (f"The temperature right now is {temperature} degrees Celsius.")



main()



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 8 : Star Wars Quiz")


'''
This project allows users to take a quiz to test their Star Wars knowledge.
The number of correct/incorrect answers are tracked and the user receives different messages depending on how well they did on the quiz.

'''

# Here is an array of dictionaries, containing those questions and answers

data = [
    {
        "question": "What is Baby Yoda's real name?",
        "answer": "Grogu"
    },
    {
        "question": "Where did Obi-Wan take Luke after his birth?",
        "answer": "Tatooine"
    },
    {
        "question": "What year did the first Star Wars movie come out?",
        "answer": "1977"
    },
    {
        "question": "Who built C-3PO?",
        "answer": "Anakin Skywalker"
    },
    {
        "question": "Anakin Skywalker grew up to be who?",
        "answer": "Darth Vader"
    },
    {
        "question": "What species is Chewbacca?",
        "answer": "Wookiee"
    }
]

num_of_correct_answers = 0

# a list of wrong_answers
wrong_answers = []

# a function that asks the questions to the user, and check his answers.
# Track the number of correct, incorrect answers.
def ask_questions():

    global num_of_correct_answers
    
    for item in data:
        answer = input(item["question"] + " ")  # ask the user for an answer to the question
        if answer == item["answer"]:
            num_of_correct_answers += 1  # correct answer
        else:
            wrong_answers.append( {"question" : item["question"] , "answer" : item["answer"] , "wrong answer" : answer } )  # incorrect answer - append the info to a new list


# a function that informs the user of his number of correct/incorrect answers.
def inform_number_of_correct_incorrect():
    
    global num_of_correct_answers
    
    print(f"The number of correct answers is: {num_of_correct_answers}")
    print(f"The number of incorrect answers is: {len(wrong_answers)}")


# Bonus : display to the user the questions he answered wrong, his answer, and the correct answer.
def questions_you_answered_wrong():

    print("These are your wrong answers:")
    for item in wrong_answers:
        print(f"Question: {item["question"]}")
        print(f"Correct answer: {item["answer"]}")
        print(f"Your wrong answer: {item["wrong answer"]}")


# If he had more then 3 wrong answers, ask him to play again.
while (True):
    ask_questions()
    inform_number_of_correct_incorrect()
    questions_you_answered_wrong()
    if len(wrong_answers) <= 3 :
        break
    else:
        print ("You will have to take the quiz again.")
        num_of_correct_answers = 0
        wrong_answers = []

