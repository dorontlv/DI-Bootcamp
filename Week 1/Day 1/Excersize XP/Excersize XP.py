
# Exercise 1 : Hello World

print ("Hello world\n" * 4)


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



# Exercise 2 : Some Math

print ((99*3)*8)

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


""" 
Exercise 3 : What is the output ?

>>> 5 < 3  # False
>>> 3 == 3  # True
>>> 3 == "3"  # False
>>> "3" > 3  # Error
>>> "Hello" == "hello"  # False
"""

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

""" 
🌟 Exercise 4 : Your computer brand
"""
computer_brand = "Lenovo"
print (f"I have a {computer_brand} computer")

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

""" 
🌟 Exercise 5 : Your information
"""

name = "Doron"
age = 222
shoe_size = 43
info = "My name is " + name + " and my age is " + str(age) + " and my shoe size is " + str(shoe_size)
print (info)

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

""" 
🌟 Exercise 6 : A & B
"""

a = 5
b = 6
if a > b:
    print ("Hello World")

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

""" 
Exercise 7 : Odd or Even
"""

number = int(input("Enter a number: "))
if number%2 == 1:
    print("The number is odd")
else:
    print("The number is even")

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

""" 
🌟 Exercise 8 : What’s your name ?
"""

name = input("what is your name ? ")
if name == "Doron":
    print("That's funny, we have the same name !")


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

""" 
🌟 Exercise 9 : Tall enough to ride a roller coaster
"""

height = int(input("What is your height? "))
if height > 145:
    print("You are tall enough to ride.")
else:
    print("You need to grow some more to ride.")


