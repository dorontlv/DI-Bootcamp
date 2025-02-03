'''
Exercises XP

'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 1 : Favorite Numbers")


my_fav_numbers = {1,2,3}
my_fav_numbers.add(4)
my_fav_numbers.add(5)
print(my_fav_numbers)

# convert a set to a list, so that we can remove an item
my_fav_numbers_set_converted_to_a_list = list (my_fav_numbers)
print(my_fav_numbers_set_converted_to_a_list)

# remove the last item
my_fav_numbers_set_converted_to_a_list.pop()
print(my_fav_numbers_set_converted_to_a_list)

friend_fav_numbers = {6,7}

# Concatenate my_fav_numbers and friend_fav_numbers to a new list.
our_fav_numbers = my_fav_numbers.union(friend_fav_numbers)
print(our_fav_numbers)


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 2 : a tuple is immutable")


'''
Given a tuple which value is integers, is it possible to add more integers to the tuple?

ANSWER:
No - a tuple is immutable

'''


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 3: List")


fruits = ["Banana", "Apples", "Oranges", "Blueberries"]
fruits.remove("Banana")
fruits.remove("Blueberries")
fruits.append("Kiwi")
fruits.insert(0, "Apples")

print (fruits)

# Count how many apples are in the basket.
count = 0
for item in fruits:
    if item == "Apples":
        count += 1
print (f"There are {count} Apples in the basket.")

fruits.clear()

print(fruits)


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 4: Floats")


'''
Recap – What is a float? What is the difference between an integer and a float?

ANSWER:
A float number is when you have a decimal point.
An integer doesn't have a decimal point.

'''

numbers = []

i = 1.5
while i <= 5:
    if (i == int(i)):  # check if it's a rounded number or a decimal
        numbers.append(int(i))  # if it's a rounded number than add it as an int to the list
    else:
        numbers.append(i)  # otherwise add it as a float to the list
    i += 0.5
# so we get a list with mixed types

# Another way to generate a sequence of floats
numbers = []
for item in range(3, 10+1):  # 3 till 10, but then we devide by 2 so we get 1.5 till 5
    numbers.append(item/2.0)  # we devide by 2.0 so we get float numbers

print (numbers)


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 5: For Loop")


# print all numbers from 1 to 20
for num in range(1,20+1):
    print (num)

print("")

index = 0  # the first element is index 0
for num in range(1,20+1):
    if index%2==0:  # print out every element which has an even index.
        print (num)
    index +=1


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 6 : While Loop")


while True:
    name = input("What is your name ? ")
    if name == "Doron":
        break


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 7: Favorite fruits")


print ("Please input your favorite fruit(s) (one or several fruits).")
print ("Separate the fruit list with a single space.")

fruits = input("What are your fruits ? ")

fruit_list = fruits.split()  # split a string into a list of words

print(fruits)
print(fruit_list)

one_fruit = input("Now please choose one fruit: ")

in_list = False

# Search to see if the fruit is in the list.
for item in fruit_list:
    if item == one_fruit:
        in_list = True
        break

if in_list:
    print ("You chose one of your favorite fruits! Enjoy!")
else:
    print ("You chose a new fruit. I hope you enjoy.")


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 8: Who ordered a pizza ?")


topping_list = []
while True:
    topping = input("Enter a pizza topping name.  Enter quit to end: ")
    if topping == "quit":
        break
    else:
        print ("I’ll add that topping to your pizza.")
        topping_list.append(topping)  # a list of toppings

price = 10  # the basic price of a pizza
print ("These are the toppings you ordered:")
for item in topping_list:
    print(item)
    price += 2.5  # every topping costs 2.5 shekels

print (f"For the pizza you ordered, you'll have to pay {price} shekels.")



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 9: Cinemax")


total_cost = 0

while True:
    age = int(input("What is the age of the family member ? (If there are no more family members then please enter the age 0) : "))
    if age == 0:  # no more family members
        break
    # a different cost for different ages
    if age>=3 and age<=12:
        total_cost += 10
    elif age>12:
        total_cost += 15

print (f"The total cost is: {total_cost}")
print("")

# ~~~~~~

print ("Check who is permitted to watch the movie.")

name_list = ['name1', 'name2', 'name3', 'name4']
permitted_list = []

# Create the permitted list of names
for item in name_list:
    the_age = int(input(f"What is the age of {item} ? "))
    if the_age > 21:
        permitted_list.append(item)

print("These are the people who are allowed to view the movie:")
for item in permitted_list:
    print(item)


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 10 : Sandwich Orders")


sandwich_orders = ["Tuna sandwich", "Pastrami sandwich", "Avocado sandwich", "Pastrami sandwich", "Egg sandwich", "Chicken sandwich", "Pastrami sandwich"]

# We remove all "Pastrami sandwich" instances
while "Pastrami sandwich" in sandwich_orders:
    sandwich_orders.remove("Pastrami sandwich")

finished_sandwiches = []

while len(sandwich_orders)!=0:
    sandwich = sandwich_orders.pop()  # pop one item from the list
    finished_sandwiches.append(sandwich)  # add this item to the second list

# print all the finished sandwiches
for item in finished_sandwiches:
    print(f"I made your {item}")
    
