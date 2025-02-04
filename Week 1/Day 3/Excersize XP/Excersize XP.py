'''
Exercises XP

'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 1 : Convert lists into dictionaries")

'''
Expected output:
{'Ten': 10, 'Twenty': 20, 'Thirty': 30}

'''

keys = ['Ten', 'Twenty', 'Thirty']
values = [10, 20, 30]

dict1 = dict(zip(keys, values))

print (dict1)


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 2 : Cinemax #2")


'''
Given the following object:
family = {"rick": 43, 'beth': 13, 'morty': 5, 'summer': 8}

How much does each family member have to pay ?

Print out the family’s total cost for the movies.

'''

family = {"rick": 43, 'beth': 13, 'morty': 5, 'summer': 8}

total_cost = 0

for name, age in family.items():
    # a different cost for different ages
    if age < 3:
        print (f"{name} will need to pay nothing.")
    elif age>=3 and age<=12:
        print (f"{name} will need to pay 10$.")
        total_cost += 10
    elif age>12:
        print (f"{name} will need to pay 15$.")
        total_cost += 15

print (f"The total cost for the family is: {total_cost}")
print("\n")


# ~~~~~~~~~~~~~~~~~~~

'''
Bonus: Ask the user to input the names and ages instead of using the provided family variable
Hint: ask the user for names and ages and add them into a family dictionary that is initially empty.

'''

family = {}

print ("Please enter the family members' details.  To end enter the name \'quit\'.")
while True:
    name = input("What is the name of the family member ? ")
    if name == 'quit':  # no more family members
        break
    age = int(input("What is the age of the family member ? "))
    family[name] = age

total_cost = 0

for name, age in family.items():
    # a different cost for different ages
    if age < 3:
        print (f"{name} will need to pay nothing.")
    elif age>=3 and age<=12:
        print (f"{name} will need to pay 10$.")
        total_cost += 10
    elif age>12:
        print (f"{name} will need to pay 15$.")
        total_cost += 15

print (f"The total cost for the family is: {total_cost}")
print("\n")


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 3: Zara")


'''

1. Here is some information about a brand.

2. Create a dictionary called brand which value is the information from part one (turn the info into keys and values).
The values type_of_clothes and international_competitors should be a list. The value of major_color should be a dictionary.

'''
brand  = {
    "name": "Zara" ,
    "creation_date": 1975 ,
    "creator_name": "Amancio Ortega Gaona" ,
    "type_of_clothes": ['men', 'women', 'children', 'home'] ,
    "international_competitors": ['Gap', 'H&M', 'Benetton'] ,
    "number_stores": 7000 ,
    "major_color":{
        "France": ["blue"],
        "Spain": ["red"],
        "US": ["pink", "green"]
    }
}

# 3. Change the number of stores to 2.

brand["number_stores"] = 2

# 4. Use the key [type_of_clothes] to print a sentence that explains who Zaras clients are.

print (f"The clients of {brand['name']} are:")
for client in brand["type_of_clothes"]:
    print (client)

# 5. Add a key called country_creation with a value of Spain.

brand["country_creation"] = "Spain"

# 6. Check if the key international_competitors is in the dictionary. If it is, add the store Desigual.

if "international_competitors" in brand:
    brand["international_competitors"].append("Desigual")

# 7. Delete the information about the date of creation.

del brand["creation_date"]

# 8. Print the last international competitor.

print (brand["international_competitors"][-1])

# 9. Print the major clothes colors in the US.

for color in brand["major_color"]["US"]:
    print (color)

# 10. Print the amount of key value pairs (ie. length of the dictionary).

print (len(brand))

# 11. Print the keys of the dictionary.

for key in brand.keys():
    print (key)

# 12. Create another dictionary called more_on_zara with the following details:

more_on_zara = {
    "creation_date": 1975 ,
    "number_stores": 10000
}

# 13. Use a method to add the information from the dictionary more_on_zara to the dictionary brand.

brand.update(more_on_zara)


# 14. Print the value of the key number_stores.

print (brand["number_stores"])

# What just happened ?

# The "number_stores" value was updated.
# The "creation_date" item was recreated in the dictionary.


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 4 : Disney characters")


'''
Use a for loop to recreate the 1st result. Tip : don’t hardcode the numbers.
Use a for loop to recreate the 2nd result. Tip : don’t hardcode the numbers.
Use a method to recreate the 3rd result. Hint: The 3rd result is sorted alphabetically.

'''

# Use this list :
users = ["Mickey","Minnie","Donald","Ariel","Pluto"]

# Analyse these results :

#1/

# >>> print(disney_users_A)
# {"Mickey": 0, "Minnie": 1, "Donald": 2, "Ariel": 3, "Pluto": 4}

disney_users_A = {}
count = 0
for name in users:
    disney_users_A[name] = count
    count += 1
print(disney_users_A)



#2/

# >>> print(disney_users_B)
# {0: "Mickey",1: "Minnie", 2: "Donald", 3: "Ariel", 4: "Pluto"}

disney_users_B = {}
count = 0
for name in users:
    disney_users_B[count] = name
    count += 1
print(disney_users_B)



#3/ 

# >>> print(disney_users_C)
# {"Ariel": 0, "Donald": 1, "Mickey": 2, "Minnie": 3, "Pluto": 4}

# we will get a sorted list
sorted_users = sorted(users)

disney_users_C = {}

count = 0
for name in sorted_users:
    disney_users_C[name] = count
    count += 1
print(disney_users_C)


'''
Only recreate the 1st result for:
The characters, which names contain the letter “i”.
The characters, which names start with the letter “M” or “P”.

'''

print ("The characters, which names contain the letter i:")
disney_users_A = {}
count = 0
for name in users:
    if 'i' in name:
        disney_users_A[name] = count
        count += 1
print(disney_users_A)

print ("The characters, which names start with the letter M or P:")
disney_users_A = {}
count = 0
for name in users:
    if name[0] in "MP":
        disney_users_A[name] = count
        count += 1
print(disney_users_A)


