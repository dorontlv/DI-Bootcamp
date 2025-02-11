'''

Exercises XP

What you will learn
Dunder Methods
Modules

'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Exercise 1: Currencies")


class Currency:
    def __init__(self, currency, amount):
        self.currency = currency
        self.amount = amount

    # My code starts HERE

    def __str__(self):
        return f"{self.amount} {self.currency}"

    def __int__(self):
        return self.amount
    
    def __repr__(self):
        return str(self)


    def __add__(self, other):
        if isinstance(other, int):  # check if the other object is an int
            return self.amount + other
        
        if isinstance(other, Currency):  # check if the other object is of type Currency
            if self.currency == other.currency:
                return Currency(self.currency, self.amount + other.amount)
            else:
                raise Exception(f"Cannot add between Currency type <{self.currency}> and <{other.currency}>")  # raise an exception

        return ""  # return an empty string


    def __iadd__(self, other):
        if isinstance(other, int):
            return Currency(self.currency, self.amount + other)
        
        if isinstance(other, Currency):
            if self.currency == other.currency:
                return Currency(self.currency, self.amount + other.amount)
            else:
                raise Exception(f"Cannot add between Currency type <{self.currency}> and <{other.currency}>")  # raise an exception

        return ""
    




c1 = Currency('dollar', 5)
c2 = Currency('dollar', 10)
c3 = Currency('shekel', 1)
c4 = Currency('shekel', 10)

print(str(c1))  # '5 dollars'

print(int(c1))  # 5

print(repr(c1))  # '5 dollars'

print(c1 + 5)  # 10

print(c1 + c2)  # 15 dollars

print(c1)  # 5 dollars  # this is actually print(repr(c1)) .  printing an object is actually using the repr function.

c1 += 5
print(c1)  # 10 dollars

c1 += c2
print(c1)  # 20 dollars

c1 + c3  # TypeError: Cannot add between Currency type <dollar> and <shekel>


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 2: Import")


'''

import module_name 

OR 

from module_name import function_name 

OR 

from module_name import function_name as fn 

OR

import module_name as mn

'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 3: String module")


import random

string = ""

for index in range(5):
    letter = chr(ord('a') + random.randint(1,26) -1)  # randomly choose a letter between a...z
    if random.randint(0,1):  # randomly uppercase or not
        letter = letter.upper()
    string += letter

print(string)


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 4 : Current Date")



import datetime

def get_current_date():

    today_date = datetime.date.today()

    print(f"Today is the {today_date.strftime("%d/%m/%Y")}")


get_current_date()


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 5 : Amount of time left until January 1st")


from datetime import datetime

def time_till_1_jan():

    now = datetime.now()  # current moment
    
    next_year = now.year + 1
    jan_1 = datetime(next_year, 1, 1)  # 1.1.2026
    time_diff = jan_1 - now

    days = time_diff.days
    total_seconds = time_diff.seconds  # seconds this year

    seconds = total_seconds % 60
    hours = int(total_seconds / 3600)
    minutes = int((total_seconds % 3600) / 60)

    print (f"The 1st of January is in {days} days and {hours}:{minutes}:{seconds} hours).")


time_till_1_jan()


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 6 : Birthday and minutes")



from datetime import datetime

def minutes_lived_from_birth(birthdate):
    
    birthdate_dt = datetime.strptime(birthdate, "%Y-%m-%d")  # format it into a datetime object
    now = datetime.now()  # current moment

    time_lived = now - birthdate_dt  # datetime object
    
    minutes = int(time_lived.total_seconds() / 60)  # total seconds in time lived
    
    print(f"From birth till now, you have lived {minutes} minutes.")



birthdate = input("Please enter your birthdate [YYYY-MM-YY]: ")

minutes_lived_from_birth(birthdate)



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 7 : Faker Module")



from faker import Faker

def add_dict(users:list):
    fake = Faker()
    users.append( {'name':fake.name(), 'address':fake.address(), 'language_code':fake.country()} )



users = []  # a list of dicts

# add 10 dicts into the users list
for _ in range(10):
    add_dict(users)

for user in users:
    print(user)


