'''
Exercises XP

What you will learn :
Inheritance

'''


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 1 : Pets")


class Pets():
    def __init__(self, animals):
        self.animals = animals

    def walk(self):
        for animal in self.animals:
            print(animal.walk())

class Cat():
    is_lazy = True

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def walk(self):
        return f'{self.name} is just walking around'


class Bengal(Cat):
    def sing(self, sounds):
        return f'{sounds}'


class Chartreux(Cat):
    def sing(self, sounds):
        return f'{sounds}'


class Siamese(Cat):
    def sing(self, sounds):
        return f'{sounds}'
    

x = Bengal("Bengal_name", 5)
y = Chartreux("Chartreux_name", 6)
z = Siamese("Siamese_name", 7)

all_cats = [x,y,z]

sara_pets = Pets(all_cats)
sara_pets.walk()



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 4 : Family")


class Family:
    
    def __init__(self, members:list, last_name:str):
        self.members = members
        self.last_name = last_name

    # adding a child to the family
    def born(self, **kwargs):  # kwargs is a dict
        self.members.append(kwargs)
        print("Congratulations for having a new child in the family !")


    def is_18(self, name):
        for member in self.members:
            if member['name'] == name:  # looking for that name in the dict
                return (member['age'] > 18)

    def family_presentation(self):
        print(f"The family last name is: {self.last_name}")
        print("The family members are:")
        for member in self.members:
            for key,value in member.items():  # for each dict take the key and value
                if key != "is_child":
                    print(f"{key}:{value}", end="  ")
            if member["is_child"]:  # The is_child key is a special case
                print ("a child")
            else:
                print("a prent")





one_family = Family(
[
        {'name':'Michael','age':35,'gender':'Male','is_child':False},
        {'name':'Sarah','age':32,'gender':'Female','is_child':False}
]
,
"Regular family"
)

one_family.born(name="Dani", age=16, gender="Male", is_child=True)

one_family.is_18("Sarah")
one_family.is_18("Dani")

one_family.family_presentation()


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 5 : TheIncredibles Family")



class TheIncredibles(Family):

    def use_power(self, name):
        for member in self.members:
            if member['name'] == name:  # looking for that name in the dict
                if member['age'] > 18:
                    print (f"The power of {name} is: {member['power']}.")  # only if over 18
                else:
                    raise Exception(f"{name} is not over 18 years old.")  # raise an exception


    def incredible_presentation(self):
        print ("Here is our powerful family:")
        super().family_presentation()  # call the parent class method
        



another_family = TheIncredibles(
[
        {'name':'Michael','age':35,'gender':'Male','is_child':False,'power': 'fly','incredible_name':'MikeFly'},
        {'name':'Sarah','age':32,'gender':'Female','is_child':False,'power': 'read minds','incredible_name':'SuperWoman'}
]
,
"Incredibles"
)

another_family.use_power("Michael")
another_family.use_power("Jack")

another_family.incredible_presentation()
another_family.born(name="Jack", age=0, gender="Male", is_child=True, power='Unknown Power',incredible_name='GreatBaby')
another_family.incredible_presentation()
another_family.use_power("Jack")

