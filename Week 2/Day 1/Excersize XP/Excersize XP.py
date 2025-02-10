'''
Exercises XP

'''


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 1: Cats\n")


'''
Exercise 1: Cats

'''

class Cat:
    def __init__(self, cat_name, cat_age):
        self.name = cat_name
        self.age = cat_age

cat1 = Cat("Cat name 1", 6)
cat2 = Cat("Cat name 2", 8)
cat3 = Cat("Cat name 3", 7)

largest_age = 0

cats = [cat1, cat2, cat3]

def oldest_cat(cat_list:list) -> Cat :
    global largest_age
    item_index = -1
    for i,cat in enumerate(cat_list):
        if cat.age > largest_age:
            largest_age = cat.age
            item_index = i
    return cat_list[item_index]

oldest = oldest_cat(cats)

print (f"The oldest cat is {oldest.name}, and is {oldest.age} years old.")


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 2 : Dogs\n")


'''
Exercise 2 : Dogs

'''

class Dog:
    def __init__(self, name:str, height:int):
        self.name = name
        self.height = height

    def bark(self):
        print(f"{self.name} goes woof!")
    
    def jump(self):
        print(f"{self.name} jumps {self.height*2} cm high!")
    
    
davids_dog = Dog("Rex", 50)
print(f"David's dog name is: {davids_dog.name}")
print(f"David's dog height is: {davids_dog.height}")
davids_dog.bark()
davids_dog.jump()

sarahs_dog = Dog("Teacup", 20)
print(f"sarah's dog name is: {sarahs_dog.name}")
print(f"sarah's dog height is: {sarahs_dog.height}")
sarahs_dog.bark()
sarahs_dog.jump()

bigger_dog = ""

if davids_dog.height > sarahs_dog.height:
    bigger_dog = davids_dog.name
else:
    bigger_dog = sarahs_dog.name

print(f"The bigger dog is {bigger_dog}.")


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 3 : Who’s the song producer?\n")


'''
Exercise 3 : Who’s the song producer?

stairway= Song(["There’s a lady who's sure","all that glitters is gold", "and she’s buying a stairway to heaven"])

There’s a lady who's sure
all that glitters is gold
and she’s buying a stairway to heaven

'''


class Song:
    def __init__(self, lyrics:list):
        self.lyrics = lyrics

    def sing_me_a_song(self):
        for item in self.lyrics:
            print(item)
        

stairway = Song(["There’s a lady who's sure","all that glitters is gold", "and she’s buying a stairway to heaven"])

stairway.sing_me_a_song()



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 4 : Afternoon at the Zoo\n")


'''
Exercise 4 : Afternoon at the Zoo

Example

{ 
    A: ["Ape"],
    B: ["Baboon", "Bear"],
    C: ['Cat', 'Cougar'],
    E: ['Eel', 'Emu']
}


'''

class Zoo:

    def __init__(self, zoo_name):
        self.animals = []
        self.name = zoo_name
        self.grouped_animals = {}  # an empty dict

    def add_animal(self, name):
        if name.capitalize() not in self.animals:
            self.animals.append(name.capitalize())

    # print the entire animal list
    def get_animals(self):
        print("These are the animals in the zoo:")
        for animal in self.animals:
            print(animal)
        print ("")

    def sell_animal(self, animal_sold):
        if animal_sold.capitalize() in self.animals:
            self.animals.remove(animal_sold.capitalize())  # remove the animal from the list

    # group the animal list into a sorted dict
    def sort_animals(self):
        sorted_animal_list = sorted(self.animals)   # sort the list
        self.grouped_animals.clear()                # clear the dict

        for animal in sorted_animal_list:
            if animal[0] not in self.grouped_animals:
                self.grouped_animals[animal[0]] = [animal]  # adding a list of 1 item to the dict
            else:
                self.grouped_animals[animal[0]].append(animal)  # adding 1 item to a list that already exists in the dict

    # print the entire sorted dict
    def get_groups(self):
        print("Here are the sorted grouped animal list:")
        for letter, animal_list in self.grouped_animals.items():
            print(', '.join(animal_list))
        print ("")





x = Zoo("ramat_gan_safari")
# azoo.add_animal("Giraffe")

x.add_animal("ccc")
x.add_animal("bBb")
x.add_animal("ccC")
x.add_animal("aaa")

x.get_animals()

x.sell_animal("ddd")
x.sell_animal("bbb")

x.get_animals()

x.sort_animals()
x.get_groups()

x.add_animal("apE")

x.sort_animals()
x.get_groups()

x.get_animals()


