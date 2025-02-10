'''
Daily challenge: Old MacDonald’s Farm

Instructions : Old MacDonald’s Farm
Take a look at the following code and output:
File: market.py

'''



class Farm:

    def __init__(self, farm_name):
        self.farm_name = farm_name
        self.animal_list = {}  # a list of animals (a dict)


    def add_animal(self, animal_name, amount=1):
        animal_name = animal_name.lower()
        if animal_name not in self.animal_list:
            self.animal_list[animal_name] = amount  # adding the animal for the first time, with the passed number
        else:
            self.animal_list[animal_name] += amount  # increase the number of animals, for an animal that already exists in the dict


    def get_info(self):
        
        print (f"{self.farm_name}'s farm\n")

        # printing the dict
        for animal,amount in self.animal_list.items():
            print (f"{animal} : {amount}")

        print("\n\tE-I-E-I-0!\n")
        return ""
    

    def get_animal_types(self):
        return sorted([animal for animal in self.animal_list.keys()])  # sorted list comprehension
    

    def get_short_info(self):

        print (f"{self.farm_name}'s farm has ", end="")
        sorted_list = self.get_animal_types()
        list_len = len(sorted_list)

        if list_len == 0:
            print (f"no animals.")
            return ""

        if list_len == 1:
            print (f"{sorted_list[0]}s.")
            return ""

        if list_len == 2:
            print (f"{sorted_list[0]}s and {sorted_list[1]}s.")
            return ""
    
        for index in range(list_len-2):
            print(sorted_list[index], "s, ", sep="", end="")
        print (f"{sorted_list[list_len-2]}s and {sorted_list[list_len-1]}s.")

        return ""
        



macdonald = Farm("McDonald")
macdonald.add_animal('cow',5)
macdonald.add_animal('shEep')
macdonald.add_animal('sheep')
macdonald.add_animal('goat', 12)
macdonald.add_animal('dog', 7)

print(macdonald.get_info())

print(macdonald.get_animal_types())

print (macdonald.get_short_info())



