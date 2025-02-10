print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 3 : Dogs Domesticated")


import random

import DogFile

class PetDog(DogFile.Dog):  # we inherit from Dog
    
    def __init__(self,  name, age, weight, trained = False):
        self.trained = trained
        super().__init__(name, age, weight)  # running the parent's class init

    def train(self):
        print (self.bark())
        self.trained = True

    def play(self, *dog_names):  # dog_names is a tuple
        print(f"{self.name}, ", end="")  # print our own dog's name
        print(", ".join(dog_names), end="")  # together with all the other dogs' names
        print (" all play together.\n")

    def do_a_trick(self):
        
        if (self.trained):
            num = random.randint(1, 4)  # random number
            sentences = [   "does a barrel roll",
                            "stands on his back legs",
                            "shakes your hand",
                            "plays dead"
                        ]
            print (self.name, sentences[num-1])  # print a random sentence



aa = PetDog("Dog4", 14, 40)
aa.train()
aa.play(DogFile.x.name, DogFile.y.name, DogFile.z.name)
aa.do_a_trick()


