print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Exercise 2 : Dogs")


class Dog:

    def __init__(self, name, age, weight):
        self.name = name
        self.age = age
        self.weight = weight

    def bark(self):
        return f"{self.name} is barking."
    
    def run_speed(self):
        return (self.weight/self.age*10)
    
    def fight(self, other_dog):
        speed1 = other_dog.run_speed() * other_dog.weight
        speed2 = self.run_speed() * self.weight
        if speed1 > speed2:
            return f"{other_dog.name} won the fight"
        else:
            return f"{self.name} won the fight"



x = Dog("Dog1", 11, 10)
y = Dog("Dog2", 12, 20)
z = Dog("Dog3", 13, 30)

whos_barking = x.bark()
his_speed = x.run_speed()
who_won = x.fight(y)


