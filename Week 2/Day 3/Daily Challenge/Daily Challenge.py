'''

Daily Challenge - Circle

What You will learn :
OOP dunder methods

'''

import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def get_radius(self):
        return self.radius
    
    def get_diameter(self):
        return self.radius * 2  # twice the radius
    
    def get_area(self):
        return math.pi * self.radius**2  # Pi*(R^2)

    def __str__(self):  # returning the radius
        return f"{self.radius}"
    
    def __repr__(self):  # getting the attributes of the circle
        return f"The circle's radius is {self.radius}, and its area is {self.get_area()}"
    
    def __add__(self, other):
        
        if not isinstance(other, Circle):  # check if the other object is of type Circle
            return self  # if not then return the current object
        
        return Circle(self.radius + other.radius)
    
    def __lt__(self, other):  # <
        return self.radius < other.radius

    def __gt__(self, other):  # >
        return self.radius > other.radius

    def __eq__(self, other):  # ==
        return self.radius == other.radius
    
    def sort(self, other):
        if self<other:
            return [self,other]
        else:
            return [other,self]

        

    

    
c1 = Circle(3)
c2 = Circle(4)

print( c1.get_radius() )
print( c1.get_diameter() )
print( c1.get_area() )

print(c1)
print(str(c1))

print(c1+c2)
print(c1<c2)
print(c1>c2)
print(c1==c2)

print(c2.sort(c1))

