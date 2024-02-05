


import math
#ststische Methoden
class Circle:


    def __init__(self, _radius):
        self._radius

       

    def area(self):
        return math.pi * self._radius ** 2

    def get_radius(self):
        return self._radius

    def set_radius(self, radius):
        if radius >= 0:
            self.radius = radius

c = Circle(5)
print(c.area())
c.set_radius = 20
print(c.get_radius())
