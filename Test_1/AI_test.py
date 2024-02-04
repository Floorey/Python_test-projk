import numpy

array_1 = numpy.array([1,2,3,4,5,6])
array_3 = numpy.array([1,2,3,4,5,6])



array_2 = array_1 + array_3

print(array_2)
print(type(array_2))


def car_init(self, brand):
    self.brand = brand


def drvie(self):
    print(f'{self.brand}: drive()')


Car = type('Car', (),{
    '__init__': car_init,
    'drive': drvie
})

car = Car('Volvo')
car.drive()








values = ('a1', 'b2', 'c2')

print(dict(values))