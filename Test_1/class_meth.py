

# Klassen Attribute
class Person:

    counter = 0

    def __init__(self, first_name, last_name, age):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        Person.counter += 1

    def print_details(self):
        print(self.first_name + " " + self.last_name + ": ALter"+ str(self.age))


person1 = Person('Max', 'Mustermann', 19)
person2 = Person('Paul', 'Mustermann', 23)

print(Person.counter)
