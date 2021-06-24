import queue
class Animal:
    def __init__(self):
        print("Animal : ")
class Perro(Animal):
    def __init__(self):
        super().__init__()
        print("Perro")

ggm = Perro()
print(isinstance(ggm, Perro))

