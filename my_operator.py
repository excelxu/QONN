# import pennylane as qml
class RBS(Operator):

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def RBS(self):
