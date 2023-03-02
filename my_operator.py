'''
参考文章 QONN in pennylanes
'''
import numpy as np
import pennylane as qml
from pennylane.operation import Operation

class RBS(Operation):
    num_params = 1
    num_wires = 2
    par_domain = "R"

    def __init__(self, theta, wires):
        self.theta = theta
        super().__init__(theta, wires=wires)

    def expand(self):
    # expand 定义了此量子操作的展开形式
        qml.Hadamard(wires=self.wires[0])
        qml.Hadamard(wires=self.wires[1])
        qml.CZ(wires=[self.wires[0],self.wires[1]])
        qml.RY(self.theta/2, wires=self.wires[0])
        qml.RY(-self.theta/2, wires=self.wires[1])
        qml.CZ(wires=[self.wires[0],self.wires[1]])
        qml.Hadamard(wires=self.wires[0])
        qml.Hadamard(wires=self.wires[1])




