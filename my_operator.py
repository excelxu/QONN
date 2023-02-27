'''
参考文章 QONN in pennylanes
'''

import pennylane as qml
from pennylane.operation import Operation

class RBS(Operation):
    num_params = 1
    num_wires = 1
    par_domain = "R"

    def __init__(self, theta, wires):
        super().__init__(theta, wires=wires)

    def expand(self):
        qml.RZ(self.parameters[0], wires=self.wires)
