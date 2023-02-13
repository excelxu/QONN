import os
import pennylane as qml
from pennylane import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

dev = qml.device('default.qubit', wires=4) # 定义量子模拟器

def RBS(on_wire, theta):
    if type(on_wire) is tuple:
        qml.Hadamard(wires=on_wire[0])
        qml.Hadamard(wires=on_wire[1])
        qml.CZ(wires=on_wire)
        qml.RY(theta / 2, wires=on_wire[0])
        qml.RY(-theta / 2, wires=on_wire[1])
        qml.CZ(wires=on_wire)
        qml.Hadamard(wires=on_wire[0])
        qml.Hadamard(wires=on_wire[1])
    else:
        raise('RBS作用线路编号为元组形式')

@qml.qnode(dev)
def layer_42(in_state, params):
    # 装载初始数据
    qml.QubitStateVector([]
                         ,wires=range(4))
    # 金字塔电路主体
    RBS((0, 1), params[0])
    qml.Snapshot("zeta_0")
    RBS((1, 2), params[1])
    qml.Snapshot("zeta_1")
    RBS((0, 1), params[2])
    qml.Snapshot("zeta_2")
    RBS((2, 3), params[3])
    qml.Snapshot("zeta_3")
    RBS((1, 2), params[4])
    return qml.state()


if __name__ == '__main__':






    # 测试🐔
    @qml.qnode(dev)
    def cir():
        qml.CZ(wires=[1, 2])
        RBS([0, 1], 1)
        return qml.state()
    print(qml.draw(cir)())
    print('hi')
    print(qml.draw(layer_42)())
