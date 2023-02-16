import os
import pennylane as qml
from pennylane import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

dev = qml.device('default.qubit', wires=[0,1,2,3,'aux']) # 定义量子模拟器

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

def data_to_state(data):
    # 设定计算基的矢量模式
    basis10000 = basis01000 = basis00100 =  basis00010 = basis00001 = np.zeros(2**5)
    basis10000[2**4] = 1.0
    basis01000[2**3] = 1.0
    basis00100[2**2] = 1.0
    basis00010[2**1] = 1.0
    basis00001[2**0] = 1.0
    state =  data[0] * basis10000 \
            +data[1] * basis01000 \
            +data[2] * basis00100 \
            +data[3] * basis00010 \
            +data[4] * basis00001 # 第五位是辅助量子比特上的振幅，一般为0.2
    return state

@qml.qnode(dev)
def layer_42(in_state, params):
    # 装载初始数据
    # 拥有五条线路，最下面一条是辅助量子比特
    qml.QubitStateVector(in_state
                        ,wires=[0,1,2,3,'aux'])
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
    ini_state = np.zeros(2**5)
    ini_state[0] = 1.
    s = layer_42(ini_state, [0,0,0,0,0])
    # print(s)
    # print(qml.draw(layer_42)(ini_state, [0,0,0,0,0]))



# # 确定计算基底的矢量形式
#     @qml.qnode(dev)
#     def basis():
#         qml.PauliX(wires= 2 )
#         return qml.state()
#     s = basis()
#     print(s)


