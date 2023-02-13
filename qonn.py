# 这是一个示例 Python 脚本。
import os
import pennylane as qml
from pennylane import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

'''
    创建QONN类
    （1）RBS门
    （2）自动生成金字塔形状的电路
    （3）电路中间态可以读取
    （4）使用参数位移法则进行梯度升级
    （ ）使用传统的方法进行升级
'''
class qonn():
    def __init__(self, in_dim, out_dim, params,type = 'default.qubit'):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = max(self.in_dim,self.out_dim) #获取最大的维度，作为整个电路的qubits个数
        self.params = params
        # 量子模拟器设置（dev）
        dev = qml.device(type, wires=self.dim)

    def RBS(self, on_wire, theta):
        qml.Hadamard(wires=on_wire[0])
        qml.Hadamard(wires=on_wire[1])
        qml.CZ(wires=on_wire)
        qml.RY(theta/2, wires=on_wire[0])
        qml.RY(-theta/2, wires=on_wire[1])
        qml.CZ(wires=on_wire)
        qml.Hadamard(wires=on_wire[0])
        qml.Hadamard(wires=on_wire[1])

    # 考虑拓展性
    def layer(self):

        pass
    # 先出个结果再说吧！
    # 调用这个函数直接生成一段量子电路




    def layer_42(self):
        dev_42 = qml.device(type, wires = 4)
        @qml.device(dev_42)
        def circuit_42():






def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 ⌘F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')
    dev = qml.device('default.qubit', wires=2)  # 选择量子模拟器
    m = qonn(2,2)
    @qml.qnode(dev)
    def cir():
        m.RBS((0,1), np.pi)
        qml.CNOT((0,1))
        m.RBS((0,1), np.pi/2)
        return qml.state()



