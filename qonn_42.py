import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

dev = qml.device('default.mixed', wires=[0,1,2,3,'aux']) # 定义量子模拟器,包含自动微分模式

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
    basis10000 = np.zeros(2**5)
    basis01000 = np.zeros(2**5)
    basis00100 = np.zeros(2**5)
    basis00010 = np.zeros(2**5)
    basis00001 = np.zeros(2**5)
    basis10000[2**4] = 1.0
    basis01000[2**3] = 1.0
    basis00100[2**2] = 1.0
    basis00010[2**1] = 1.0
    basis00001[2**0] = 1.0
    state =  data[0]*basis10000\
            +data[1]*basis01000\
            +data[2]*basis00100\
            +data[3]*basis00010\
            +data[4]*basis00001 # 第五位是辅助量子比特上的振幅，一般为0.2
    return state

# 为了更好地使用qml,初始化量子态必须
# (1)通过生成电路生成
# (2)作为全局变量导入
@qml.qnode(dev,  diff_method="parameter-shift")
def layer_42_on_3(params):
    # 装载初始数据
    # 拥有五条线路，最下面一条是辅助量子比特
    qml.QubitStateVector(ini_state
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
    return qml.expval(qml.PauliZ(2))

@qml.qnode(dev,  diff_method="parameter-shift")
def layer_42_on_4(params):
    # 装载初始数据
    # 拥有五条线路，最下面一条是辅助量子比特
    qml.QubitStateVector(ini_state
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
    return qml.expval(qml.PauliZ(3))


if __name__ == '__main__':

    # 测试🐔
    # 用于导入量子态数据的全局变量
    global ini_state
    # 读取excel中数据
    ab_train = pd.read_excel('IrisData/ab20_train_100.xlsx') # pd默认将第一行作为标题
    bc_train = pd.read_excel('IrisData/bc20_train_100.xlsx')
    ab_test = pd.read_excel('IrisData/irisAB_test_80.xlsx')
    bc_test = pd.read_excel('IrisData/irisBC_test_80.xlsx')

    print(ab_train)
    # 量子网络的初始化
    param = np.array([1, 0.1, 3, 3, 0]) # 初始化参数
    print(qml.draw(layer_42_on_3)(param)) # 绘制网络框架

    for i in range(1):
        ini_state = data_to_state([0, 0.8, 0.6, 0, 0])  # 自定义的初始态
        print(layer_42_on_3(param))
        grad = qml.gradients.param_shift(layer_42_on_3)(param)
        param = param - grad* 0.2

    print(param)







# # 确定计算基底的矢量形式
#     @qml.qnode(dev)
#     def basis():
#         qml.PauliX(wires= 2 )
#         return qml.state()
#     s = basis()
#     print(s)


